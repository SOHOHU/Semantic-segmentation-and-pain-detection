"""
EMA (Exponential Moving Average) 模块
用于模型权重的指数移动平均，提高模型泛化能力和稳定性
"""
import torch
import torch.nn as nn
from copy import deepcopy


class ModelEMA:
    """
    模型指数移动平均
    在训练过程中维护模型参数的移动平均版本
    """
    def __init__(self, model, decay=0.9999, device=None):
        """
        Args:
            model: 要应用EMA的模型
            decay: 衰减率,通常设置为0.999-0.9999
            device: 设备
        """
        # 创建EMA模型
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        
        if self.device is not None:
            self.ema.to(device)
        
        # 冻结EMA模型的参数
        for param in self.ema.parameters():
            param.requires_grad_(False)
        
        self.updates = 0

    def update(self, model):
        """
        更新EMA参数
        Args:
            model: 当前训练的模型
        """
        with torch.no_grad():
            self.updates += 1
            d = self.decay
            
            # 动态调整decay
            # 在训练初期使用较小的decay,后期使用较大的decay
            d = min(d, (1 + self.updates) / (10 + self.updates))
            
            # 获取模型状态字典
            msd = model.state_dict()
            esd = self.ema.state_dict()
            
            # 更新EMA参数
            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """
        更新EMA模型的属性
        """
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(self.ema, k, v)

    def __call__(self, *args, **kwargs):
        """
        使用EMA模型进行推理
        """
        return self.ema(*args, **kwargs)

    def state_dict(self):
        """
        返回EMA模型的状态字典
        """
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        """
        加载EMA模型的状态字典
        """
        self.ema.load_state_dict(state_dict)


class WarmupScheduler:
    """
    学习率预热调度器
    在训练初期逐渐增加学习率
    """
    def __init__(self, optimizer, warmup_epochs, base_lr, warmup_lr_start=1e-6, method='linear'):
        """
        Args:
            optimizer: 优化器
            warmup_epochs: 预热轮数
            base_lr: 基础学习率
            warmup_lr_start: 预热起始学习率
            method: 预热方法 ('linear' or 'exp')
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_lr_start = warmup_lr_start
        self.method = method

    def step(self, epoch):
        """
        根据当前epoch更新学习率
        """
        if epoch < self.warmup_epochs:
            if self.method == 'linear':
                lr = self.warmup_lr_start + (self.base_lr - self.warmup_lr_start) * epoch / self.warmup_epochs
            elif self.method == 'exp':
                lr = self.warmup_lr_start * ((self.base_lr / self.warmup_lr_start) ** (epoch / self.warmup_epochs))
            else:
                lr = self.base_lr
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr


class CosineAnnealingWarmupRestarts:
    """
    余弦退火学习率调度器 + Warmup + Restarts
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, eta_min=0):
        """
        Args:
            optimizer: 优化器
            T_0: 第一个重启周期
            T_mult: 每次重启后周期的乘数
            eta_max: 最大学习率
            T_up: warmup步数
            gamma: 每次重启后最大学习率的衰减因子
            eta_min: 最小学习率
        """
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_up = T_up
        self.gamma = gamma
        self.eta_min = eta_min
        
        self.T_cur = 0
        self.T_i = T_0
        self.cycle = 0

    def step(self, epoch=None):
        """
        更新学习率
        """
        if epoch is None:
            epoch = self.T_cur + 1
            self.T_cur = self.T_cur + 1
        else:
            self.T_cur = epoch
        
        if self.T_cur >= self.T_i:
            self.cycle += 1
            self.T_cur = self.T_cur - self.T_i
            self.T_i = self.T_i * self.T_mult
        
        if self.T_cur < self.T_up:
            # Warmup阶段
            lr = (self.eta_max - self.eta_min) * self.T_cur / self.T_up + self.eta_min
        else:
            # 余弦退火阶段
            lr = self.eta_min + (self.eta_max * (self.gamma ** self.cycle) - self.eta_min) * \
                 (1 + np.cos(np.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class GradientAccumulator:
    """
    梯度累积器
    用于在小batch size下模拟大batch size的效果
    """
    def __init__(self, model, optimizer, accumulation_steps=1, scaler=None):
        """
        Args:
            model: 模型
            optimizer: 优化器
            accumulation_steps: 累积步数
            scaler: 混合精度训练的scaler
        """
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.scaler = scaler
        self.step_count = 0

    def step(self, loss):
        """
        执行一步梯度累积
        Args:
            loss: 当前步的损失
        Returns:
            是否执行了参数更新
        """
        # 归一化损失
        loss = loss / self.accumulation_steps
        
        # 反向传播
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.step_count += 1
        
        # 累积足够步数后更新参数
        if self.step_count % self.accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            return True
        
        return False


class LookAhead:
    """
    Lookahead优化器
    参考: https://arxiv.org/abs/1907.08610
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        """
        Args:
            optimizer: 基础优化器
            k: 更新慢权重的频率
            alpha: 慢权重更新的步长
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = {}
        
        # 保存慢权重
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p] = {}
                param_state['slow_param'] = torch.zeros_like(p.data)
                param_state['slow_param'].copy_(p.data)
        
        self.step_count = 0

    def step(self, closure=None):
        """
        执行优化步骤
        """
        loss = self.optimizer.step(closure)
        self.step_count += 1
        
        if self.step_count % self.k == 0:
            # 更新慢权重
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    param_state['slow_param'] += self.alpha * (p.data - param_state['slow_param'])
                    p.data.copy_(param_state['slow_param'])
        
        return loss

    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()

    def state_dict(self):
        """返回状态字典"""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.optimizer.load_state_dict(state_dict)


import numpy as np


class PolyLRScheduler:
    """
    多项式学习率衰减调度器
    常用于语义分割任务
    """
    def __init__(self, optimizer, max_iter, power=0.9, min_lr=1e-6):
        """
        Args:
            optimizer: 优化器
            max_iter: 最大迭代次数
            power: 多项式的幂
            min_lr: 最小学习率
        """
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.power = power
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, cur_iter):
        """
        更新学习率
        """
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr = max(self.base_lrs[i] * (1 - cur_iter / self.max_iter) ** self.power, self.min_lr)
            param_group['lr'] = lr


class StochasticWeightAveraging:
    """
    随机权重平均 (SWA)
    在训练后期对多个检查点的权重进行平均
    """
    def __init__(self, model, swa_start=75, swa_freq=5):
        """
        Args:
            model: 模型
            swa_start: 开始SWA的epoch
            swa_freq: SWA的频率
        """
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_model = None
        self.swa_n = 0

    def update(self, epoch):
        """
        更新SWA模型
        """
        if epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_freq == 0:
            if self.swa_model is None:
                self.swa_model = deepcopy(self.model)
                self.swa_n = 1
            else:
                # 更新SWA权重
                with torch.no_grad():
                    for swa_param, param in zip(self.swa_model.parameters(), self.model.parameters()):
                        swa_param.data = (swa_param.data * self.swa_n + param.data) / (self.swa_n + 1)
                self.swa_n += 1

    def get_swa_model(self):
        """
        获取SWA模型
        """
        return self.swa_model if self.swa_model is not None else self.model


class AdaptiveGradientClipping:
    """
    自适应梯度裁剪
    参考NFNet论文
    """
    def __init__(self, clip_factor=0.01, eps=1e-3):
        """
        Args:
            clip_factor: 裁剪因子
            eps: 防止除零的小常数
        """
        self.clip_factor = clip_factor
        self.eps = eps

    def __call__(self, parameters):
        """
        对参数进行自适应梯度裁剪
        """
        parameters = list(parameters)
        for p in parameters:
            if p.grad is None:
                continue
            
            # 计算参数范数和梯度范数
            param_norm = torch.norm(p.detach(), 2)
            grad_norm = torch.norm(p.grad.detach(), 2)
            
            # 计算裁剪系数
            max_norm = param_norm * self.clip_factor
            trigger = grad_norm > max_norm
            
            if trigger:
                # 裁剪梯度
                clipped_grad = p.grad * (max_norm / (grad_norm + self.eps))
                p.grad.detach().copy_(clipped_grad)


