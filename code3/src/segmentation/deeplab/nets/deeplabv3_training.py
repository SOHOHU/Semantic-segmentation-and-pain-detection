import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def OHEM_CE_Loss(inputs, target, cls_weights, num_classes=21, thresh=0.7, min_kept=100000):
    """
    在线困难样本挖掘交叉熵损失
    Online Hard Example Mining (OHEM) Cross Entropy Loss
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)
    
    # 计算每个像素的损失
    valid_mask = temp_target != num_classes
    valid_inputs = temp_inputs[valid_mask]
    valid_target = temp_target[valid_mask]
    
    # 计算损失
    loss = F.cross_entropy(valid_inputs, valid_target, weight=cls_weights, reduction='none')
    
    # OHEM: 选择困难样本
    loss_sorted, _ = torch.sort(loss, descending=True)
    
    # 动态确定保留样本数量
    if loss_sorted.numel() > min_kept:
        threshold_idx = min(min_kept, loss_sorted.numel())
        threshold = loss_sorted[threshold_idx]
        
        # 选择损失大于阈值的样本
        keep_mask = loss >= threshold
        ohem_loss = loss[keep_mask].mean()
    else:
        ohem_loss = loss.mean()
    
    return ohem_loss


def Boundary_Loss(inputs, target, num_classes=21):
    """
    边界损失函数
    帮助模型更好地学习物体边界
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    
    # 计算边界
    target_np = target.cpu().numpy()
    boundary_targets = []
    
    for i in range(n):
        # 计算每个样本的边界
        target_single = target_np[i]
        boundaries = np.zeros_like(target_single, dtype=np.float32)
        
        # 对每个类别计算边界
        for cls in range(num_classes):
            mask = (target_single == cls).astype(np.uint8)
            if mask.sum() > 0:
                # 使用形态学操作提取边界
                kernel = np.ones((3, 3), np.uint8)
                import cv2
                eroded = cv2.erode(mask, kernel, iterations=1)
                boundary = mask - eroded
                boundaries += boundary
        
        boundaries = np.clip(boundaries, 0, 1)
        boundary_targets.append(boundaries)
    
    boundary_targets = torch.from_numpy(np.array(boundary_targets)).float().to(inputs.device)
    
    # 计算预测的边界
    pred_softmax = F.softmax(inputs, dim=1)
    pred_max, _ = torch.max(pred_softmax, dim=1)
    pred_boundary = 1 - pred_max
    
    # 计算边界损失 (MSE)
    boundary_loss = F.mse_loss(pred_boundary, boundary_targets)
    
    return boundary_loss


def Lovasz_Softmax_Loss(inputs, target, num_classes=21):
    """
    Lovasz-Softmax损失
    对IoU进行优化的损失函数
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    
    probas = F.softmax(inputs, dim=1)
    
    total_loss = 0
    cnt = 0
    
    # 对每个类别计算Lovasz损失
    for cls in range(num_classes):
        cls_prob = probas[:, cls, :, :].contiguous().view(-1)
        cls_target = (target == cls).float().view(-1)
        
        if cls_target.sum() > 0:
            errors = (cls_target - cls_prob).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            cls_target_sorted = cls_target[perm]
            
            inter = cls_target_sorted.sum() - cls_target_sorted.cumsum(0)
            union = cls_target_sorted.sum() + (1 - cls_target_sorted).cumsum(0)
            iou = 1 - inter / union
            
            p = len(cls_target_sorted)
            if p > 1:
                iou[1:p] = iou[1:p] - iou[0:-1]
            
            loss = torch.dot(errors_sorted, iou)
            total_loss += loss
            cnt += 1
    
    return total_loss / max(cnt, 1)


def Tversky_Loss(inputs, target, num_classes=21, alpha=0.7, beta=0.3, smooth=1e-5):
    """
    Tversky损失
    可以通过调整alpha和beta来平衡假阳性和假阴性
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    
    # 转换为one-hot编码
    target_one_hot = torch.zeros(n, num_classes, ht, wt).to(inputs.device)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    
    probas = F.softmax(inputs, dim=1)
    
    # 计算Tversky指数
    dims = (0, 2, 3)
    intersection = torch.sum(probas * target_one_hot, dims)
    fps = torch.sum(probas * (1 - target_one_hot), dims)
    fns = torch.sum((1 - probas) * target_one_hot, dims)
    
    tversky = (intersection + smooth) / (intersection + alpha * fps + beta * fns + smooth)
    tversky_loss = 1 - torch.mean(tversky)
    
    return tversky_loss


def Combined_Loss(inputs, target, cls_weights, num_classes=21, 
                  use_ce=True, use_dice=True, use_boundary=True, use_ohem=False,
                  ce_weight=1.0, dice_weight=1.0, boundary_weight=0.5):
    """
    组合损失函数
    结合多种损失以获得更好的性能
    """
    total_loss = 0
    
    if use_ohem and use_ce:
        ce_loss = OHEM_CE_Loss(inputs, target, cls_weights, num_classes)
        total_loss += ce_weight * ce_loss
    elif use_ce:
        ce_loss = CE_Loss(inputs, target, cls_weights, num_classes)
        total_loss += ce_weight * ce_loss
    
    if use_dice:
        # 需要转换target为one-hot格式
        n, c, h, w = inputs.size()
        nt, ht, wt = target.size()
        target_one_hot = torch.zeros(n, ht, wt, num_classes + 1).to(inputs.device)
        target_copy = target.clone()
        target_copy[target_copy >= num_classes] = num_classes
        target_one_hot.scatter_(3, target_copy.unsqueeze(3), 1)
        
        dice_loss = Dice_loss(inputs, target_one_hot)
        total_loss += dice_weight * dice_loss
    
    if use_boundary:
        boundary_loss = Boundary_Loss(inputs, target, num_classes)
        total_loss += boundary_weight * boundary_loss
    
    return total_loss


class ModelEMA:
    """
    模型指数移动平均
    Model Exponential Moving Average
    有助于提升模型的稳定性和泛化能力
    """
    def __init__(self, model, decay=0.9999, device=None):
        self.ema = self._copy_model(model)
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.ema.to(device=device)
        self.ema.eval()
        
        # 冻结EMA模型参数
        for param in self.ema.parameters():
            param.requires_grad = False
    
    def _copy_model(self, model):
        """深拷贝模型"""
        import copy
        return copy.deepcopy(model)
    
    def update(self, model):
        """更新EMA模型"""
        with torch.no_grad():
            # 获取当前训练步数调整后的decay
            for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
                if self.device is not None:
                    model_param = model_param.to(self.device)
                ema_param.copy_(self.decay * ema_param + (1. - self.decay) * model_param)
    
    def update_attr(self, model):
        """更新EMA模型的属性"""
        for k, v in model.__dict__.items():
            if not k.startswith('_') and k != 'training':
                setattr(self.ema, k, v)
