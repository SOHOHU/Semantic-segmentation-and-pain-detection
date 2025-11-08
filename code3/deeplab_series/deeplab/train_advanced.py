import datetime
import os
from functools import partial
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

# 导入新的模块
from utils.advanced_losses import UnifiedLoss, OHEMLoss
from utils.ema import ModelEMA, WarmupScheduler, PolyLRScheduler, StochasticWeightAveraging
from utils.augmentations import apply_mixup_or_cutmix

'''
改进版训练脚本 - 集成所有性能优化
主要改进:
1. 高级数据增强 (Cutout, GridMask, ColorJitter等)
2. 统一损失函数 (CE + Dice + Lovász + Boundary + Tversky)
3. EMA (指数移动平均)
4. 改进的学习率调度器
5. 在线难样本挖掘 (OHEM)
6. 标签平滑
7. MixUp/CutMix
8. 随机权重平均 (SWA)
'''

if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    是否使用Cuda
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed    用于固定随机种子
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #---------------------------------------------------------------------#
    fp16            = True
    #-----------------------------------------------------#
    #   num_classes     训练自己的数据集必须要修改的
    #-----------------------------------------------------#
    num_classes     = 7
    #---------------------------------#
    #   所使用的的主干网络
    #---------------------------------#
    backbone        = "mobilenet"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      是否使用主干网络的预训练权重
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = True
    model_path      = "model_data/deeplab_mobilenetv2.pth"
    #---------------------------------------------------------#
    #   downsample_factor   下采样的倍数
    #---------------------------------------------------------#
    downsample_factor   = 16
    #------------------------------#
    #   输入图片的大小
    #------------------------------#
    input_shape         = [512, 512]
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   训练参数设置
    #----------------------------------------------------------------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50  # 降低冻结epoch以更快进入解冻训练
    Freeze_batch_size   = 12  # 增加batch size
    UnFreeze_Epoch      = 300  # 增加训练轮数
    Unfreeze_batch_size = 6    # 增加batch size
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #   优化器参数
    #------------------------------------------------------------------#
    Init_lr             = 7e-3
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 5e-4  # 降低weight decay
    #------------------------------------------------------------------#
    #   学习率下降方式
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   保存相关
    #------------------------------------------------------------------#
    save_period         = 10
    save_dir            = 'logs_advanced'
    #------------------------------------------------------------------#
    #   评估相关
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 5
    #------------------------------------------------------------------#
    #   数据集路径
    #------------------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    
    #------------------------------------------------------------------#
    #   高级训练选项
    #------------------------------------------------------------------#
    # 损失函数配置
    use_unified_loss    = True  # 使用统一损失函数
    use_ce              = True
    use_dice            = True
    use_lovasz          = True  # 使用Lovász损失
    use_boundary        = False  # 边界损失（计算开销大）
    use_tversky         = False  # Tversky损失
    use_ohem            = False  # OHEM
    use_label_smoothing = True   # 标签平滑
    label_smoothing     = 0.1
    
    # EMA相关
    use_ema             = True   # 使用EMA
    ema_decay           = 0.9999
    
    # SWA相关
    use_swa             = True   # 使用随机权重平均
    swa_start           = 200    # 从第200个epoch开始SWA
    swa_freq            = 5      # 每5个epoch更新一次
    
    # MixUp/CutMix
    use_mixup_cutmix    = True   # 使用MixUp/CutMix
    mixup_alpha         = 0.2
    cutmix_prob         = 0.5
    
    # 高级数据增强
    use_advanced_aug    = True   # 使用高级数据增强
    
    # Warmup
    use_warmup          = True   # 使用学习率预热
    warmup_epochs       = 5
    
    #------------------------------------------------------------------#
    #   类别权重
    #------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------------------#
    #   num_workers
    #------------------------------------------------------------------#
    num_workers         = 4

    seed_everything(seed)
    
    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #----------------------------------------------------#
    #   下载预训练权重
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    model   = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    #----------------------#
    #   初始化EMA
    #----------------------#
    ema = None
    if use_ema and local_rank == 0:
        ema = ModelEMA(model, decay=ema_decay, device=device)
        if local_rank == 0:
            print("EMA initialized with decay:", ema_decay)

    #----------------------#
    #   初始化SWA
    #----------------------#
    swa = None
    if use_swa and local_rank == 0:
        swa = StochasticWeightAveraging(model, swa_start=swa_start, swa_freq=swa_freq)
        if local_rank == 0:
            print("SWA initialized, will start at epoch:", swa_start)

    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    #------------------------------------------------------------------#
    #   混合精度训练
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    
    #----------------------------#
    #   多卡同步Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        print("\n" + "="*50)
        print("高级训练配置:")
        print(f"统一损失函数: {use_unified_loss}")
        if use_unified_loss:
            print(f"  - CE Loss: {use_ce}")
            print(f"  - Dice Loss: {use_dice}")
            print(f"  - Lovász Loss: {use_lovasz}")
            print(f"  - Boundary Loss: {use_boundary}")
            print(f"  - Tversky Loss: {use_tversky}")
            print(f"  - Label Smoothing: {use_label_smoothing} (smoothing={label_smoothing})")
        print(f"EMA: {use_ema} (decay={ema_decay})")
        print(f"SWA: {use_swa} (start={swa_start}, freq={swa_freq})")
        print(f"MixUp/CutMix: {use_mixup_cutmix} (alpha={mixup_alpha}, cutmix_prob={cutmix_prob})")
        print(f"高级数据增强: {use_advanced_aug}")
        print(f"学习率Warmup: {use_warmup} (epochs={warmup_epochs})")
        print("="*50 + "\n")
        
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, 
            Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))
        
    #------------------------------------------------------#
    #   训练
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   选择优化器
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   学习率下降公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   Warmup调度器
        #---------------------------------------#
        warmup_scheduler = None
        if use_warmup:
            warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs, Init_lr_fit, warmup_lr_start=Min_lr_fit)
        
        #---------------------------------------#
        #   数据集
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset   = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path, 
                                        use_advanced_aug=use_advanced_aug)
        val_dataset     = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path, 
                                        use_advanced_aug=False)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        #----------------------#
        #   初始化统一损失函数
        #----------------------#
        if use_unified_loss:
            criterion = UnifiedLoss(
                num_classes=num_classes,
                use_ce=use_ce,
                use_dice=use_dice,
                use_lovasz=use_lovasz,
                use_boundary=use_boundary,
                use_tversky=use_tversky,
                use_ohem=use_ohem,
                use_label_smoothing=use_label_smoothing,
                ce_weight=1.0,
                dice_weight=1.0,
                lovasz_weight=0.5,
                boundary_weight=0.3,
                tversky_weight=0.5,
                smoothing=label_smoothing
            )
            if Cuda:
                criterion = criterion.cuda()

        #----------------------#
        #   eval callback
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   开始训练
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   解冻
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 16
                lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == "xception":
                    lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            # Warmup
            if use_warmup and epoch < warmup_epochs:
                warmup_scheduler.step(epoch)
            else:
                set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # 训练一个epoch (使用统一损失函数)
            if use_unified_loss:
                from utils.utils_fit_advanced import fit_one_epoch_advanced
                fit_one_epoch_advanced(
                    model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, 
                    criterion, num_classes, fp16, scaler, save_period, save_dir, local_rank,
                    use_mixup_cutmix=use_mixup_cutmix, mixup_alpha=mixup_alpha, cutmix_prob=cutmix_prob,
                    ema=ema, cls_weights=cls_weights
                )
            else:
                # 使用原始训练函数
                fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                        epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, 
                        dice_loss=True, focal_loss=False, cls_weights=cls_weights, num_classes=num_classes, 
                        fp16=fp16, scaler=scaler, save_period=save_period, save_dir=save_dir, local_rank=local_rank)
            
            # 更新EMA
            if use_ema and ema is not None:
                ema.update(model)
            
            # 更新SWA
            if use_swa and swa is not None:
                swa.update(epoch)

            if distributed:
                dist.barrier()
        
        # 保存EMA模型
        if use_ema and ema is not None and local_rank == 0:
            print("Saving EMA model...")
            torch.save(ema.state_dict(), os.path.join(save_dir, "ema_final.pth"))
        
        # 保存SWA模型
        if use_swa and swa is not None and local_rank == 0:
            swa_model = swa.get_swa_model()
            if swa_model is not None:
                print("Saving SWA model...")
                torch.save(swa_model.state_dict(), os.path.join(save_dir, "swa_final.pth"))

        if local_rank == 0:
            loss_history.writer.close()


