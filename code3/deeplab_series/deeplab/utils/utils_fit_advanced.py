import os
import torch
from tqdm import tqdm
from utils.utils import get_lr
from utils.utils_metrics import f_score
from utils.augmentations import apply_mixup_or_cutmix


def fit_one_epoch_advanced(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                          epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, 
                          criterion, num_classes, fp16, scaler, save_period, save_dir, local_rank=0,
                          use_mixup_cutmix=False, mixup_alpha=0.2, cutmix_prob=0.5,
                          ema=None, cls_weights=None):
    """
    改进版训练函数 - 集成统一损失函数、MixUp/CutMix、EMA等
    """
    total_loss      = 0
    total_f_score   = 0
    loss_dict_sum   = {}

    val_loss        = 0
    val_f_score     = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    
    model_train.train()
    
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                if cls_weights is not None:
                    weights = torch.from_numpy(cls_weights).cuda(local_rank)
                else:
                    weights = None
        
        #----------------------#
        #   应用MixUp/CutMix
        #----------------------#
        if use_mixup_cutmix and epoch > 10:  # 在训练稳定后再使用
            with torch.no_grad():
                imgs, pngs = apply_mixup_or_cutmix(imgs, pngs, alpha=mixup_alpha, cutmix_prob=cutmix_prob)
        
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            
            #----------------------#
            #   计算损失
            #----------------------#
            loss, loss_dict = criterion(outputs, pngs, cls_weights=weights)

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                
                #----------------------#
                #   计算损失
                #----------------------#
                loss, loss_dict = criterion(outputs, pngs, cls_weights=weights)

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)
                    
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        # 累积各个损失分量
        for key, value in loss_dict.items():
            if key not in loss_dict_sum:
                loss_dict_sum[key] = 0
            loss_dict_sum[key] += value
            
        if local_rank == 0:
            postfix_dict = {
                'total_loss': total_loss / (iteration + 1), 
                'f_score'   : total_f_score / (iteration + 1),
                'lr'        : get_lr(optimizer)
            }
            # 添加各个损失分量到显示
            for key, value in loss_dict_sum.items():
                postfix_dict[key] = value / (iteration + 1)
            
            pbar.set_postfix(**postfix_dict)
            pbar.update(1)

    # 更新EMA
    if ema is not None and local_rank == 0:
        ema.update(model)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        
        # 打印详细的损失信息
        print("\n训练损失详情:")
        for key, value in loss_dict_sum.items():
            print(f"  {key}: {value / epoch_step:.4f}")
        
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                if cls_weights is not None:
                    weights = torch.from_numpy(cls_weights).cuda(local_rank)
                else:
                    weights = None

            #----------------------#
            #   前向传播
            #----------------------#
            outputs     = model_train(imgs)
            
            #----------------------#
            #   计算损失
            #----------------------#
            loss, _ = criterion(outputs, pngs, cls_weights=weights)
            
            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            _f_score    = f_score(outputs, labels)

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
            if local_rank == 0:
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                    'f_score'   : val_f_score / (iteration + 1),
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


