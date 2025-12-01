"""
改进版mIoU评估脚本 - 支持TTA和后处理
"""
import os
from PIL import Image
from tqdm import tqdm

from predict_advanced import DeeplabV3Advanced
from utils.utils_metrics import compute_mIoU, show_results

'''
使用改进的模型评估mIoU
支持TTA和后处理以获得更准确的评估结果
'''

if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 7
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    name_classes    = ["background","1","2","3","4","5","6"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    
    #-------------------------------------------------------#
    #   是否使用TTA和后处理
    #-------------------------------------------------------#
    use_tta         = True   # 使用测试时增强（推荐）
    use_morphology  = True   # 使用形态学后处理（推荐）
    use_crf         = False  # 使用CRF后处理（可选，需安装pydensecrf）
    
    # TTA配置（评估时建议用较少尺度以加快速度）
    tta_scales      = [1.0, 1.25]  # 评估时用2个尺度即可
    
    #-------------------------------------------------------#
    #   模型路径
    #-------------------------------------------------------#
    # 可以评估3种模型
    model_paths = [
        'logs_advanced/best_epoch_weights.pth',  # 最佳验证损失模型
        # 'logs_advanced/ema_final.pth',          # EMA模型（取消注释以评估）
        # 'logs_advanced/swa_final.pth',          # SWA模型（取消注释以评估）
    ]
    
    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    pred_dir        = "miou_out"
    
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        
        print("="*80)
        print("开始使用改进模型生成预测结果")
        print("="*80)
        print(f"TTA: {'开启' if use_tta else '关闭'}")
        if use_tta:
            print(f"  - 多尺度: {tta_scales}")
        print(f"形态学后处理: {'开启' if use_morphology else '关闭'}")
        print(f"CRF后处理: {'开启' if use_crf else '关闭'}")
        print("="*80)
        
        for model_path in model_paths:
            print(f"\n评估模型: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"  ⚠️ 模型文件不存在，跳过: {model_path}")
                continue
            
            # 创建改进的预测器
            deeplab = DeeplabV3Advanced(
                model_path=model_path,
                num_classes=num_classes,
                use_tta=use_tta,
                tta_multiscale=use_tta,
                tta_scales=tta_scales,
                use_morphology=use_morphology,
                use_crf=use_crf
            )
            
            # 为不同模型创建不同的输出目录
            model_name = os.path.basename(model_path).replace('.pth', '')
            current_pred_dir = os.path.join(pred_dir, model_name)
            if not os.path.exists(current_pred_dir):
                os.makedirs(current_pred_dir)
            
            print(f"生成预测结果到: {current_pred_dir}")
            
            for image_id in tqdm(image_ids):
                image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                image       = Image.open(image_path)
                
                # 获取预测结果（已应用TTA和后处理）
                image       = deeplab.get_miou_png(image)
                
                # 保存预测结果
                image.save(os.path.join(current_pred_dir, image_id + ".png"))
            
            print(f"✅ 预测完成: {len(image_ids)} 张图像")
            
            # 立即计算这个模型的mIoU
            if miou_mode == 0:
                print(f"\n{'='*80}")
                print(f"计算 {model_name} 的mIoU")
                print(f"{'='*80}")
                hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, current_pred_dir, image_ids, num_classes, name_classes)
                print(f"{'='*80}")
                show_results(miou_out_path=current_pred_dir, hist=hist, IoUs=IoUs, PA_Recall=PA_Recall, Precision=Precision, name_classes=name_classes)
                print(f"{'='*80}\n")
        
        print("\n" + "="*80)
        print("所有模型评估完成！")
        print("="*80)

    if miou_mode == 0 or miou_mode == 2:
        # 如果只是计算mIoU（miou_mode == 2），使用默认的pred_dir
        if miou_mode == 2:
            print("计算mIoU。")
            hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  
            show_results(miou_out_path=pred_dir, hist=hist, IoUs=IoUs, PA_Recall=PA_Recall, Precision=Precision, name_classes=name_classes)


