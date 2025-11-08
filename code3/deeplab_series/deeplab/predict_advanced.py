"""
改进版预测脚本 - 集成TTA和后处理优化
主要改进:
1. 测试时增强(TTA) - 多尺度、翻转
2. CRF后处理 - 细化边界
3. 形态学后处理 - 去除噪声
4. 滑动窗口推理 - 处理大尺寸图像
5. 自适应阈值
6. 边界细化
"""
import colorsys
import copy
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.deeplabv3_plus import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image, show_config
from utils.tta import (TestTimeAugmentation, SlidingWindowInference, 
                       CRFPostProcessing, MorphologicalPostProcessing,
                       AdaptiveThresholding, BoundaryRefinement)


class DeeplabV3Advanced(object):
    _defaults = {
        "model_path"        : 'logs_advanced/best_epoch_weights.pth',
        "num_classes"       : 7,
        "backbone"          : "mobilenet",
        "input_shape"       : [512, 512],
        "downsample_factor" : 16,
        "mix_type"          : 1,
        "cuda"              : True,
        
        # 高级预测选项
        "use_tta"           : True,   # 使用测试时增强
        "tta_flip"          : True,   # TTA中使用翻转
        "tta_multiscale"    : True,   # TTA中使用多尺度
        "tta_scales"        : [0.75, 1.0, 1.25],  # TTA的尺度
        
        "use_crf"           : False,  # 使用CRF后处理(需要安装pydensecrf)
        "use_morphology"    : True,   # 使用形态学后处理
        "use_adaptive_threshold" : False,  # 使用自适应阈值
        "use_boundary_refine"    : False,  # 使用边界细化
        
        "use_sliding_window" : False,  # 对于大图使用滑动窗口
        "window_size"        : (512, 512),  # 滑动窗口大小
        "window_overlap"     : 0.25,  # 滑动窗口重叠率
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        # 设置颜色
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        
        # 加载模型
        self.generate()
        
        # 初始化后处理模块
        if self.use_tta:
            self.tta = TestTimeAugmentation(
                use_flip=self.tta_flip,
                use_multiscale=self.tta_multiscale,
                scales=self.tta_scales,
                use_rotate=False
            )
        
        if self.use_crf:
            self.crf = CRFPostProcessing(
                max_iter=10,
                pos_w=3,
                pos_xy_std=3,
                bi_w=5,
                bi_xy_std=50,
                bi_rgb_std=5
            )
        
        if self.use_morphology:
            self.morphology = MorphologicalPostProcessing(
                kernel_size=5,
                use_opening=True,
                use_closing=True,
                use_fill_holes=True
            )
        
        if self.use_adaptive_threshold:
            self.adaptive_thresh = AdaptiveThresholding(
                base_threshold=0.5,
                confidence_margin=0.2
            )
        
        if self.use_boundary_refine:
            self.boundary_refine = BoundaryRefinement(boundary_width=5)
        
        if self.use_sliding_window:
            self.sliding_window = SlidingWindowInference(
                window_size=self.window_size,
                overlap=self.window_overlap
            )
        
        show_config(**self._defaults)

    def generate(self, onnx=False):
        self.net = DeepLab(num_classes=self.num_classes, backbone=self.backbone, 
                          downsample_factor=self.downsample_factor, pretrained=False)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, count=False, name_classes=None):
        """
        改进版图像检测 - 集成TTA和后处理
        """
        # 转换为RGB
        image       = cvtColor(image)
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        
        # 准备图像数据
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            #---------------------------------------------------#
            #   使用TTA或普通预测
            #---------------------------------------------------#
            if self.use_tta:
                # TTA预测
                pr = self.tta(self.net, images, self.cuda)
                # TTA返回的是[H, W]的预测结果
                pr_for_post = pr  # 用于后处理的预测
                
                # 获取概率用于CRF和其他后处理
                outputs = self.net(images)[0]
                probas = F.softmax(outputs, dim=0).cpu().numpy()
                probas = probas[:, int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh),
                                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
                probas = np.transpose(probas, (1, 2, 0))
                probas_resized = cv2.resize(probas, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
                probas_resized = np.transpose(probas_resized, (2, 0, 1))
            else:
                # 普通预测
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh),
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
                
                probas_resized = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
                probas_resized = np.transpose(probas_resized, (2, 0, 1))
                pr_for_post = np.argmax(probas_resized, axis=0)
        
        #---------------------------------------------------#
        #   后处理
        #---------------------------------------------------#
        # 自适应阈值
        if self.use_adaptive_threshold:
            pr_for_post = self.adaptive_thresh(probas_resized)
        
        # CRF后处理
        if self.use_crf:
            image_np = np.array(old_img)
            pr_for_post = self.crf(image_np, probas_resized)
        
        # 形态学后处理
        if self.use_morphology:
            pr_for_post = self.morphology(pr_for_post)
        
        # 边界细化
        if self.use_boundary_refine:
            image_np = np.array(old_img)
            pr_for_post = self.boundary_refine(image_np, pr_for_post, probas_resized)
        
        pr = pr_for_post
        
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
    
        # 生成可视化图像
        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image   = Image.fromarray(np.uint8(seg_img))
            image   = Image.blend(old_img, image, 0.7)
        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image   = Image.fromarray(np.uint8(seg_img))
        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            image = Image.fromarray(np.uint8(seg_img))
        
        return image

    def get_FPS(self, image, test_interval):
        """测试FPS"""
        image       = cvtColor(image)
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh),
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_miou_png(self, image):
        """用于mIoU计算"""
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            if self.use_tta:
                # 使用TTA
                pr = self.tta(self.net, images, self.cuda)
            else:
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh),
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
                pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
                pr = pr.argmax(axis=-1)
        
        # 应用形态学后处理
        if self.use_morphology:
            pr = self.morphology(pr)
        
        image = Image.fromarray(np.uint8(pr))
        return image


if __name__ == "__main__":
    # 使用示例
    deeplab = DeeplabV3Advanced(
        model_path='logs_advanced/best_epoch_weights.pth',
        use_tta=True,
        tta_multiscale=True,
        tta_scales=[0.75, 1.0, 1.25],
        use_morphology=True,
        use_crf=False  # 需要安装pydensecrf
    )
    
    # 读取测试图像
    img_path = 'img/test.jpg'
    image = Image.open(img_path)
    
    # 预测
    print("开始预测(使用TTA和后处理)...")
    start_time = time.time()
    r_image = deeplab.detect_image(image)
    end_time = time.time()
    print(f"预测完成，耗时: {end_time - start_time:.2f}秒")
    
    # 保存结果
    r_image.save('img_out/result_advanced.png')
    print("结果已保存到 img_out/result_advanced.png")


