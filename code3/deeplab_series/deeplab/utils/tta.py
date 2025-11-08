"""
测试时增强 (Test Time Augmentation) 和后处理模块
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class TestTimeAugmentation:
    """
    测试时增强 - 通过多种变换提高预测精度
    """
    def __init__(self, 
                 use_flip=True, 
                 use_multiscale=True,
                 scales=[0.75, 1.0, 1.25, 1.5],
                 use_rotate=False,
                 angles=[0, 90, 180, 270]):
        """
        Args:
            use_flip: 是否使用翻转增强
            use_multiscale: 是否使用多尺度增强
            scales: 多尺度的缩放比例
            use_rotate: 是否使用旋转增强
            angles: 旋转角度
        """
        self.use_flip = use_flip
        self.use_multiscale = use_multiscale
        self.scales = scales
        self.use_rotate = use_rotate
        self.angles = angles

    def __call__(self, model, image, cuda=True):
        """
        对图像进行TTA预测
        Args:
            model: 模型
            image: 输入图像 tensor [1, C, H, W]
            cuda: 是否使用cuda
        Returns:
            predictions: 预测结果 [H, W]
        """
        predictions = []
        
        # 原始预测
        with torch.no_grad():
            pred = model(image)[0]
            pred = F.softmax(pred, dim=0)
            predictions.append(pred.cpu().numpy())
        
        # 水平翻转
        if self.use_flip:
            flipped = torch.flip(image, [3])
            with torch.no_grad():
                pred = model(flipped)[0]
                pred = F.softmax(pred, dim=0)
                pred = torch.flip(pred, [2])
                predictions.append(pred.cpu().numpy())
        
        # 多尺度
        if self.use_multiscale:
            h, w = image.shape[2:]
            for scale in self.scales:
                if scale == 1.0:
                    continue
                
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = F.interpolate(image, size=(new_h, new_w), 
                                      mode='bilinear', align_corners=True)
                
                with torch.no_grad():
                    pred = model(scaled)[0]
                    pred = F.softmax(pred, dim=0)
                    pred = F.interpolate(pred.unsqueeze(0), size=(h, w), 
                                        mode='bilinear', align_corners=True)[0]
                    predictions.append(pred.cpu().numpy())
                
                # 翻转 + 多尺度
                if self.use_flip:
                    flipped_scaled = torch.flip(scaled, [3])
                    with torch.no_grad():
                        pred = model(flipped_scaled)[0]
                        pred = F.softmax(pred, dim=0)
                        pred = torch.flip(pred, [2])
                        pred = F.interpolate(pred.unsqueeze(0), size=(h, w), 
                                            mode='bilinear', align_corners=True)[0]
                        predictions.append(pred.cpu().numpy())
        
        # 旋转
        if self.use_rotate:
            for angle in self.angles:
                if angle == 0:
                    continue
                
                # 旋转图像
                rotated = self._rotate_tensor(image, angle)
                
                with torch.no_grad():
                    pred = model(rotated)[0]
                    pred = F.softmax(pred, dim=0)
                    # 反向旋转预测结果
                    pred = self._rotate_tensor(pred.unsqueeze(0), -angle)[0]
                    predictions.append(pred.cpu().numpy())
        
        # 平均所有预测
        final_pred = np.mean(predictions, axis=0)
        final_pred = np.argmax(final_pred, axis=0)
        
        return final_pred

    def _rotate_tensor(self, tensor, angle):
        """
        旋转tensor
        Args:
            tensor: [B, C, H, W]
            angle: 旋转角度
        """
        if angle == 90:
            return torch.rot90(tensor, k=1, dims=[2, 3])
        elif angle == 180:
            return torch.rot90(tensor, k=2, dims=[2, 3])
        elif angle == 270:
            return torch.rot90(tensor, k=3, dims=[2, 3])
        elif angle == -90:
            return torch.rot90(tensor, k=-1, dims=[2, 3])
        else:
            return tensor


class SlidingWindowInference:
    """
    滑动窗口推理 - 用于处理大尺寸图像
    """
    def __init__(self, window_size=(512, 512), overlap=0.25):
        """
        Args:
            window_size: 窗口大小
            overlap: 重叠率
        """
        self.window_size = window_size
        self.overlap = overlap

    def __call__(self, model, image, num_classes, cuda=True):
        """
        使用滑动窗口进行推理
        Args:
            model: 模型
            image: 输入图像 [C, H, W] numpy array
            num_classes: 类别数
            cuda: 是否使用cuda
        Returns:
            prediction: 预测结果 [H, W]
        """
        c, h, w = image.shape
        window_h, window_w = self.window_size
        
        # 计算步长
        stride_h = int(window_h * (1 - self.overlap))
        stride_w = int(window_w * (1 - self.overlap))
        
        # 创建预测累积器
        pred_sum = np.zeros((num_classes, h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        # 滑动窗口
        for y in range(0, h - window_h + 1, stride_h):
            for x in range(0, w - window_w + 1, stride_w):
                # 确保不超出边界
                y_end = min(y + window_h, h)
                x_end = min(x + window_w, w)
                y = max(0, y_end - window_h)
                x = max(0, x_end - window_w)
                
                # 提取窗口
                window = image[:, y:y_end, x:x_end]
                
                # 转换为tensor
                window_tensor = torch.from_numpy(window).unsqueeze(0).float()
                if cuda:
                    window_tensor = window_tensor.cuda()
                
                # 预测
                with torch.no_grad():
                    pred = model(window_tensor)[0]
                    pred = F.softmax(pred, dim=0).cpu().numpy()
                
                # 累积预测
                pred_sum[:, y:y_end, x:x_end] += pred
                count_map[y:y_end, x:x_end] += 1
        
        # 处理边缘
        if h % stride_h != 0:
            y = h - window_h
            for x in range(0, w - window_w + 1, stride_w):
                x_end = min(x + window_w, w)
                x = max(0, x_end - window_w)
                
                window = image[:, y:y+window_h, x:x_end]
                window_tensor = torch.from_numpy(window).unsqueeze(0).float()
                if cuda:
                    window_tensor = window_tensor.cuda()
                
                with torch.no_grad():
                    pred = model(window_tensor)[0]
                    pred = F.softmax(pred, dim=0).cpu().numpy()
                
                pred_sum[:, y:y+window_h, x:x_end] += pred
                count_map[y:y+window_h, x:x_end] += 1
        
        if w % stride_w != 0:
            x = w - window_w
            for y in range(0, h - window_h + 1, stride_h):
                y_end = min(y + window_h, h)
                y = max(0, y_end - window_h)
                
                window = image[:, y:y_end, x:x+window_w]
                window_tensor = torch.from_numpy(window).unsqueeze(0).float()
                if cuda:
                    window_tensor = window_tensor.cuda()
                
                with torch.no_grad():
                    pred = model(window_tensor)[0]
                    pred = F.softmax(pred, dim=0).cpu().numpy()
                
                pred_sum[:, y:y_end, x:x+window_w] += pred
                count_map[y:y_end, x:x+window_w] += 1
        
        # 平均预测
        pred_sum = pred_sum / (count_map + 1e-10)
        final_pred = np.argmax(pred_sum, axis=0)
        
        return final_pred


class CRFPostProcessing:
    """
    CRF (条件随机场) 后处理
    用于细化分割边界
    """
    def __init__(self, max_iter=10, pos_w=3, pos_xy_std=3, 
                 bi_w=5, bi_xy_std=50, bi_rgb_std=5):
        """
        Args:
            max_iter: CRF最大迭代次数
            pos_w: 位置权重
            pos_xy_std: 位置标准差
            bi_w: 双边权重
            bi_xy_std: 双边空间标准差
            bi_rgb_std: 双边颜色标准差
        """
        self.max_iter = max_iter
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
            self.dcrf = dcrf
            self.unary_from_softmax = unary_from_softmax
            self.available = True
        except ImportError:
            print("Warning: pydensecrf not installed. CRF post-processing will be disabled.")
            self.available = False

    def __call__(self, image, probabilities):
        """
        应用CRF后处理
        Args:
            image: 原始图像 [H, W, 3] numpy array (RGB, 0-255)
            probabilities: 类别概率 [C, H, W] numpy array
        Returns:
            refined: 细化后的预测 [H, W]
        """
        if not self.available:
            return np.argmax(probabilities, axis=0)
        
        h, w = image.shape[:2]
        n_classes = probabilities.shape[0]
        
        # 创建CRF模型
        d = self.dcrf.DenseCRF2D(w, h, n_classes)
        
        # Unary potential
        unary = self.unary_from_softmax(probabilities)
        d.setUnaryEnergy(unary)
        
        # Pairwise potentials
        # 位置项
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        
        # 双边项
        d.addPairwiseBilateral(sxy=self.bi_xy_std, srgb=self.bi_rgb_std, 
                               rgbim=image.astype(np.uint8), compat=self.bi_w)
        
        # 推理
        Q = d.inference(self.max_iter)
        Q = np.array(Q).reshape((n_classes, h, w))
        
        return np.argmax(Q, axis=0)


class MorphologicalPostProcessing:
    """
    形态学后处理
    用于去除噪声和填充孔洞
    """
    def __init__(self, kernel_size=5, 
                 use_opening=True, 
                 use_closing=True,
                 use_fill_holes=True):
        """
        Args:
            kernel_size: 形态学操作的核大小
            use_opening: 是否使用开运算(去除小噪声)
            use_closing: 是否使用闭运算(填充小孔洞)
            use_fill_holes: 是否填充孔洞
        """
        self.kernel_size = kernel_size
        self.use_opening = use_opening
        self.use_closing = use_closing
        self.use_fill_holes = use_fill_holes
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                (kernel_size, kernel_size))

    def __call__(self, mask):
        """
        应用形态学后处理
        Args:
            mask: 分割掩码 [H, W] numpy array
        Returns:
            processed: 处理后的掩码 [H, W]
        """
        mask = mask.astype(np.uint8)
        
        # 对每个类别分别处理
        unique_classes = np.unique(mask)
        result = np.zeros_like(mask)
        
        for class_id in unique_classes:
            if class_id == 0:  # 跳过背景
                continue
            
            # 提取当前类别的掩码
            class_mask = (mask == class_id).astype(np.uint8)
            
            # 开运算 - 去除小的噪声点
            if self.use_opening:
                class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, self.kernel)
            
            # 闭运算 - 填充小的孔洞
            if self.use_closing:
                class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, self.kernel)
            
            # 填充孔洞
            if self.use_fill_holes:
                # 使用floodfill填充孔洞
                h, w = class_mask.shape
                flood_mask = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(class_mask, flood_mask, (0, 0), 255)
                class_mask = cv2.bitwise_not(class_mask)
            
            # 合并结果
            result[class_mask > 0] = class_id
        
        # 恢复背景
        result[mask == 0] = 0
        
        return result


class EnsemblePredictor:
    """
    模型集成预测器
    """
    def __init__(self, models, weights=None):
        """
        Args:
            models: 模型列表
            weights: 每个模型的权重,默认为均等权重
        """
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights

    def __call__(self, image, cuda=True):
        """
        集成预测
        Args:
            image: 输入图像 tensor [1, C, H, W]
            cuda: 是否使用cuda
        Returns:
            prediction: 预测结果 [H, W]
        """
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(image)[0]
                pred = F.softmax(pred, dim=0)
                predictions.append(pred.cpu().numpy())
        
        # 加权平均
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += pred * weight
        
        final_pred = np.argmax(weighted_pred, axis=0)
        return final_pred


class AdaptiveThresholding:
    """
    自适应阈值后处理
    根据预测置信度动态调整阈值
    """
    def __init__(self, base_threshold=0.5, confidence_margin=0.2):
        """
        Args:
            base_threshold: 基础阈值
            confidence_margin: 置信度边界
        """
        self.base_threshold = base_threshold
        self.confidence_margin = confidence_margin

    def __call__(self, probabilities):
        """
        应用自适应阈值
        Args:
            probabilities: 类别概率 [C, H, W] numpy array
        Returns:
            prediction: 预测结果 [H, W]
        """
        max_probs = np.max(probabilities, axis=0)
        prediction = np.argmax(probabilities, axis=0)
        
        # 低置信度区域设为背景
        low_confidence = max_probs < self.base_threshold
        prediction[low_confidence] = 0
        
        return prediction


class BoundaryRefinement:
    """
    边界细化
    对分割边界进行精细化处理
    """
    def __init__(self, boundary_width=5):
        """
        Args:
            boundary_width: 边界宽度
        """
        self.boundary_width = boundary_width

    def __call__(self, image, prediction, probabilities):
        """
        细化边界
        Args:
            image: 原始图像 [H, W, 3]
            prediction: 初始预测 [H, W]
            probabilities: 类别概率 [C, H, W]
        Returns:
            refined: 细化后的预测 [H, W]
        """
        # 检测边界
        edges = self._detect_edges(prediction)
        
        # 在边界区域使用更精细的判断
        refined = prediction.copy()
        
        for i in range(self.boundary_width):
            dilated_edges = cv2.dilate(edges.astype(np.uint8), 
                                       np.ones((3, 3), np.uint8), 
                                       iterations=1)
            
            # 在边界区域重新判断
            boundary_mask = dilated_edges > 0
            refined[boundary_mask] = np.argmax(probabilities[:, boundary_mask], axis=0)
            
            edges = dilated_edges
        
        return refined

    def _detect_edges(self, mask):
        """
        检测边界
        """
        # 使用Sobel算子检测边界
        sobel_x = cv2.Sobel(mask.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(mask.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = (edges > 0).astype(np.uint8)
        
        return edges


