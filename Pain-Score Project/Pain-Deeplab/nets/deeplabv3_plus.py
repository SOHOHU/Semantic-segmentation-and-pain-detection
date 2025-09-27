import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSH(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSH, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1)

        self.conv5x5_1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv5x5_2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1)

        self.conv7x7_2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv7x7_3 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv3x3 = self.conv3x3(x)

        conv5x5_1 = self.conv5x5_1(x)
        conv5x5 = self.conv5x5_2(conv5x5_1)

        conv7x7_2 = self.conv5x5_2(conv5x5_1)
        conv7x7 = self.conv7x7_3(conv7x7_2)

        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        out = F.relu(out, inplace=True)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        # 第一次卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True)
        )
        # 第二次卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True)
        )
        # 第三次卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 8),
            nn.ReLU(inplace=True)
        )
        # 上采样
        self.interpolate1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.interpolate2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # 融合卷积
        self.conv_fuse1 = nn.Conv2d(out_channels * 2 + in_channels, in_channels, kernel_size=1, bias=False)
        self.batchnorm_fuse1 = nn.BatchNorm2d(in_channels)
        self.conv_fuse2 = nn.Conv2d(out_channels * 4 + out_channels * 8, out_channels * 4, kernel_size=1, bias=False)
        self.batchnorm_fuse2 = nn.BatchNorm2d(out_channels * 4)

    def forward(self, x):
        # 浅层特征
        shallow_features = x

        # 第一次卷积
        conv1 = self.conv1(x)
        # 第二次卷积
        conv2 = self.conv2(conv1)
        # 第三次卷积
        conv3 = self.conv3(conv2)

        # 第一次卷积的结果上采样后与浅层特征作1x1卷积融合
        conv1_up = self.interpolate1(conv1)
        fuse1 = torch.cat((shallow_features, conv1_up), dim=1)
        fuse1 = self.conv_fuse1(fuse1)
        fuse1 = self.batchnorm_fuse1(fuse1)
        fuse1 = F.relu(fuse1, inplace=True)

        # 第二次卷积的结果与第三次卷积上采样的结果做1x1卷积融合
        conv3_up = self.interpolate2(conv3)
        # print("conv3_up size:", conv3_up.size())
        # 确保大小匹配
        if conv3_up.size() != conv2.size():
            conv3_up = F.interpolate(conv3_up, size=conv2.size()[2:], mode='bilinear', align_corners=False)
        fuse2 = torch.cat((conv2, conv3_up), dim=1)
        fuse2 = self.conv_fuse2(fuse2)
        fuse2 = self.batchnorm_fuse2(fuse2)
        fuse2 = F.relu(fuse2, inplace=True)

        # 返回两个融合后的特征
        return fuse1, fuse2


class ECANet(nn.Module):
    def __init__(self, in_channels, r=4):
        super(ECANet, self).__init__()
        inter_channels = in_channels // r

        # ECA-Net 通道注意力模块
        self.ECA_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.ECA_act = nn.ReLU(inplace=True)
        self.ECA_conv2 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # AFD-Net 自适应特征蒸馏模块
        self.AFD_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.AFD_conv2 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # ECA注意力
        out = self.ECA_conv1(x)
        out = self.ECA_act(out)
        out = self.ECA_conv2(out)

        # AFD蒸馏
        out_distill = self.AFD_conv1(out)
        out = self.AFD_conv2(out_distill)

        return out


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x

    # -----------------------------------------#


#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 插入ECANet模块
        self.ECANet = ECANet(in_channels=dim_out * 5, r=4)

    def forward(self, x):
        b, c, row, col = x.size()

        # 五个分支的处理
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        # 全局平均池化分支
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # 拼接特征
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)

        # 应用ECANet
        eca_feature = self.ECANet(feature_cat)

        # 最后的卷积整合
        result = self.conv_cat(eca_feature)

        return result


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)
        self.sf_processor = FPN(in_channels=low_level_channels, out_channels=low_level_channels)
        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
        # 新增的上采样和堆叠层
        self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        # 调整输入通道数以匹配实际的输入
        self.conv_after_upsample = nn.Sequential(
            nn.Conv2d(376, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 新增的SSH模块
        self.ssh = SSH(256, 256)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, x = self.backbone(x)
        # 自定义浅层特征处理
        fuse1, fuse2 = self.sf_processor(low_level_features)
        # save_dir = "shallow_features"
        # save_shallow_features(fuse2, save_dir)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        # 新增的上采样和堆叠操作
        # 将fuse2和x堆叠后的结果做上采样得到128*128的特征层
        fused = torch.cat((fuse2, x), dim=1)
        fused = self.upsample(fused)

        # 将这个特征层与fuse1堆叠
        fused = torch.cat((fused, fuse1), dim=1)
        fused = self.conv_after_upsample(fused)
        fused = self.ssh(fused)
        x = self.ssh(x)
        fused = self.cls_conv(fused)
        fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=True)

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return fused


def save_shallow_features(shallow_features, save_dir):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 将张量转换为NumPy数组
    shallow_features_np = shallow_features.detach().cpu().numpy()

    # 遍历每个特征图并保存
    for i in range(shallow_features_np.shape[1]):
        feature_map = shallow_features_np[0, i, :, :]
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) * 255
        feature_map = feature_map.astype(np.uint8)

        # 保存为PNG图像
        save_path = os.path.join(save_dir, f"shallow_feature_{i}.png")
        Image.fromarray(feature_map).save(save_path)
