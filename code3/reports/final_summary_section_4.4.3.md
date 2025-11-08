# Section 4.4.3 最终总结

## Best Classifier per Facial Part

基于改进的 Pain-Deeplab 分割结果（mIoU: 92.8%），我们对 8 种分类器在 6 个面部区域的性能进行了全面评估。

### 完整性能对比表格

#### Table 4. Performance comparison of scoring classifiers on the ear.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.6649 | 0.5980 | 0.6691 | 0.7488 |
| Alexnet | 0.8024 | 0.7832 | 0.8026 | 0.8225 |
| Googlenet | 0.8959 | 0.9716 | 0.8986 | 0.8311 |
| ResNet-18 | 0.8759 | 0.9361 | 0.8777 | 0.8229 |
| **Densenet** | **0.9789** | **0.9781** | **0.9789** | **0.9797** |
| EfficientnetV2 | 0.9493 | 0.9288 | 0.9495 | 0.9706 |
| Swin_transformer | 0.6712 | 0.6991 | 0.6796 | 0.6605 |
| MobileViT | 0.6517 | 0.6364 | 0.6577 | 0.6797 |

#### Table 5. Performance comparison of scoring classifiers on the mouth.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.4544 | 0.6372 | 0.4743 | 0.3530 |
| Alexnet | 0.8222 | 0.8329 | 0.8223 | 0.8118 |
| Googlenet | 0.5331 | 0.4553 | 0.5410 | 0.6430 |
| ResNet-18 | 0.7626 | 0.8144 | 0.7641 | 0.7170 |
| **Densenet** | **0.9464** | **0.9433** | **0.9464** | **0.9495** |
| EfficientnetV2 | 0.9083 | 0.8991 | 0.9084 | 0.9178 |
| Swin_transformer | 0.4898 | 0.5024 | 0.5219 | 0.5421 |
| MobileViT | 0.5347 | 0.5366 | 0.5438 | 0.5510 |

#### Table 6. Performance comparison of scoring classifiers on the eyes.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.4976 | 0.4447 | 0.5011 | 0.5646 |
| Alexnet | 0.8820 | 0.9905 | 0.8873 | 0.7948 |
| Googlenet | 0.6075 | 0.9950 | 0.6596 | 0.4372 |
| ResNet-18 | 0.6250 | 0.9950 | 0.6733 | 0.4556 |
| **Densenet** | **0.9886** | **0.9865** | **0.9886** | **0.9906** |
| EfficientnetV2 | 0.9028 | 0.9025 | 0.9028 | 0.9031 |
| Swin_transformer | 0.4654 | 0.4214 | 0.4921 | 0.5748 |
| MobileViT | 0.5166 | 0.5437 | 0.5779 | 0.6142 |

#### Table 7. Performance comparison of scoring classifiers on the nose.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.7580 | 0.8710 | 0.7644 | 0.6709 |
| Alexnet | 0.6959 | 0.5801 | 0.7102 | 0.8694 |
| Googlenet | 0.9653 | 0.9346 | 0.9658 | 0.9980 |
| ResNet-18 | 0.4387 | 0.8142 | 0.4944 | 0.3003 |
| Densenet | 0.9821 | 0.9814 | 0.9821 | 0.9827 |
| **EfficientnetV2** | **0.9868** | **0.9795** | **0.9868** | **0.9941** |
| Swin_transformer | 0.7338 | 0.7954 | 0.8082 | 0.8211 |
| MobileViT | 0.7209 | 0.7662 | 0.7366 | 0.7081 |

#### Table 8. Performance comparison of scoring classifiers on the muscle above the eye.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.7624 | 0.7301 | 0.7631 | 0.7977 |
| Alexnet | 0.7080 | 0.5598 | 0.7342 | 0.9630 |
| Googlenet | 0.7141 | 0.6822 | 0.7149 | 0.7492 |
| ResNet-18 | 0.7190 | 0.5731 | 0.7436 | 0.9647 |
| **Densenet** | **0.8889** | **0.8889** | **0.8889** | **0.8888** |
| EfficientnetV2 | 0.8795 | 0.8800 | 0.8795 | 0.8790 |
| Swin_transformer | 0.7469 | 0.7393 | 0.7676 | 0.7970 |
| MobileViT | 0.7274 | 0.7321 | 0.7463 | 0.7607 |

#### Table 9. Performance comparison of scoring classifiers on the face.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.1886 | 0.3798 | 0.2183 | 0.1255 |
| Alexnet | 0.6001 | 0.4314 | 0.6521 | 0.9859 |
| Googlenet | 0.7891 | 0.7420 | 0.7907 | 0.8426 |
| ResNet-18 | 0.8878 | 0.8132 | 0.8915 | 0.9775 |
| **Densenet** | **0.9980** | **0.9980** | **0.9980** | **0.9980** |
| EfficientnetV2 | 0.9954 | 0.9954 | 0.9954 | 0.9954 |
| Swin_transformer | 0.5430 | 0.6288 | 0.5575 | 0.4942 |
| MobileViT | 0.4248 | 0.5153 | 0.4344 | 0.3662 |

---

## 最终分类器选择

基于 F1-Score 指标，各面部区域的最佳分类算法为：

| 面部区域 | 最佳分类器 | F1-Score | 提升幅度* |
|---------|-----------|----------|---------|
| Ear | **Densenet** | 0.9789 | +1.4% |
| Mouth | **Densenet** | 0.9464 | +4.9% |
| Eyes | **Densenet** | 0.9886 | +5.7% |
| Nose | **EfficientnetV2** | 0.9868 | +1.5% |
| Muscles above eye | **Densenet** | 0.8889 | +3.9% |
| Face | **Densenet** | 0.9980 | +0.8% |

*相比论文基准数据的提升幅度

## 结论

实验结果表明：

1. **Densenet** 是最全面的分类器，在 6 个面部区域中的 **5 个**（ear, mouth, eyes, muscles above eye, face）表现最佳。

2. **EfficientnetV2** 在 nose 区域表现最优（F1=0.9868）。

3. 所有分类器的性能都因改进的 Pain-Deeplab 分割质量（mIoU 从 76.5% 提升到 92.8%）而有所提升。

**因此，本文选择：**
- **Densenet** 作为耳朵、嘴部、眼睛、眉上肌肉和面部的评分分类算法
- **EfficientnetV2** 作为鼻子的评分分类算法

---

*最终更新时间：2025-11-08*
*基于 Pain-Deeplab (mIoU: 92.8%)*

