# Detailed Classifier Performance Comparison (Section 4.4.3)

To select the optimal scoring classifier for each dairy cow facial region obtained, we compared the performance of 8 classifiers across 6 facial regions. The performance of the 8 classifiers on each facial region is shown in Tables 4 to 9.

## Table 4. Performance comparison of scoring classifiers on the ear.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.7626 | 0.7747 | 0.7627 | 0.7510 |
| Alexnet | 0.6948 | 0.7058 | 0.6949 | 0.6842 |
| Googlenet | 0.7946 | 0.8071 | 0.7947 | 0.7824 |
| ResNet-18 | 0.8452 | 0.8586 | 0.8453 | 0.8323 |
| Densenet | 0.8594 | 0.8730 | 0.8595 | 0.8463 |
| EfficientnetV2 | 0.8584 | 0.8720 | 0.8586 | 0.8453 |
| Swin_transformer | 0.6712 | 0.6991 | 0.4568 | 0.6605 |
| MobileViT | 0.6517 | 0.6364 | 0.5944 | 0.6797 |

## Table 5. Performance comparison of scoring classifiers on the mouth.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.6993 | 0.7103 | 0.6993 | 0.6886 |
| Alexnet | 0.5901 | 0.5995 | 0.5902 | 0.5811 |
| Googlenet | 0.6283 | 0.6382 | 0.6283 | 0.6186 |
| ResNet-18 | 0.7352 | 0.7469 | 0.7353 | 0.7240 |
| Densenet | 0.7594 | 0.7714 | 0.7595 | 0.7477 |
| EfficientnetV2 | 0.7617 | 0.7737 | 0.7618 | 0.7500 |
| Swin_transformer | 0.4898 | 0.5024 | 0.4278 | 0.5421 |
| MobileViT | 0.5347 | 0.5366 | 0.4808 | 0.5510 |

## Table 6. Performance comparison of scoring classifiers on the eyes.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.6989 | 0.7099 | 0.6990 | 0.6882 |
| Alexnet | 0.6441 | 0.6542 | 0.6442 | 0.6342 |
| Googlenet | 0.6862 | 0.6971 | 0.6863 | 0.6757 |
| ResNet-18 | 0.7241 | 0.7355 | 0.7241 | 0.7130 |
| Densenet | 0.7732 | 0.7854 | 0.7733 | 0.7613 |
| EfficientnetV2 | 0.7555 | 0.7674 | 0.7556 | 0.7439 |
| Swin_transformer | 0.4654 | 0.4214 | 0.3926 | 0.5748 |
| MobileViT | 0.5166 | 0.5437 | 0.3300 | 0.6142 |

## Table 7. Performance comparison of scoring classifiers on the nose.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.7312 | 0.7427 | 0.7313 | 0.7200 |
| Alexnet | 0.6713 | 0.6819 | 0.6714 | 0.6610 |
| Googlenet | 0.7242 | 0.7356 | 0.7242 | 0.7131 |
| ResNet-18 | 0.7385 | 0.7502 | 0.7386 | 0.7272 |
| Densenet | 0.7977 | 0.8103 | 0.7978 | 0.7855 |
| EfficientnetV2 | 0.8145 | 0.8273 | 0.8146 | 0.8020 |
| Swin_transformer | 0.7338 | 0.7954 | 0.2670 | 0.8211 |
| MobileViT | 0.7209 | 0.7662 | 0.2955 | 0.7081 |

## Table 8. Performance comparison of scoring classifiers on the muscle above the eye.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| Swin_transformer | 0.7469 | 0.7393 | 0.6774 | 0.7970 |
| MobileViT | 0.7274 | 0.7321 | 0.6454 | 0.7607 |

## Table 9. Performance comparison of scoring classifiers on the face.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.7042 | 0.7153 | 0.7043 | 0.6934 |
| Alexnet | 0.6244 | 0.6343 | 0.6245 | 0.6148 |
| Googlenet | 0.6708 | 0.6814 | 0.6709 | 0.6605 |
| ResNet-18 | 0.7190 | 0.7304 | 0.7191 | 0.7080 |
| Densenet | 0.7673 | 0.7794 | 0.7674 | 0.7555 |
| EfficientnetV2 | 0.7606 | 0.7726 | 0.7607 | 0.7490 |
| Swin_transformer | 0.5430 | 0.6288 | 0.8637 | 0.4942 |
| MobileViT | 0.4248 | 0.5153 | 0.6485 | 0.3662 |

## Best Classifier Selection

The experimental results show that:

- **ear**: Densenet (F1-Score: 0.8594)
- **mouth**: EfficientnetV2 (F1-Score: 0.7617)
- **eyes**: Densenet (F1-Score: 0.7732)
- **nose**: EfficientnetV2 (F1-Score: 0.8145)
- **muscles above the eye**: Swin_transformer (F1-Score: 0.7469)
- **face**: Densenet (F1-Score: 0.7673)

## Summary

Based on the F1-Score metric, the best-performing classification algorithms for each facial region are:

- ear: **Densenet**
- mouth: **EfficientnetV2**
- eyes: **Densenet**
- nose: **EfficientnetV2**
- muscles above the eye: **Swin_transformer**
- face: **Densenet**
