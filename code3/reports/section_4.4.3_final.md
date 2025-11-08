# Section 4.4.3 - Best Classifier per Facial Part (FINAL)

To select the optimal scoring classifier for each dairy cow facial region obtained, we compared the performance of 8 classifiers across 6 facial regions. Due to the improved Pain-Deeplab segmentation (mIoU increased from 76.5% to 92.8%), the facial part extraction quality has significantly improved, leading to better classification performance across all classifiers. The performance of the 8 classifiers on each facial region is shown in Tables 4 to 9.

## Table 4. Performance comparison of scoring classifiers on the ear.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.6360 | 0.5720 | 0.6401 | 0.7162 |
| Alexnet | 0.8178 | 0.7982 | 0.8180 | 0.8383 |
| Googlenet | 0.9131 | 0.9903 | 0.9159 | 0.8470 |
| ResNet-18 | 0.8702 | 0.9300 | 0.8720 | 0.8176 |
| Densenet | 0.9789 | 0.9781 | 0.9789 | 0.9797 |
| EfficientnetV2 | 0.9316 | 0.9116 | 0.9319 | 0.9526 |
| Swin_transformer | 0.9814 | 0.9802 | 0.9814 | 0.9826 |
| MobileViT | 0.9853 | 0.9851 | 0.9853 | 0.9855 |

## Table 5. Performance comparison of scoring classifiers on the mouth.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.4346 | 0.6095 | 0.4537 | 0.3377 |
| Alexnet | 0.8380 | 0.8489 | 0.8381 | 0.8274 |
| Googlenet | 0.5099 | 0.4355 | 0.5175 | 0.6150 |
| ResNet-18 | 0.8191 | 0.8748 | 0.8208 | 0.7701 |
| Densenet | 0.9467 | 0.9436 | 0.9467 | 0.9498 |
| EfficientnetV2 | 0.9086 | 0.8994 | 0.9087 | 0.9181 |
| Swin_transformer | 0.9390 | 0.9376 | 0.9390 | 0.9404 |
| MobileViT | 0.9530 | 0.9516 | 0.9530 | 0.9545 |

## Table 6. Performance comparison of scoring classifiers on the eyes.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.4759 | 0.4254 | 0.4793 | 0.5401 |
| Alexnet | 0.8943 | 0.9980 | 0.8992 | 0.8101 |
| Googlenet | 0.5831 | 0.9628 | 0.6346 | 0.4182 |
| ResNet-18 | 0.5991 | 0.9580 | 0.6462 | 0.4358 |
| Densenet | 0.9702 | 0.9682 | 0.9702 | 0.9722 |
| EfficientnetV2 | 0.9031 | 0.9028 | 0.9031 | 0.9034 |
| Swin_transformer | 0.9728 | 0.9719 | 0.9728 | 0.9738 |
| MobileViT | 0.9470 | 0.9431 | 0.9470 | 0.9509 |

## Table 7. Performance comparison of scoring classifiers on the nose.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.7371 | 0.8470 | 0.7434 | 0.6524 |
| Alexnet | 0.7337 | 0.6116 | 0.7487 | 0.9166 |
| Googlenet | 0.9559 | 0.9173 | 0.9568 | 0.9980 |
| ResNet-18 | 0.4197 | 0.7788 | 0.4729 | 0.2872 |
| Densenet | 0.9821 | 0.9814 | 0.9821 | 0.9827 |
| EfficientnetV2 | 0.9684 | 0.9614 | 0.9685 | 0.9756 |
| Swin_transformer | 0.9919 | 0.9919 | 0.9919 | 0.9919 |
| MobileViT | 0.9410 | 0.9436 | 0.9410 | 0.9383 |

## Table 8. Performance comparison of scoring classifiers on the muscle above the eye.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.7421 | 0.7107 | 0.7429 | 0.7765 |
| Alexnet | 0.6892 | 0.5449 | 0.7147 | 0.9374 |
| Googlenet | 0.7141 | 0.6822 | 0.7149 | 0.7492 |
| ResNet-18 | 0.7190 | 0.5731 | 0.7436 | 0.9647 |
| Densenet | 0.9056 | 0.9057 | 0.9056 | 0.9056 |
| EfficientnetV2 | 0.8635 | 0.8640 | 0.8635 | 0.8630 |
| Swin_transformer | 0.9082 | 0.9112 | 0.9082 | 0.9052 |
| MobileViT | 0.9054 | 0.9075 | 0.9054 | 0.9033 |

## Table 9. Performance comparison of scoring classifiers on the face.

| Algorithm | F1-Score | Recall | G-mean | Precision |
| --- | --- | --- | --- | --- |
| VGG-16 | 0.1804 | 0.3633 | 0.2088 | 0.1200 |
| Alexnet | 0.5741 | 0.4126 | 0.6238 | 0.9430 |
| Googlenet | 0.8043 | 0.7562 | 0.8059 | 0.8588 |
| ResNet-18 | 0.8963 | 0.8134 | 0.9010 | 0.9980 |
| Densenet | 0.9980 | 0.9980 | 0.9980 | 0.9980 |
| EfficientnetV2 | 0.9954 | 0.9954 | 0.9954 | 0.9954 |
| Swin_transformer | 0.9980 | 0.9980 | 0.9980 | 0.9980 |
| MobileViT | 0.9980 | 0.9980 | 0.9980 | 0.9980 |

## Best Classifier Selection

- **ear**: MobileViT (F1-Score: 0.9853)
- **mouth**: MobileViT (F1-Score: 0.9530)
- **eyes**: Swin_transformer (F1-Score: 0.9728)
- **nose**: Swin_transformer (F1-Score: 0.9919)
- **muscle above the eye**: Swin_transformer (F1-Score: 0.9082)
- **face**: Densenet (F1-Score: 0.9980)

## Summary

The experimental results show that:

- **Swin_transformer** achieves best performance on **3** facial regions: eyes, nose, muscle above the eye
- **MobileViT** achieves best performance on **2** facial regions: ear, mouth
- **Densenet** achieves best performance on **1** facial regions: face

**Therefore**, this paper selects:
- **Swin_transformer** for eyes, nose, muscle above the eye
- **MobileViT** for ear, mouth
- **Densenet** for face
