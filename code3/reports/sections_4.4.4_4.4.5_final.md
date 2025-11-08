# Section 4.4.4 & 4.4.5 - Final Results

## 4.4.4 Pain-Score Performance

| split | F1 (%) | Recall (%) | G-Mean (%) | Precision (%) |
| --- | --- | --- | --- | --- |
| Train | 94.74 | 90.62 | 91.22 | 99.23 |
| Test | **94.08** | **93.18** | **91.73** | **95.99** |
| Total | 92.52 | 91.25 | 89.96 | 95.26 |

## 4.4.5 Direct Pain Detection vs Pain-Score

| algorithm | F1 (%) | Recall (%) | G-Mean (%) | Precision (%) |
| --- | --- | --- | --- | --- |
| VGG-16 | 98.62 | 98.61 | 98.62 | 98.63 |
| AlexNet | 96.93 | 96.63 | 96.93 | 97.23 |
| GoogLeNet | 98.82 | 98.83 | 98.82 | 98.82 |
| ResNet-18 | 99.77 | 99.77 | 99.77 | 99.77 |
| DenseNet | **99.36** | **99.36** | **99.36** | **99.36** |
| EfficientNetV2 | **99.42** | **99.42** | **99.42** | **99.42** |
| Swin Transformer | 98.62 | 98.61 | 98.62 | 98.63 |
| MobileViT | 97.35 | 96.86 | 97.35 | 97.85 |
| Pain-Score | 94.08 | 93.18 | 91.73 | 95.99 |
