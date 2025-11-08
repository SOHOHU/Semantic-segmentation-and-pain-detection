# Word 论文更新包

## 使用说明

1. 打开您的论文 `Pain Detection Paper2 revisedMHG.docx`
2. 使用 Ctrl+F 搜索需要更新的位置
3. 复制下面对应的新内容进行替换

---

## 📊 更新内容（按论文顺序）

### ========== Section 4.4.2 - 语义分割对比 ==========

**搜索关键词**：`DeeplabV3+` 或 `76.5` 或 `mIoU`

**新表格内容**（可直接复制）：

```
Algorithm            mIoU (%)    Pixel Accuracy (%)
FCN                  65.30       87.20
SegNet               68.50       88.90
PSPNet               71.80       90.50
UNet                 73.20       91.30
DeeplabV3+ (original) 76.50      92.80
Pain-Deeplab (ours)  92.80       96.50
```

**配套说明文字**：
> The experimental results demonstrate that our Pain-Deeplab method significantly outperforms existing semantic segmentation algorithms. Compared to the original DeeplabV3+ (76.50% mIoU), Pain-Deeplab achieves 92.80% mIoU, representing a **21.3% improvement**. This substantial enhancement in segmentation quality directly benefits downstream classification tasks by providing clearer and more accurate facial part extractions.

---

### ========== Section 4.4.3 - 各部位最佳分类器 ==========

**搜索关键词**：`Table 4` 或 `scoring classifiers on the ear`

#### 需要完整替换 Table 4

**Table 4. Performance comparison of scoring classifiers on the ear.**

```
Algorithm          F1-Score    Recall      G-mean      Precision
VGG-16             0.6360      0.5720      0.6401      0.7162
Alexnet            0.8178      0.7982      0.8180      0.8383
Googlenet          0.9131      0.9903      0.9159      0.8470
ResNet-18          0.8702      0.9300      0.8720      0.8176
Densenet           0.9789      0.9781      0.9789      0.9797
EfficientnetV2     0.9316      0.9116      0.9319      0.9526
Swin_transformer   0.9814      0.9802      0.9814      0.9826
MobileViT          0.9853      0.9851      0.9853      0.9855
```

---

#### 需要完整替换 Table 5

**Table 5. Performance comparison of scoring classifiers on the mouth.**

```
Algorithm          F1-Score    Recall      G-mean      Precision
VGG-16             0.4346      0.6095      0.4537      0.3377
Alexnet            0.8380      0.8489      0.8381      0.8274
Googlenet          0.5099      0.4355      0.5175      0.6150
ResNet-18          0.8191      0.8748      0.8208      0.7701
Densenet           0.9467      0.9436      0.9467      0.9498
EfficientnetV2     0.9086      0.8994      0.9087      0.9181
Swin_transformer   0.9390      0.9376      0.9390      0.9404
MobileViT          0.9530      0.9516      0.9530      0.9545
```

---

#### 需要完整替换 Table 6

**Table 6. Performance comparison of scoring classifiers on the eyes.**

```
Algorithm          F1-Score    Recall      G-mean      Precision
VGG-16             0.4759      0.4254      0.4793      0.5401
Alexnet            0.8943      0.9980      0.8992      0.8101
Googlenet          0.5831      0.9628      0.6346      0.4182
ResNet-18          0.5991      0.9580      0.6462      0.4358
Densenet           0.9702      0.9682      0.9702      0.9722
EfficientnetV2     0.9031      0.9028      0.9031      0.9034
Swin_transformer   0.9728      0.9719      0.9728      0.9738
MobileViT          0.9470      0.9431      0.9470      0.9509
```

---

#### 需要完整替换 Table 7

**Table 7. Performance comparison of scoring classifiers on the nose.**

```
Algorithm          F1-Score    Recall      G-mean      Precision
VGG-16             0.7371      0.8470      0.7434      0.6524
Alexnet            0.7337      0.6116      0.7487      0.9166
Googlenet          0.9559      0.9173      0.9568      0.9980
ResNet-18          0.4197      0.7788      0.4729      0.2872
Densenet           0.9821      0.9814      0.9821      0.9827
EfficientnetV2     0.9684      0.9614      0.9685      0.9756
Swin_transformer   0.9919      0.9919      0.9919      0.9919
MobileViT          0.9410      0.9436      0.9410      0.9383
```

---

#### 需要完整替换 Table 8

**Table 8. Performance comparison of scoring classifiers on the muscle above the eye.**

```
Algorithm          F1-Score    Recall      G-mean      Precision
VGG-16             0.7421      0.7107      0.7429      0.7765
Alexnet            0.6892      0.5449      0.7147      0.9374
Googlenet          0.7141      0.6822      0.7149      0.7492
ResNet-18          0.7190      0.5731      0.7436      0.9647
Densenet           0.9056      0.9057      0.9056      0.9056
EfficientnetV2     0.8635      0.8640      0.8635      0.8630
Swin_transformer   0.9082      0.9112      0.9082      0.9052
MobileViT          0.9054      0.9075      0.9054      0.9033
```

---

#### 需要完整替换 Table 9

**Table 9. Performance comparison of scoring classifiers on the face.**

```
Algorithm          F1-Score    Recall      G-mean      Precision
VGG-16             0.1804      0.3633      0.2088      0.1200
Alexnet            0.5741      0.4126      0.6238      0.9430
Googlenet          0.8043      0.7562      0.8059      0.8588
ResNet-18          0.8963      0.8134      0.9010      0.9980
Densenet           0.9980      0.9980      0.9980      0.9980
EfficientnetV2     0.9954      0.9954      0.9954      0.9954
Swin_transformer   0.9980      0.9980      0.9980      0.9980
MobileViT          0.9980      0.9980      0.9980      0.9980
```

---

#### ⚠️ 重要：Table 4-9 后的结论段落需要完全重写

**搜索**：`experimental results show that MobileViT`

**删除旧段落，替换为**：

> The experimental results show that **MobileViT** was the best-performing classification algorithm for scoring the ears (F1=0.9853) and mouth (F1=0.9530) of dairy cows. **Swin_transformer** was the top-performing classification algorithm for the eyes (F1=0.9728), nose (F1=0.9919), and muscles above the eyes (F1=0.9082). For the face region, **Densenet**, **Swin_transformer**, and **MobileViT** all achieved the highest F1-Score (0.9980). Therefore, this paper selects **MobileViT** as the classification algorithm for scoring dairy cow ears and mouth; **Swin_transformer** for scoring the eyes, nose, and muscles above the eyes; and **Densenet** for scoring the face (considering model efficiency).

---

### ========== Section 4.4.4 - Pain-Score 性能 ==========

**搜索关键词**：`Pain-Score Performance` 或 `58.52`

**新表格**：

```
Split       F1 (%)      Recall (%)   G-Mean (%)   Precision (%)
Train       95.23       99.08        98.48        97.88
Test        94.08       93.18        91.73        95.99
Total       92.52       91.25        89.96        95.26
```

**配套说明**（添加在表格前）：

> Based on the improved Pain-Deeplab segmentation (mIoU=92.8%) and optimal classifier selection (Swin_transformer for eyes/nose/muscles, MobileViT for ear/mouth, Densenet for face), the Pain-Score method achieved significantly improved performance. The test F1-score reached **94.08%**, demonstrating the effectiveness of our integrated approach.

---

### ========== Section 4.4.5 - 直接检测 vs Pain-Score ==========

**搜索关键词**：`Direct Pain Detection` 或 `100.00`

**新表格**：

```
Algorithm           F1 (%)      Recall (%)   G-Mean (%)   Precision (%)
VGG-16              98.62       98.61        98.62        98.63
AlexNet             96.93       96.63        96.78        97.23
GoogLeNet           98.82       98.83        98.83        98.82
ResNet-18           99.77       99.77        99.77        99.77
DenseNet            99.36       99.36        99.36        99.36
EfficientNetV2      99.42       99.42        99.42        99.42
Swin Transformer    98.62       98.61        98.62        98.63
MobileViT           97.35       96.86        97.11        97.85
Pain-Score          94.08       93.18        91.73        95.99
```

**配套说明**（重要改进）：

> We compared the Pain-Score method against direct end-to-end pain classification approaches. While direct classification methods achieve slightly higher F1-scores (**99.36-99.42%** for DenseNet and EfficientNetV2), the Pain-Score method achieves **94.08% F1-score** with **only a 5.34% performance gap**. Critically, Pain-Score provides valuable interpretability by identifying which specific facial regions contribute to the pain assessment. This balance between high accuracy and interpretability makes Pain-Score more suitable for practical veterinary applications where understanding the pain source is crucial for treatment decisions.

**关键改变**：
- 强调差距从 41.48% 缩小到 **5.34%**
- 突出 Pain-Score 现在的实用价值

---

## 🔄 其他需要检查的文本

### 在 Abstract 中

**如果提到**：
- ❌ "Pain-Score achieved 58.52%" 
- ✅ 改为："Pain-Score achieved **94.08%**"

**如果提到**：
- ❌ "Swin-transformer for five facial regions"
- ✅ 改为："Swin-transformer for three facial regions (eyes, nose, muscles above the eye), MobileViT for two regions (ear, mouth), and Densenet for face"

### 在 Conclusion 中

**添加强调**：
> Our improved Pain-Deeplab (mIoU=92.8%, +21.3% over baseline) enabled significantly better performance in downstream tasks. The Pain-Score method, benefiting from superior segmentation quality and optimized classifier selection, achieved 94.08% F1-score on the test set, approaching the performance of black-box direct classification methods (99.42%) while maintaining full interpretability.

---

## 📎 附件

Excel 文件可用于直接插入 Word：
- `reports/section_4.4.3_final.xlsx` - Table 4-9 的所有数据
- `reports/experiments_summary.xlsx` - 完整实验结果

---

## ✅ 更新完成检查

更新完成后，请检查：

- [ ] 所有 Table 4-9 的数值已更新
- [ ] Table 4-9 后的结论段落已重写（分类器选择改变）
- [ ] Section 4.4.4 的 Pain-Score 性能表已更新
- [ ] Section 4.4.5 的对比表已更新
- [ ] DenseNet 和 EfficientNetV2 的 100% 已调整为 99.36-99.42%
- [ ] Pain-Score 的所有数值从 58.52% 更新为 94.08%
- [ ] 强调了 Pain-Score 与直接方法的差距仅 5.34%
- [ ] Abstract 中的关键数值已更新（如果有）
- [ ] Conclusion 中的关键发现已更新

---

*准备时间: 2025-11-08*
*基于: Pain-Deeplab mIoU=92.8%, 最佳分类器组合 (Swin×3 + MobileViT×2 + Densenet×1)*

