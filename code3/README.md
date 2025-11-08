# 奶牛疼痛检测系统

基于深度学习的奶牛疼痛检测系统，通过语义分割提取面部特征，结合多个面部部件的分类器进行疼痛评分。

**最后更新**: 2025-11-08  
**状态**: 已清理优化，仅保留核心代码和结果数据

---

## 📋 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [使用说明](#使用说明)
- [实验结果](#实验结果)
- [清理说明](#清理说明)
- [常见问题](#常见问题)

---

## 项目概述

本系统采用两阶段方法进行奶牛疼痛检测：

1. **语义分割阶段**: 使用Pain-Deeplab模型分割奶牛面部，提取6个关键部件
   - 耳朵 (Ear)
   - 眼睛 (Eyes)
   - 面部 (Face)
   - 嘴巴 (Mouth)
   - 眼上肌肉 (Muscles Above Eye)
   - 鼻子 (Nose)

2. **分类阶段**: 对每个面部部件进行疼痛等级分类（0/1/2级）
   - 0级：无痛
   - 1级：轻度疼痛
   - 2级：重度疼痛

3. **疼痛评分**: 综合6个部件的分类结果，生成最终疼痛评分

**主要特点**:
- ✅ 完整的端到端流程
- ✅ 已清理优化，项目结构清晰
- ✅ 包含完整的实验结果和报告
- ✅ 提供便捷的训练和预测脚本

---

## 快速开始

### 使用现有结果（无需训练）

```bash
# 查看实验结果
cd reports/
# experiments_summary.xlsx - 完整实验结果
# face_part_best.csv - 各部件最佳分类器
# sections_4.4.4_4.4.5_final.md - 详细分析报告
```

### 从头开始训练

```bash
# 1. 准备数据
python voc_annotation.py

# 2. 训练语义分割模型
cd deeplab_series/deeplab
python train.py  # 或使用 train_optimized.bat (Windows) / train_optimized.sh (Linux)

# 3. 提取面部部件
cd ../..
python organize_and_classify_face_parts.py

# 4. 训练分类器
python run_swin_mobilevit_face_parts.py
# 或
python batch_train_all_classifiers.py  # 训练多个分类器对比
```

### 一键提取面部部件

```bash
# Windows
quick_extract_face_parts.bat

# Linux/Mac
bash quick_extract_face_parts.sh
```

---

## 项目结构

```
code3/
├── deeplab_series/                    # 语义分割模块
│   └── deeplab/                       # Pain-Deeplab核心代码
│       ├── train.py                   # 训练脚本
│       ├── predict.py                 # 预测脚本
│       ├── get_miou.py                # 评估脚本
│       ├── nets/                      # 网络架构（DeepLabV3+）
│       ├── utils/                     # 工具函数
│       ├── model_data/                # 预训练权重（MobileNetV2骨干）
│       └── VOCdevkit/                 # 训练数据集（VOC格式）
│           └── VOC2007/
│               ├── JPEGImages/        # 原始图像（1,588张）
│               ├── SegmentationClass/ # 分割标签（1,588张）
│               └── ImageSets/         # 数据集划分
├── results/                           # 所有实验结果（已整合）
│   ├── face_parts/                    # 分割后的面部部件图像
│   │   ├── ear/                       # 耳朵（2,973张）
│   │   ├── eyes/                      # 眼睛（2,971张）
│   │   ├── face/                      # 面部（2,939张）
│   │   ├── mouth/                     # 嘴巴（2,676张）
│   │   ├── muscles_above_eye/         # 眼上肌肉（2,879张）
│   │   ├── nose/                      # 鼻子（2,994张）
│   │   └── Pain/                      # 原始全图（3,176张）
│   ├── classification/                # 分类器性能指标
│   │   ├── all_classifiers_results.csv
│   │   ├── all_classifiers_results.json
│   │   ├── swin_mobilevit_results.csv
│   │   ├── swin_mobilevit_results.json
│   │   └── confusion_matrix_detailed_analysis.json
│   ├── segmentation/                  # 分割结果和指标
│   │   ├── ablation/                  # 消融实验
│   │   ├── segmentation_ablation.csv
│   │   └── segmentation_baselines.csv
│   └── pain_score/                    # 疼痛评分结果
├── direct_detection_weights/          # 直接检测方法权重（基准）
│   ├── alexnet_best.pth
│   ├── densenet121_best.pth
│   ├── efficientnet_v2_s_best.pth
│   ├── googlenet_best.pth
│   ├── mobilevit_best.pth
│   ├── resnet18_best.pth
│   ├── swin_transformer_best.pth
│   └── vgg16_best.pth
├── reports/                           # 实验报告和论文数据
│   ├── experiments_summary.xlsx       # 完整实验结果
│   ├── face_part_best.csv             # 最佳分类器汇总
│   ├── detailed_classifier_tables.xlsx
│   ├── section_4.4.3_final.md         # 论文4.4.3节
│   ├── sections_4.4.4_4.4.5_final.md  # 论文4.4.4-4.4.5节
│   ├── 专家评审回复-QA格式.md             # 评审回复文档
│   ├── 论文修改指南-完整版.md             # 论文修改指南
│   └── figures/                       # 实验图表
├── organize_and_classify_face_parts.py  # 面部部件提取和组织
├── run_swin_mobilevit_face_parts.py     # Swin/MobileViT训练
├── batch_train_all_classifiers.py       # 批量训练多个分类器
├── generate_comparison.py               # 生成对比报告
├── voc_annotation.py                    # VOC数据集生成
├── json_to_dataset.py                   # JSON标注转换
├── quick_extract_face_parts.bat/sh      # 快捷提取脚本
├── Pain Detection Paper2 revisedMHG.docx # 论文原稿
├── PROJECT_CLEANUP_SUMMARY.md           # 清理总结报告
└── README.md                            # 本文档
```

### 关键说明

- **已删除的目录**: 
  - `FCN/`, `PSPNet/`, `segnet_series/`, `unet_series/` - 基准分割方法（约15GB）
  - `pytorch_classification/Test*` - 测试示例代码（约2GB）
  - 中间临时文件和备份（约8GB）
  
- **目录整合**: 
  - 原 `result/` → `results/face_parts/`
  - 原 `results/` → `results/` （保持）
  - 统一管理所有实验数据

- **分割结果**:
  - `results/segmentation/ablation/baseline/` 仅保留指标和可视化图
  - 大量中间 `detection-results/` 图像已清理，可通过 `get_miou_advanced.py` 重新生成

- **报告文档**:
  - `reports/` 现仅保留最终报告与指标文件（experiments_summary、sections_4.4.x、face_part_best 等）
  - 早期的 improved / updated 草稿与临时 Excel 已删除，避免重复

- **节省空间**: 约25-30GB，文件数从45,000减少到25,000

---

## 环境配置

### 系统要求

- **Python**: 3.9+
- **PyTorch**: 2.0+（CUDA 11.8+）
- **GPU**: 建议NVIDIA GPU（至少8GB显存）
- **操作系统**: Windows 10/11 或 Linux

### 依赖安装

```bash
# 1. 安装DeepLab依赖
pip install -r deeplab_series/deeplab/requirements.txt

# 主要依赖包括:
# - torch >= 2.0.0
# - torchvision >= 0.15.0
# - opencv-python
# - numpy
# - pillow
# - tqdm
# - matplotlib

# 2. 可选依赖（用于报告生成）
pip install pandas openpyxl python-docx seaborn
```

### 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

---

## 使用说明

### 1. 数据准备

#### 方法1：使用VOC格式数据

```bash
python voc_annotation.py
```

**数据组织**:
```
VOCdevkit/VOC2007/
├── JPEGImages/         # 原始图像
├── SegmentationClass/  # 分割mask（PNG格式）
└── ImageSets/Segmentation/
    ├── train.txt       # 训练集列表
    ├── val.txt         # 验证集列表
    └── test.txt        # 测试集列表（可选）
```

#### 方法2：从JSON标注转换

```bash
python json_to_dataset.py
```

支持LabelMe等工具生成的JSON标注文件。

### 2. 训练语义分割模型

```bash
cd deeplab_series/deeplab

# 基础训练
python train.py

# 或使用优化版本（推荐）
python train_advanced.py

# 或使用便捷脚本
# Windows:
train_optimized.bat

# Linux/Mac:
bash train_optimized.sh
```

**训练配置**:
- **输入尺寸**: 512×512
- **批次大小**: 8-16（根据GPU显存调整）
- **优化器**: Adam + EMA（Exponential Moving Average）
- **学习率**: 1e-4（余弦退火）
- **训练轮次**: 100 epochs（Early Stopping）
- **数据增强**: 随机翻转、旋转、缩放、颜色抖动

**输出**:
- 模型权重保存在 `deeplab_series/deeplab/logs/`
- 最佳模型: `best_epoch_weights_ema.pth`
- 训练日志: TensorBoard格式

### 3. 评估分割模型

```bash
cd deeplab_series/deeplab

# 计算mIoU等指标
python get_miou.py

# 或使用高级版本（包含更多指标）
python get_miou_advanced.py
```

**评估指标**:
- mIoU (Mean Intersection over Union)
- Pixel Accuracy
- Precision/Recall（每个类别）
- 混淆矩阵

### 4. 预测新图像

```bash
   cd deeplab_series/deeplab

# 单张图像预测
python predict.py

# 批量预测
python predict_advanced.py
```

预测结果保存在 `img_out/` 目录（会自动创建）。

### 5. 提取面部部件

```bash
# 从分割结果中提取面部部件并组织为分类数据集
python organize_and_classify_face_parts.py
```

**功能**:
- 从分割mask中裁剪6个面部部件
- 按疼痛等级（0/1/2）自动分类
- 划分训练集和验证集（8:2比例）
- 保存到 `results/face_parts/` 目录

**输出结构**:
```
results/face_parts/
├── ear/
│   ├── 0/        # 无痛样本
│   ├── 2/        # 重度疼痛样本
│   ├── train/    # 训练集
│   └── val/      # 验证集
├── eyes/
├── face/
├── mouth/
├── muscles_above_eye/
└── nose/
```

### 6. 训练分类器

#### 方法1：训练Swin Transformer和MobileViT（推荐）

```bash
python run_swin_mobilevit_face_parts.py
```

为6个面部部件分别训练Swin和MobileViT分类器（共12个模型）。

#### 方法2：训练所有8个分类器（完整基准测试）

```bash
python batch_train_all_classifiers.py
```

训练以下分类器：
- AlexNet
- VGG16
- GoogLeNet
- ResNet-18
- DenseNet-121
- EfficientNet-V2
- Swin Transformer
- MobileViT

**训练参数**:
- 训练轮次: 100 epochs
- 批次大小: 32
- 学习率: 1e-4（Swin），5e-5（MobileViT）
- 优化器: AdamW
- 学习率调度: Cosine Annealing
- 数据增强: 随机裁剪、翻转、颜色调整

**输出**:
- 性能指标: `results/classification/`
- 训练日志: 控制台输出

### 7. 生成分析报告

```bash
python generate_comparison.py
```

生成详细的对比分析报告，包括：
- 分割方法对比表格
- 分类器性能对比
- 可视化图表
- 保存到 `reports/` 目录

---

## 实验结果

所有实验结果保存在 `reports/` 和 `results/` 目录中。

### 语义分割性能

| 模型 | mIoU | Pixel Acc | FPS |
|------|------|-----------|-----|
| Pain-Deeplab | 85.3% | 92.1% | 28 |

*详细结果见*: `reports/section_4.4.3_final.md`

### 分类性能（最佳）

| 面部部件 | 最佳模型 | 准确率 | 精确率 | 召回率 | F1分数 |
|---------|---------|--------|--------|--------|--------|
| 耳朵 | Swin | 94.2% | 94.5% | 93.8% | 0.941 |
| 眼睛 | MobileViT | 95.8% | 96.1% | 95.5% | 0.957 |
| 面部 | Swin | 93.6% | 93.9% | 93.2% | 0.935 |
| 嘴巴 | MobileViT | 96.3% | 96.7% | 95.9% | 0.962 |
| 眼上肌肉 | Swin | 94.7% | 95.0% | 94.3% | 0.946 |
| 鼻子 | MobileViT | 95.1% | 95.4% | 94.8% | 0.950 |

*详细结果见*: `reports/face_part_best.csv`

### 疼痛评分准确率

- **基于分割的方法**: 95.2%
- **直接检测方法**: 89.7%
- **性能提升**: +5.5%

*详细对比见*: `reports/experiments_summary.xlsx`

### 各分类器对比（平均准确率）

| 分类器 | 平均准确率 | 训练时间 |
|--------|-----------|---------|
| Swin Transformer | 94.8% | ~4小时 |
| MobileViT | 95.1% | ~3小时 |
| EfficientNet-V2 | 93.2% | ~3.5小时 |
| DenseNet-121 | 92.7% | ~3小时 |
| ResNet-18 | 91.5% | ~2小时 |
| VGG16 | 90.3% | ~2.5小时 |
| GoogLeNet | 89.8% | ~2小时 |
| AlexNet | 87.2% | ~1.5小时 |

*详细结果见*: `results/classification/all_classifiers_results.csv`

---

## 清理说明

本项目已于 **2025-11-08** 进行全面清理优化。

### 已删除内容（节省约25-30GB）

1. **基准分割方法** (~15GB)
   - FCN, PSPNet, SegNet, UNet系列
   - 对比实验已完成，结果已保存

2. **DeepLab备份** (~8GB)
   - deeplab_zz, deeplabv3-original
   - 非生产版本

3. **分类器测试代码** (~2GB)
   - pytorch_classification/Test1-11系列
   - 示例和演示代码

4. **中间结果** (~3GB)
   - face_parts_extracted, face_parts_classified
   - 临时输出文件
   - Python缓存 (__pycache__)

5. **临时文件**
   - experiments, logs备份
   - Dataset备份
   - IDE配置文件

### 目录整合

- **result/ + results/ → results/**
  - 原因: 避免混淆，统一管理
  - 影响: 所有结果数据现在在 `results/` 下
  - `result/` → `results/face_parts/`

### 保留内容

✅ **核心代码**: DeepLab分割，分类器训练脚本  
✅ **训练数据**: VOCdevkit数据集（1,588张图像+标注）  
✅ **实验结果**: results/ 和 reports/（论文数据）  
✅ **基准权重**: direct_detection_weights/（8个模型）  
✅ **文档**: README, 实验报告

### 清理效果

- **空间节省**: 25-30GB
- **文件数量**: 45,000 → 25,000
- **目录数量**: 减少约200个
- **结构**: 更清晰，易于维护

---

## 常见问题

### Q1: 为什么没有pytorch_classification目录？

A: 项目已经过精简，分类器代码整合在训练脚本中：
- `run_swin_mobilevit_face_parts.py` - Swin/MobileViT实现
- `batch_train_all_classifiers.py` - 多个分类器实现

如需独立的分类器模块，可以从训练脚本中提取模型定义。

### Q2: 权重文件在哪里？

A: 
- **分割模型权重**: 训练后保存在 `deeplab_series/deeplab/logs/`
- **分类器权重**: 训练后保存在脚本指定的目录
- **直接检测权重**: `direct_detection_weights/`（基准方法）

如需预训练权重，需要重新训练或从备份中恢复。

### Q3: 如何恢复results/face_parts数据？

A: 重新运行提取脚本（需约30分钟）：

```bash
# 方法1：使用便捷脚本
quick_extract_face_parts.bat  # Windows
bash quick_extract_face_parts.sh  # Linux

# 方法2：手动运行
   python organize_and_classify_face_parts.py
```

**前提**: 需要有分割模型权重和VOCdevkit数据

### Q4: 训练被中断如何恢复？

A: DeepLab训练支持断点续训：

```bash
cd deeplab_series/deeplab
python train.py  # 自动从最后检查点恢复
```

分类器训练通常较快，建议重新开始。

### Q5: 如何备份项目？

A: 推荐备份以下内容：

```bash
# 1. 代码和脚本
git add *.py *.md *.sh *.bat
git commit -m "Backup scripts"

# 2. 训练数据（必需）
tar -czf VOCdevkit_backup.tar.gz deeplab_series/deeplab/VOCdevkit/

# 3. 实验结果（重要）
tar -czf results_backup.tar.gz results/ reports/

# 4. 模型权重（耗时训练，强烈建议）
tar -czf weights_backup.tar.gz deeplab_series/deeplab/logs/
```

### Q6: 显存不足怎么办？

A: 调整训练参数：

```python
# 在train.py中修改
batch_size = 4  # 默认8，减小到4或2
input_shape = [384, 384]  # 默认512，减小到384
```

### Q7: 如何添加新的分类器？

A: 修改 `batch_train_all_classifiers.py`：

```python
# 1. 导入新模型
from your_model import YourModel

# 2. 添加到模型字典
models = {
    'your_model': YourModel(num_classes=3),
    # ... 其他模型
}

# 3. 运行训练
python batch_train_all_classifiers.py
```

### Q8: 预测速度慢怎么优化？

A: 
1. **使用GPU**: 确保CUDA可用
2. **批量预测**: 使用 `predict_advanced.py`
3. **模型优化**: 
   ```python
   model.eval()
   with torch.no_grad():
       # 推理代码
   ```
4. **TensorRT**: 转换模型为TensorRT（高级）

### Q9: 如何查看详细实验结果？

A: 查看以下文件：

```bash
cd reports/

# 完整实验数据
experiments_summary.xlsx

# 各部件最佳分类器
face_part_best.csv

# 论文对应章节
section_4.4.3_final.md          # 分割方法对比
sections_4.4.4_4.4.5_final.md   # 分类器对比

# 详细表格
detailed_classifier_tables.xlsx
```

### Q10: 项目适合什么硬件配置？

A: 

**最低配置**:
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 16GB
- GPU: NVIDIA GTX 1660 (6GB)
- 存储: 50GB可用空间

**推荐配置**:
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 32GB
- GPU: NVIDIA RTX 3060 (12GB)
- 存储: 100GB SSD

**最佳配置**:
- CPU: Intel i9 / AMD Ryzen 9
- RAM: 64GB
- GPU: NVIDIA RTX 3090 / 4090 (24GB)
- 存储: 500GB NVMe SSD

---

## 维护建议

### 定期备份

1. **权重文件** (最重要)
   - 训练耗时长，强烈建议云端备份
   - 百度网盘、阿里云OSS、腾讯云COS

2. **实验结果**
   - results/ 和 reports/ 目录
   - 论文数据源，建议每次实验后备份

3. **训练数据**
   - VOCdevkit/ 目录
   - 原始标注数据，丢失难以恢复

### 版本控制

使用Git管理代码，排除大文件：

```bash
# .gitignore
*.pth
*.jpg
*.png
__pycache__/
results/face_parts/
deeplab_series/deeplab/logs/
```

### 清理缓存

定期清理Python缓存：

```bash
# Windows
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# Linux/Mac
find . -type d -name "__pycache__" -exec rm -rf {} +
```

---

## 论文引用

如果本项目对您的研究有帮助，请引用我们的论文：

```bibtex
@article{pain-score-detection-2025,
  title={Pain-Score Detection for Dairy Cows Using Semantic Segmentation and Classification},
  author={[作者名]},
  journal={[期刊名]},
  year={2025},
  volume={[卷号]},
  pages={[页码]}
}
```

---

## 许可证

本项目采用 [MIT License](LICENSE)

---
