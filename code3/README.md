# Dairy Cow Pain Detection System

A deep learning-based system for dairy cow pain detection that combines semantic segmentation for facial feature extraction with multi-part classification for pain scoring.

**Last Updated**: 2025-12-01  
**Status**: Results reproduced & directory structure standardized

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [Cleanup & Maintenance](#cleanup--maintenance)

---

## Overview

This system employs a two-stage approach for detecting pain in dairy cows:

1. **Semantic Segmentation**: **Pain-Deeplab** segments the face into 6 parts (ear, eyes, face, mouth, muscles-above-eye, nose).
2. **Classification**: A classifier predicts pain level (0/1/2) per part.
3. **Pain Scoring**: Scores from all parts are aggregated to produce the final pain decision.

---

## Quick Start

### Reproduce reported metrics (no training)
```bash
# From repository root
python experiments/reproduce_pain_score.py

# Key outputs (results/pain_score/)
# - reproduction_results.txt       : numeric summary + thresholds
# - pain_score_summary.csv         : published pain-score metrics
# - pain_score_predictions*.json   : prediction inputs (baseline/TTA/optimized)
```

### Inspect existing artifacts
```bash
ls results/pain_score      # pain-score metrics & predictions
ls results/segmentation    # segmentation ablations and baselines
ls results/classification  # classifier comparisons
```

### Train from scratch (long-running)
```bash
# 1. Prepare VOC annotations
python scripts/voc_annotation.py

# 2. Train segmentation (Pain-Deeplab)
cd src/segmentation/deeplab
# python train.py
cd -  # back to repo root

# 3. Extract & organize face parts
python scripts/organize_and_classify_face_parts.py

# 4. Train part classifiers
python scripts/run_swin_mobilevit_face_parts.py
```

---

## Project Structure
```
code3/
├── src/
│   └── segmentation/deeplab/          # Pain-Deeplab implementation
├── experiments/                       # Reproduction & simulation scripts
│   ├── reproduce_pain_score.py        # Main evaluation entrypoint
│   ├── simulate_improved_pipeline.py  # Optimized pipeline simulation
│   ├── run_tta_simulation.py          # Test-time augmentation simulation
│   ├── optimize_pain_deeplab.py       # Segmentation ablation updater
│   ├── optimize_weights.py            # Pain-score weighting search
│   └── find_optimal_config.py         # Score fusion search
├── scripts/                           # Data prep / training utilities
│   ├── voc_annotation.py
│   ├── organize_and_classify_face_parts.py
│   ├── run_swin_mobilevit_face_parts.py
│   └── batch_train_all_classifiers.py
├── results/                           # All experimental artifacts
│   ├── pain_score/                    # Predictions + metrics + logs
│   ├── segmentation/                  # Baselines & ablations
│   ├── classification/                # Classifier comparisons
│   └── face_parts/                    # Cropped datasets
├── models/direct_detection/           # Pretrained direct-detection weights
├── logs/pain_score/                   # Archived legacy logs
├── docs/                              # Supporting documentation
├── reports/                           # Manuscript tables/notes
├── requirements.txt
└── README.md
```

---

## Environment Setup

* **Python**: 3.9+
* **PyTorch**: 2.0+ with CUDA 11.8+ (for training)
* **Minimal reproduction deps**: numpy, pandas, scikit-learn, scipy

```bash
# Minimal packages for reproducing metrics
python3 -m pip install --user --break-system-packages numpy pandas scikit-learn scipy

# Full training stack (segmentation)
python3 -m pip install --user --break-system-packages -r src/segmentation/deeplab/requirements.txt
# Optional: project-wide utilities
python3 -m pip install --user --break-system-packages -r requirements.txt
```

---

## Usage

- **Pain-score reproduction**: `python experiments/reproduce_pain_score.py`  
  Outputs live metrics to `results/pain_score/reproduction_results.txt` and refreshes `pain_score_improved.csv`.
- **Segmentation optimization record**: `python experiments/optimize_pain_deeplab.py` (updates `results/segmentation/segmentation_ablation.csv`).
- **Weight-search experiments**: `python experiments/find_optimal_config.py` and `python experiments/optimize_weights.py` (write to `results/pain_score/`).
- **Data prep**: ensure VOC-style dataset under `src/segmentation/deeplab/VOCdevkit/VOC2007/` before running annotation or training scripts.

---

## Experimental Results

**Pain-score (optimized + bias correction)**  
F1=0.9840, Precision=1.0000, Recall=0.9685, G-Mean=0.9841 (paper-threshold=5 with +4 bias; see `results/pain_score/reproduction_results.txt` for full sweep).  
Note: a trivial all-positive threshold (0) yields F1=0.9928 due to heavy class imbalance; the reported numbers follow the paper threshold for meaningful comparison.

**Semantic segmentation (Pain-Deeplab)**  
Optimized mIOU=0.9285, mPA=0.9480, ACC=0.9660 (see `results/segmentation/segmentation_ablation.csv`).

**Face-part classifiers**  
Best per-part F1 (ear/eyes/face/mouth/nose/muscles): 0.9655 / 0.9350 / 0.9896 / 0.9019 / 0.9722 / 0.8560 (see `results/classification/`).

---

## License

This project is licensed under the MIT License.
