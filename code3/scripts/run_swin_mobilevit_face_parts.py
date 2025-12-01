import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np


def ensure_paths():
    """将项目子模块路径加入 sys.path，便于导入模型定义。"""
    project_root = Path(__file__).resolve().parent
    swin_path = project_root / "pytorch_classification" / "swin_transformer"
    mobilevit_path = project_root / "pytorch_classification" / "MobileViT"

    for path in (swin_path, mobilevit_path):
        sys.path.append(str(path))

    return swin_path, mobilevit_path


def build_transforms(input_size: int = 224) -> Dict[str, transforms.Compose]:
    """构建训练/验证阶段的数据增强策略。"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return {
        "train": transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        "val": transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }


def load_datasets(part_root: Path) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """加载指定部件的训练/验证数据集。"""
    transforms_dict = build_transforms()
    train_dir = part_root / "train"
    val_dir = part_root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"缺少 train/val 目录: {part_root}")

    train_dataset = datasets.ImageFolder(str(train_dir), transform=transforms_dict["train"])
    val_dataset = datasets.ImageFolder(str(val_dir), transform=transforms_dict["val"])

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(f"{part_root} 中的 train/val 数据量不足，无法训练。")

    # 确保类别列表一致
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(f"{part_root} 的 train/val 类别不一致: "
                         f"{train_dataset.classes} vs {val_dataset.classes}")

    return train_dataset, val_dataset


def create_swin_model(num_classes: int, swin_path: Path) -> nn.Module:
    from model import swin_tiny_patch4_window7_224 as create_model  # type: ignore

    model = create_model(num_classes=num_classes)

    weights_path = swin_path / "swin_small_patch4_window7_224.pth"
    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location="cpu")
        checkpoint = checkpoint.get("model", checkpoint)
        checkpoint = {k: v for k, v in checkpoint.items() if "head" not in k}
        model.load_state_dict(checkpoint, strict=False)

    return model


def create_mobilevit_model(num_classes: int, mobilevit_path: Path) -> nn.Module:
    from model import mobile_vit_xx_small as create_model  # type: ignore

    model = create_model(num_classes=num_classes)

    weights_path = mobilevit_path / "mobilevit_xxs.pt"
    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location="cpu")
        checkpoint = checkpoint.get("model", checkpoint)
        checkpoint = {k: v for k, v in checkpoint.items() if "classifier" not in k}
        model.load_state_dict(checkpoint, strict=False)

    return model


def train_single_model(model: nn.Module,
                       model_name: str,
                       part_name: str,
                       train_dataset: datasets.ImageFolder,
                       val_dataset: datasets.ImageFolder,
                       device: torch.device,
                       epochs: int = 15,
                       batch_size: int = 32,
                       lr: float = 2e-4) -> Dict[str, Any]:
    """训练单个模型并返回最佳验证性能。"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_metrics: Dict[str, Any] = {
        "best_val_acc": 0.0,
        "best_epoch": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "confusion_matrix": None,
    }
    history: List[Dict[str, Any]] = []
    weight_dir = Path("swin_mobilevit_weights") / model_name
    weight_dir.mkdir(parents=True, exist_ok=True)
    best_weight_path = weight_dir / f"{part_name}_best.pth"

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in tqdm(train_loader,
                                   desc=f"[{model_name}-{part_name}] Epoch {epoch}/{epochs} Train",
                                   leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += inputs.size(0)

        train_loss = running_loss / total_samples
        train_acc = running_corrects / total_samples

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_total = 0
        all_labels: List[int] = []
        all_preds: List[int] = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader,
                                       desc=f"[{model_name}-{part_name}] Epoch {epoch}/{epochs} Val",
                                       leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_total += inputs.size(0)

                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

        val_loss /= max(val_total, 1)
        val_acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cm": cm.tolist(),
            "epoch_time_sec": time.time() - start_time,
        })

        # 更新最佳模型
        if val_acc > best_metrics["best_val_acc"]:
            best_metrics.update({
                "best_val_acc": float(val_acc),
                "best_epoch": epoch,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion_matrix": cm.tolist(),
            })
            torch.save(model.state_dict(), best_weight_path)

        scheduler.step()

        print(f"[{model_name}-{part_name}] Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Prec: {precision:.4f} Rec: {recall:.4f} F1: {f1:.4f}")

    best_metrics.update({
        "model_name": model_name,
        "part_name": part_name,
        "num_classes": len(train_dataset.classes),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "history": history,
        "weight_path": str(best_weight_path),
    })

    return best_metrics


def main():
    torch.backends.cudnn.benchmark = True
    swin_path, mobilevit_path = ensure_paths()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    face_parts = [
        "face",
        "mouth",
        "nose",
        "eyes",
        "ear",
        "muscles_above_eye",
    ]

    candidate_roots = [
        Path("face_parts_classified"),
        Path("result"),
    ]
    dataset_root = None
    for root in candidate_roots:
        if root.exists():
            sample_train = root / "face" / "train"
            if sample_train.exists() and (
                any(sample_train.rglob("*.png")) or any(sample_train.rglob("*.jpg")) or any(sample_train.rglob("*.jpeg"))
            ):
                dataset_root = root
                break

    if dataset_root is None:
        raise FileNotFoundError("未找到有效的数据集目录，请确保 face_parts_classified 或 result 已准备完成。")

    print(f"使用数据集根目录: {dataset_root}")

    results: List[Dict[str, Any]] = []

    models_to_run = ["swin_transformer", "mobilevit"]
    start_all = time.time()

    for part in face_parts:
        part_root = dataset_root / part
        print("=" * 100)
        print(f"开始训练部件: {part}")
        print("=" * 100)

        train_dataset, val_dataset = load_datasets(part_root)
        num_classes = len(train_dataset.classes)
        print(f"类别: {train_dataset.classes} (num_classes={num_classes})")
        print(f"训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)}")

        for model_flag in models_to_run:
            print("-" * 80)
            print(f"模型: {model_flag} | 部件: {part}")

            if model_flag == "swin_transformer":
                model = create_swin_model(num_classes=num_classes, swin_path=swin_path)
                lr = 2e-4
            elif model_flag == "mobilevit":
                model = create_mobilevit_model(num_classes=num_classes, mobilevit_path=mobilevit_path)
                lr = 1.5e-4
            else:
                raise ValueError(f"未知的模型: {model_flag}")

            model = model.to(device)
            metrics = train_single_model(
                model=model,
                model_name=model_flag,
                part_name=part,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                device=device,
                epochs=15,
                batch_size=32,
                lr=lr,
            )
            results.append(metrics)
            print(f"✓ 完成: {model_flag} - {part} | 最佳准确率: {metrics['best_val_acc']:.4f} (epoch {metrics['best_epoch']})")

    duration_min = (time.time() - start_all) / 60.0
    print("=" * 100)
    print(f"全部训练完成，总耗时约 {duration_min:.2f} 分钟")
    print("=" * 100)

    # 保存结果
    output_json = Path("swin_mobilevit_results.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 生成 CSV 摘要
    try:
        import pandas as pd

        summary_rows = []
        for item in results:
            summary_rows.append({
                "model": item["model_name"],
                "part": item["part_name"],
                "best_val_acc": item["best_val_acc"],
                "best_epoch": item["best_epoch"],
                "precision": item["precision"],
                "recall": item["recall"],
                "f1": item["f1"],
                "train_samples": item["train_samples"],
                "val_samples": item["val_samples"],
                "num_classes": item["num_classes"],
                "weight_path": item["weight_path"],
            })

        df = pd.DataFrame(summary_rows)
        df.to_csv("swin_mobilevit_results.csv", index=False, encoding="utf-8-sig")
        print("结果已保存到: swin_mobilevit_results.json / swin_mobilevit_results.csv")
    except Exception as exc:
        print(f"保存 CSV 失败: {exc}")
        print("JSON 已保存，可手动转换。")


if __name__ == "__main__":
    main()

