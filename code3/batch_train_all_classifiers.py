"""
批量训练所有分类器 - 自动化测试
测试8种分类器 x 5个面部部位 = 40个模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import time
import pandas as pd

# 全局配置
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {DEVICE}")
print()

# 定义所有分类器
def get_model(model_name, num_classes, pretrained=True):
    """获取指定的分类模型"""
    if model_name == 'VGG16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'AlexNet':
        model = models.alexnet(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'GoogLeNet':
        model = models.googlenet(pretrained=pretrained)
        model.fc = nn.Linear(1024, num_classes)
    elif model_name == 'ResNet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(512, num_classes)
    elif model_name == 'DenseNet':
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(1024, num_classes)
    elif model_name == 'EfficientNetV2':
        model = models.efficientnet_v2_s(pretrained=pretrained)
        model.classifier[1] = nn.Linear(1280, num_classes)
    elif model_name == 'ResNet34':
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(512, num_classes)
    elif model_name == 'MobileNetV2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(1280, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def train_single_model(model_name, part_name, data_root, num_classes, epochs=30):
    """训练单个模型"""
    
    print(f"\n{'='*80}")
    print(f"训练: {model_name} - {part_name} ({num_classes}分类)")
    print(f"{'='*80}")
    
    # 数据转换
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # 加载数据
    try:
        train_dataset = datasets.ImageFolder(
            root=os.path.join(data_root, "train"),
            transform=data_transform["train"]
        )
        val_dataset = datasets.ImageFolder(
            root=os.path.join(data_root, "val"),
            transform=data_transform["val"]
        )
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # 创建模型
    try:
        model = get_model(model_name, num_classes).to(DEVICE)
    except Exception as e:
        print(f"创建模型失败: {e}")
        return None
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    # 训练
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.double() / len(train_dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
        
        # 每5个epoch显示一次
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        scheduler.step()
    
    training_time = time.time() - start_time
    
    print(f"\n最佳验证准确率: {best_acc:.4f} (Epoch {best_epoch})")
    print(f"训练耗时: {training_time/60:.2f} 分钟")
    
    return {
        'model_name': model_name,
        'part_name': part_name,
        'num_classes': num_classes,
        'best_val_acc': float(best_acc),
        'best_epoch': best_epoch,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'training_time_minutes': training_time / 60
    }


def main():
    """主函数 - 批量训练所有组合"""
    
    print("="*80)
    print("批量分类器性能测试")
    print("="*80)
    print(f"测试: 8种分类器 x 5个面部部位")
    print()
    
    # 定义测试组合
    classifiers = ['ResNet18', 'VGG16', 'AlexNet', 'GoogLeNet', 
                   'DenseNet', 'EfficientNetV2', 'ResNet34', 'MobileNetV2']
    
    parts = {
        'face': ('result/face', 3),
        'mouth': ('result/mouth', 3),
        'nose': ('result/nose', 3),
        'eyes': ('result/eyes', 3),
        'ear': ('result/ear', 2)  # 只有2类
    }
    
    results = []
    total_tests = len(classifiers) * len(parts)
    current_test = 0
    
    print(f"总计测试: {total_tests} 个组合")
    print(f"预计总时间: {total_tests * 5} 分钟 (每个约5分钟)")
    print()
    
    # 遍历所有组合
    for part_name, (data_root, num_classes) in parts.items():
        for model_name in classifiers:
            current_test += 1
            print(f"\n进度: {current_test}/{total_tests}")
            
            try:
                result = train_single_model(
                    model_name=model_name,
                    part_name=part_name,
                    data_root=data_root,
                    num_classes=num_classes,
                    epochs=EPOCHS
                )
                
                if result:
                    results.append(result)
                    
            except Exception as e:
                print(f"训练失败: {model_name} - {part_name}, 错误: {e}")
                continue
    
    # 保存结果
    print("\n" + "="*80)
    print("保存结果...")
    print("="*80)
    
    # 保存为JSON
    with open('all_classifiers_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # 保存为CSV  
    df = pd.DataFrame(results)
    df.to_csv('all_classifiers_results.csv', index=False, encoding='utf-8-sig')
    
    # 生成总结报告
    generate_summary_report(results)
    
    print("\n✓ 结果已保存:")
    print("  - all_classifiers_results.json")
    print("  - all_classifiers_results.csv")
    print("  - 分类器性能总结报告.txt")
    print()


def generate_summary_report(results):
    """生成总结报告"""
    
    df = pd.DataFrame(results)
    
    report = []
    report.append("="*100)
    report.append("                    面部部件分类器性能测试 - 完整报告")
    report.append("="*100)
    report.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"测试组合: 8种分类器 x 5个面部部位 = 40个模型")
    report.append("")
    
    # 按部位分组统计
    report.append("="*100)
    report.append("【一】各部位最佳分类器")
    report.append("="*100)
    report.append("")
    
    for part in df['part_name'].unique():
        part_df = df[df['part_name'] == part].sort_values('best_val_acc', ascending=False)
        best_row = part_df.iloc[0]
        
        report.append(f"{part} (最佳: {best_row['model_name']})")
        report.append(f"  排名:")
        for idx, row in part_df.head(3).iterrows():
            report.append(f"    {row['model_name']:20s}: {row['best_val_acc']:.4f} "
                         f"(Epoch {row['best_epoch']})")
        report.append("")
    
    # 按分类器分组统计
    report.append("="*100)
    report.append("【二】各分类器平均性能")
    report.append("="*100)
    report.append("")
    
    model_avg = df.groupby('model_name')['best_val_acc'].agg(['mean', 'std', 'max', 'min'])
    model_avg = model_avg.sort_values('mean', ascending=False)
    
    report.append(f"{'分类器':20s} {'平均准确率':12s} {'标准差':10s} {'最高':10s} {'最低':10s}")
    report.append("-"*100)
    for model_name, row in model_avg.iterrows():
        report.append(f"{model_name:20s} {row['mean']:10.4f}   {row['std']:8.4f}  "
                     f"{row['max']:8.4f}  {row['min']:8.4f}")
    
    report.append("")
    
    # 完整结果表格
    report.append("="*100)
    report.append("【三】完整性能表格")
    report.append("="*100)
    report.append("")
    
    # 创建透视表
    pivot = df.pivot(index='model_name', columns='part_name', values='best_val_acc')
    
    report.append(f"{'分类器':15s} {'face':10s} {'mouth':10s} {'nose':10s} {'eyes':10s} {'ear':10s} {'平均':10s}")
    report.append("-"*100)
    
    for model_name in pivot.index:
        row_data = pivot.loc[model_name]
        avg = row_data.mean()
        report.append(f"{model_name:15s} ", end='')
        for part in ['face', 'mouth', 'nose', 'eyes', 'ear']:
            if part in row_data:
                report.append(f"{row_data[part]:8.4f}  ", end='')
            else:
                report.append(f"{'N/A':8s}  ", end='')
        report.append(f"{avg:8.4f}")
    
    report.append("")
    
    # 最佳组合
    report.append("="*100)
    report.append("【四】最佳部位-分类器组合 (TOP 10)")
    report.append("="*100)
    report.append("")
    
    top_results = df.nlargest(10, 'best_val_acc')
    report.append(f"{'排名':6s} {'部位':15s} {'分类器':20s} {'准确率':12s} {'训练集':10s} {'验证集':10s}")
    report.append("-"*100)
    
    for idx, (_, row) in enumerate(top_results.iterrows(), 1):
        report.append(f"{idx:4d}   {row['part_name']:15s} {row['model_name']:20s} "
                     f"{row['best_val_acc']:10.4f}  {row['train_samples']:8d}  "
                     f"{row['val_samples']:8d}")
    
    # 保存报告
    report_text = '\n'.join(report)
    with open('分类器性能总结报告.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)


if __name__ == '__main__':
    main()

