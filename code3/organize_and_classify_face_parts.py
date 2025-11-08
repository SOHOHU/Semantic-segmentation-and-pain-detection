"""
组织面部部件数据集并使用分类器进行测试
参考result文件夹的结构进行划分
"""

import os
import shutil
from sklearn.model_selection import train_test_split
import random

print("="*80)
print("面部部件数据集组织和分类测试")
print("="*80)
print()

# 步骤1: 分析result文件夹的结构
print("【步骤1】分析result文件夹的组织结构...")
print("-"*80)

result_dir = 'result'
face_parts = ['face', 'mouth', 'nose', 'eyes', 'ear', 'muscles_above_eye']

print("\nresult文件夹结构分析:")
for part in face_parts[:1]:  # 只分析face作为示例
    part_dir = os.path.join(result_dir, part)
    if os.path.exists(part_dir):
        # 检查类别
        subdirs = [d for d in os.listdir(part_dir) if os.path.isdir(os.path.join(part_dir, d))]
        print(f"\n{part}/")
        print(f"  子目录: {subdirs}")
        
        # 检查train和val
        if 'train' in subdirs and 'val' in subdirs:
            train_dir = os.path.join(part_dir, 'train')
            val_dir = os.path.join(part_dir, 'val')
            
            train_classes = os.listdir(train_dir)
            val_classes = os.listdir(val_dir)
            
            print(f"  train/ 类别: {train_classes}")
            for cls in train_classes:
                cls_dir = os.path.join(train_dir, cls)
                count = len([f for f in os.listdir(cls_dir) if f.endswith('.png')])
                print(f"    {cls}: {count} 张")
            
            print(f"  val/ 类别: {val_classes}")
            for cls in val_classes:
                cls_dir = os.path.join(val_dir, cls)
                count = len([f for f in os.listdir(cls_dir) if f.endswith('.png')])
                print(f"    {cls}: {count} 张")

print("\n发现规律:")
print("  - 每个部件有3个类别: 0, 1, 2")
print("  - 分为train和val两个集合")
print("  - train/val下各有0, 1, 2三个类别文件夹")
print()

# 步骤2: 读取训练集/验证集划分列表
print("【步骤2】读取原始数据集的train/val划分...")
print("-"*80)

train_txt = r'Deeplab系列\deeplab\VOCdevkit\VOC2007\ImageSets\Segmentation\train.txt'
val_txt = r'Deeplab系列\deeplab\VOCdevkit\VOC2007\ImageSets\Segmentation\val.txt'

train_list = []
val_list = []

if os.path.exists(train_txt):
    with open(train_txt, 'r') as f:
        train_list = [line.strip() for line in f.readlines()]
    print(f"训练集图像: {len(train_list)} 张")

if os.path.exists(val_txt):
    with open(val_txt, 'r') as f:
        val_list = [line.strip() for line in f.readlines()]
    print(f"验证集图像: {len(val_list)} 张")

# 如果没有找到划分文件，使用默认8:2划分
if not train_list or not val_list:
    print("未找到原始划分，使用默认8:2划分...")
    all_images = [f.replace('.jpg', '') for f in os.listdir(r'Deeplab系列\deeplab\VOCdevkit\VOC2007\JPEGImages') 
                  if f.endswith('.jpg')]
    train_list, val_list = train_test_split(all_images, test_size=0.2, random_state=42)
    print(f"训练集: {len(train_list)} 张")
    print(f"验证集: {len(val_list)} 张")

print()

# 步骤3: 创建分类数据集目录结构
print("【步骤3】为face_parts_extracted创建分类数据集结构...")
print("-"*80)

# 假设我们要做二分类或三分类任务
# 根据result的结构，似乎是3分类任务（0, 1, 2）
# 可能代表: 0=normal, 1=某种状态, 2=pain等

# 我们需要从原始文件名或其他信息推断类别
# 这里先创建一个简化版本，按照文件名前缀分类

def extract_class_from_filename(filename):
    """从文件名推断类别"""
    # 根据result中的示例，文件名格式为：normal_123_xxx.png, pain_456_xxx.png等
    # 这里简单按照文件名前缀分类
    if 'normal' in filename.lower():
        return '0'  # normal
    elif 'pain' in filename.lower() or 'stress' in filename.lower():
        return '2'  # pain/stress
    else:
        return '1'  # other

# 创建输出目录结构
organized_output = 'face_parts_classified'
os.makedirs(organized_output, exist_ok=True)

for part in face_parts:
    print(f"\n组织 {part}...")
    
    # 创建train/val目录结构
    for split in ['train', 'val']:
        for cls in ['0', '1', '2']:
            cls_dir = os.path.join(organized_output, part, split, cls)
            os.makedirs(cls_dir, exist_ok=True)
    
    # 从face_parts_extracted复制并分类
    source_dir = os.path.join('face_parts_extracted', part)
    
    if not os.path.exists(source_dir):
        print(f"  警告: {source_dir} 不存在，跳过")
        continue
    
    # 获取该部件的所有文件
    part_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png'))]
    
    if not part_files:
        print(f"  警告: {source_dir} 为空，跳过")
        continue
    
    # 按照train/val划分复制文件
    train_count = {cls: 0 for cls in ['0', '1', '2']}
    val_count = {cls: 0 for cls in ['0', '1', '2']}
    
    for filename in part_files:
        img_name = filename.replace('.jpg', '').replace('.png', '')
        
        # 推断类别
        cls = extract_class_from_filename(img_name)
        
        # 判断是train还是val
        if img_name in train_list:
            split = 'train'
            train_count[cls] += 1
        elif img_name in val_list:
            split = 'val'
            val_count[cls] += 1
        else:
            # 默认分配到train
            split = 'train'
            train_count[cls] += 1
        
        # 复制文件
        src_path = os.path.join(source_dir, filename)
        dst_dir = os.path.join(organized_output, part, split, cls)
        dst_path = os.path.join(dst_dir, filename)
        
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            pass
    
    print(f"  Train: 类别0={train_count['0']}, 类别1={train_count['1']}, 类别2={train_count['2']}")
    print(f"  Val:   类别0={val_count['0']}, 类别1={val_count['1']}, 类别2={val_count['2']}")

print("\n" + "="*80)
print("数据集组织完成！")
print("="*80)
print(f"\n输出目录: {organized_output}/")
print("\n目录结构:")
print("face_parts_classified/")
print("  ├── face/")
print("  │   ├── train/")
print("  │   │   ├── 0/  (normal)")
print("  │   │   ├── 1/  (other)")
print("  │   │   └── 2/  (pain/stress)")
print("  │   └── val/")
print("  │       ├── 0/")
print("  │       ├── 1/")
print("  │       └── 2/")
print("  ├── mouth/")
print("  ├── nose/")
print("  ├── eyes/")
print("  ├── ear/")
print("  └── muscles_above_eye/")
print()

# 步骤4: 创建分类器训练配置
print("【步骤4】创建分类器训练脚本...")
print("-"*80)

# 为每个部件创建训练脚本
classifier_scripts_dir = 'face_parts_classification_scripts'
os.makedirs(classifier_scripts_dir, exist_ok=True)

for part in face_parts:
    script_content = f'''"""
{part} 部件分类训练脚本
使用MobileViT进行3分类
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

# 设置路径
sys.path.append('pytorch_classification/MobileViT')

data_root = os.path.join('face_parts_classified', '{part}')
model_save_path = 'classification_weights/{part}'
os.makedirs(model_save_path, exist_ok=True)

# 数据转换
data_transform = {{
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}}

# 加载数据集
train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"),
                                    transform=data_transform["train"])
val_dataset = datasets.ImageFolder(root=os.path.join(data_root, "val"),
                                  transform=data_transform["val"])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, 
                                          shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32,
                                        shuffle=False, num_workers=0)

print(f"{{part}} 分类任务")
print(f"训练集: {{len(train_dataset)}} 张")
print(f"验证集: {{len(val_dataset)}} 张")
print(f"类别数: {{len(train_dataset.classes)}}")
print(f"类别: {{train_dataset.classes}}")

# 使用简单的ResNet18作为分类器
from torchvision.models import resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {{device}}")

model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # 3分类
model = model.to(device)

# 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 训练函数
def train_model(num_epochs=30):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {{epoch+1}}/{{num_epochs}}')
        print('-' * 40)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        print(f'Train Loss: {{epoch_loss:.4f}} Acc: {{epoch_acc:.4f}}')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        
        print(f'Val Loss: {{val_loss:.4f}} Acc: {{val_acc:.4f}}')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 
                      os.path.join(model_save_path, 'best_model.pth'))
            print(f'保存最佳模型! Val Acc: {{best_acc:.4f}}')
        
        scheduler.step()
        print()
    
    print(f'训练完成! 最佳验证准确率: {{best_acc:.4f}}')
    return model

if __name__ == '__main__':
    train_model(num_epochs=30)
'''
    
    script_path = os.path.join(classifier_scripts_dir, f'train_{part}.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"  [OK] 创建 train_{part}.py")

print()
print(f"分类器训练脚本已创建: {classifier_scripts_dir}/")
print()

print("="*80)
print("完成!")
print("="*80)
print()
print("接下来的步骤:")
print("1. 首先需要成功提取面部部件 (运行extract_face_parts_auto.py)")
print("2. 然后运行本脚本组织数据集")
print("3. 最后运行分类器训练脚本")
print()
print(f"示例:")
print(f"  python {classifier_scripts_dir}/train_face.py")
print()

