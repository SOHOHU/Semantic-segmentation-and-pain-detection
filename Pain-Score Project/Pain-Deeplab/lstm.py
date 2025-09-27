
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torchvision import transforms
from PIL import Image
import torch
import torch.optim as optim


class ImageSequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

        self.img_paths = [os.path.join(root_dir, file) for file in sorted(os.listdir(root_dir))]
        label_set = sorted(set(file.split('_')[0] for file in os.listdir(root_dir)))
        self.label_to_int = {label: idx for idx, label in enumerate(label_set)}

        self.labels = [self.label_to_int[file.split('_')[0]] for file in os.listdir(root_dir)[::sequence_length]]

    def __getitem__(self, idx):
        sequence_paths = self.img_paths[idx * self.sequence_length: (idx + 1) * self.sequence_length]
        sequence = [Image.open(img_path) for img_path in sequence_paths]

        if self.transform:
            sequence = [self.transform(img) for img in sequence]

        label = self.labels[idx]
        return torch.stack(sequence), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.img_paths) // self.sequence_length

    def _extract_label(self, img_path):
        basename = os.path.basename(img_path)
        label = basename.split('_')[0]
        return self.label_to_int[label]


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


# LSTM 模型
class ImageSequenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ImageSequenceClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out



# 超参数
input_size = 512 * 512  # 根据你的图像尺寸和转换来设定
hidden_size = 256
num_layers = 2
num_classes = 2  # 你的类别数量

# 创建模型
model = ImageSequenceClassifier(input_size, hidden_size, num_layers, num_classes).cuda()

sequence_length = 12
dataset = ImageSequenceDataset(root_dir='miou_out/detection-results', sequence_length=sequence_length,
                               transform=transform)

dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
test_size = dataset_size - train_size

train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size, dataset_size))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 假设 'dataset' 是你已经创建好的 ImageSequenceDataset 实例
num_classes_in_dataset = len(dataset.label_to_int)
print(f'There are {num_classes_in_dataset} unique classes in the dataset')


# # 训练模型
# num_epochs = 50
# for epoch in range(num_epochs):
#     for sequences, labels in train_loader:
#         # 将序列展平为 (batch_size, sequence_length * feature_size)
#         sequences = sequences.view(sequences.size(0), sequence_length, -1).cuda()
#         labels = labels.cuda()
# 
#         # 前向传播
#         outputs = model(sequences)
#         loss = criterion(outputs, labels)
# 
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
# 
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# 
# print("训练完成")

import matplotlib.pyplot as plt

# 初始化记录列表
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

num_epochs = 50
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for sequences, labels in train_loader:
        sequences = sequences.view(sequences.size(0), sequence_length, -1).cuda()
        labels = labels.cuda()

        # 前向传播
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)

    # 测试阶段
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.view(sequences.size(0), sequence_length, -1).cuda()
            labels = labels.cuda()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(100 * correct / total)

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, '
        f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.2f}%')

print("训练完成")

import pandas as pd

# 训练完成后，保存数据
results = pd.DataFrame({
    'Train Loss': train_losses,
    'Train Accuracy': train_accuracies,
    'Test Loss': test_losses,
    'Test Accuracy': test_accuracies
})

# 将结果保存为CSV文件
results.to_csv(f'training_results_{hidden_size}_{num_layers}.csv', index=False)
print("结果已保存到 'training_results.csv'")

# 绘制损失和准确率图
plt.figure(figsize=(12, 6))
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
plt.plot(range(num_epochs), test_accuracies, label='Test Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()

# 将模型设置为评估模式
model.eval()

# 用于存储预测和真实标签
all_predictions = []
all_labels = []

with torch.no_grad():  # 在这个块中，不计算梯度
    for sequences, labels in test_loader:
        sequences = sequences.view(sequences.size(0), sequence_length, -1).cuda()
        labels = labels.cuda()
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='binary')  # 默认设置为二分类问题
recall = recall_score(all_labels, all_predictions, average='binary')
f1 = f1_score(all_labels, all_predictions, average='binary')

print(f"测试集上的准确率: {accuracy:.4f}")
print(f"测试集上的精确率: {precision:.4f}")
print(f"测试集上的召回率: {recall:.4f}")
print(f"测试集上的F1-score: {f1:.4f}")

# 保存模型
torch.save(model.state_dict(), f'model_data/lstm_{hidden_size}_{num_layers}.pth')

