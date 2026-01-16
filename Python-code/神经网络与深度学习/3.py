from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch import nn
import matplotlib.pyplot as plt

#root = "D:/HuaweiMoveData/Users/24901/Desktop/大学存档/神经网络与深度学习"
root = "./data"  # 数据集将保存到项目根目录下的 `data` 文件夹中
# 数据加载和预处理
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
test_data = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练函数（带早停）
def train_with_early_stopping(model, train_loader, test_loader, learning_rate=0.001, num_epochs=5, patience=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # 早停逻辑
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return train_losses, test_losses

# 初始化模型并训练
model = MLP()
train_losses, test_losses = train_with_early_stopping(model, train_loader, test_loader)

# 绘制损失曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss with Early Stopping')
plt.legend()
plt.show()

# 任务2：定义带权重衰减的优化器
def train_with_weight_decay(model, train_loader, test_loader, learning_rate=0.001, weight_decay=0.01, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return train_losses, test_losses

# 测试不同权重衰减系数
lambdas = [0, 0.01, 0.1]
train_losses_list = []
test_losses_list = []

for lam in lambdas:
    model = MLP()
    train_losses, test_losses = train_with_weight_decay(model, train_loader, test_loader, weight_decay=lam)
    train_losses_list.append(train_losses)
    test_losses_list.append(test_losses)

# 绘制不同 lambda 下的损失曲线
plt.figure(figsize=(10, 6))
for i, lam in enumerate(lambdas):
    plt.plot(train_losses_list[i], label=f'Train Loss (λ={lam})')
    plt.plot(test_losses_list[i], label=f'Test Loss (λ={lam})', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Effect of Weight Decay on Loss')
plt.legend()
plt.show()


# 任务三：定义带 Dropout 的 MLP 模型
class MLPWithDropout(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(MLPWithDropout, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 测试不同 Dropout 概率
dropout_probs = [0, 0.5]
train_losses_list = []
test_losses_list = []

for prob in dropout_probs:
    model = MLPWithDropout(dropout_prob=prob)
    train_losses, test_losses = train_with_early_stopping(model, train_loader, test_loader)
    train_losses_list.append(train_losses)
    test_losses_list.append(test_losses)

# 绘制不同 Dropout 概率下的损失曲线
plt.figure(figsize=(10, 6))
for i, prob in enumerate(dropout_probs):
    plt.plot(train_losses_list[i], label=f'Train Loss (Dropout={prob})')
    plt.plot(test_losses_list[i], label=f'Test Loss (Dropout={prob})', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Effect of Dropout on Loss')
plt.legend()
plt.show()


# 任务4：定义带 Dropout 和权重衰减的模型
model = MLPWithDropout(dropout_prob=0.5)
train_losses, test_losses = train_with_weight_decay(model, train_loader, test_loader, weight_decay=0.01)

# 绘制损失曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss with Dropout and Weight Decay')
plt.legend()
plt.show()