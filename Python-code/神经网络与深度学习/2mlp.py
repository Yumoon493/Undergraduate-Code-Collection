import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 设置数据集保存路径
#root = "D:/HuaweiMoveData/Users/24901/Desktop/大学存档/神经网络与深度学习"
root = "./data"  # 数据集将保存到项目根目录下的 `data` 文件夹中
# 数据加载和预处理
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
test_data = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, num_hiddens, num_layers=1):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        input_size = 28 * 28
        for _ in range(num_layers):
            self.layers.append(nn.Linear(input_size, num_hiddens))
            self.layers.append(nn.ReLU())
            input_size = num_hiddens
        self.layers.append(nn.Linear(input_size, 10))

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return x

# 训练函数
def train(model, train_loader, test_loader, learning_rate, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        print(f"Train Loss: {train_losses[-1]}")

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        test_losses.append(test_loss / len(test_loader))
        print(f"Test Loss: {test_losses[-1]}")

    return train_losses, test_losses

# 任务1：修改隐藏单元数
num_hiddens_list = [64, 128]
train_losses_list = []
test_losses_list = []

for num_hiddens in num_hiddens_list:
    print(f"Training with num_hiddens={num_hiddens}")
    model = MLP(num_hiddens=num_hiddens)
    train_losses, test_losses = train(model, train_loader, test_loader, learning_rate=0.001, num_epochs=5)
    train_losses_list.append(train_losses)
    test_losses_list.append(test_losses)

# 绘制任务1结果
plt.figure(figsize=(10, 6))
for i, num_hiddens in enumerate(num_hiddens_list):
    plt.plot(train_losses_list[i], label=f'Train Loss (num_hiddens={num_hiddens})')
    plt.plot(test_losses_list[i], label=f'Test Loss (num_hiddens={num_hiddens})', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Effect of num_hiddens on Loss')
plt.legend()
plt.show()

# 任务2：修改学习率
learning_rates = [0.01, 0.1]
train_losses_list = []
test_losses_list = []

for lr in learning_rates:
    print(f"Training with learning_rate={lr}")
    model = MLP(num_hiddens=128)
    train_losses, test_losses = train(model, train_loader, test_loader, learning_rate=lr, num_epochs=5)
    train_losses_list.append(train_losses)
    test_losses_list.append(test_losses)

# 绘制任务2结果
plt.figure(figsize=(10, 6))
for i, lr in enumerate(learning_rates):
    plt.plot(train_losses_list[i], label=f'Train Loss (lr={lr})')
    plt.plot(test_losses_list[i], label=f'Test Loss (lr={lr})', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Effect of Learning Rate on Loss')
plt.legend()
plt.show()

# 任务3：增加隐藏层数
num_layers_list = [1, 2, 3, 4]
train_losses_list = []
test_losses_list = []

for num_layers in num_layers_list:
    print(f"Training with num_layers={num_layers}")
    model = MLP(num_hiddens=128, num_layers=num_layers)
    train_losses, test_losses = train(model, train_loader, test_loader, learning_rate=0.001, num_epochs=5)
    train_losses_list.append(train_losses)
    test_losses_list.append(test_losses)

# 绘制任务3结果
plt.figure(figsize=(10, 6))
for i, num_layers in enumerate(num_layers_list):
    plt.plot(train_losses_list[i], label=f'Train Loss (num_layers={num_layers})')
    plt.plot(test_losses_list[i], label=f'Test Loss (num_layers={num_layers})', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Effect of Number of Hidden Layers on Loss')
plt.legend()
plt.show()