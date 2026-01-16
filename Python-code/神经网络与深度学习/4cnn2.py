import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#任务 1：将 LeNet 中的平均池化层替换为最大池化层
#LeNet 是一个经典的卷积神经网络，最初使用平均池化层（Average Pooling）。我们可以将其替换为最大池化层（Max Pooling）。
# 定义 LeNet 模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # 输入通道1，输出通道6
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)  # 展平
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

train(model, train_loader, criterion, optimizer, epochs=5)

# 测试模型
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")

test(model, test_loader)


#任务 2：调整 LeNet 以提高准确率
class ImprovedLeNet(nn.Module):
    def __init__(self):
        super(ImprovedLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 调整卷积核大小和输出通道数
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 增加输出通道数
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 256)  # 增加全连接层大小
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 使用改进后的模型
model = ImprovedLeNet()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 调整学习率
train(model, train_loader, criterion, optimizer, epochs=10)  # 增加训练轮数
test(model, test_loader)



#任务 3：可视化 LeNet 第一层和第二层的激活值
#我们可以通过钩子（Hook）来提取并可视化卷积层的激活值。
# 定义钩子函数
activations = {}

def hook_fn(module, input, output):
    activations[module] = output

# 注册钩子
model.conv1.register_forward_hook(hook_fn)
model.conv2.register_forward_hook(hook_fn)

# 输入一张图片
image, _ = train_data[0]
image = image.unsqueeze(0)  # 增加 batch 维度
output = model(image)

# 可视化第一层和第二层的激活值
def visualize_activations(activations, layer_name):
    activation = activations[layer_name].squeeze().detach().numpy()
    plt.figure(figsize=(10, 5))
    for i in range(activation.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(activation[i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f"Activations for {layer_name}")
    plt.show()

visualize_activations(activations, model.conv1)
visualize_activations(activations, model.conv2)