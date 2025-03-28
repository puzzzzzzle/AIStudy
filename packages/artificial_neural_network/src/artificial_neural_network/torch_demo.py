import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm

# 设备检测
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 数据集加载
train_set = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_set = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# MPS需要设置num_workers=0
num_workers = 0 if device.type == "mps" else 4
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers)


# 定义神经网络模型
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练函数
def train(model, device, loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader):
        # 处理MPS设备的数据类型问题
        if device.type == "mps":
            images = images.to(device).to(torch.float32)
            labels = labels.to(device).to(torch.long)
        else:
            images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# 测试函数
def test(model, device, loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(labels).sum().item()

    return 100.0 * correct / len(loader.dataset)


# 训练循环
def main():
    epochs = 5
    for epoch in range(epochs):
        start_time = time()
        train_loss = train(model, device, train_loader, criterion, optimizer)
        test_acc = test(model, device, test_loader)
        elapsed = time() - start_time

        print(f"Epoch {epoch + 1}/{epochs} - {elapsed:.2f}s")
        print(f"Train loss: {train_loss:.4f} | Test accuracy: {test_acc:.2f}%")

    # 保存模型（可选）
    # torch.save(model.state_dict(), "mnist_model.pth")

if __name__ == '__main__':
    main()