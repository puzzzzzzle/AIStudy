import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 生成合成数据
torch.manual_seed(42)  # 保证可重复性
X = torch.linspace(-5, 5, 100).view(-1, 1)
Y = 0.5 * X ** 2 + 2 * X + 3 + torch.randn(X.size()) * 0.5  # 带噪声的二次函数


# 定义神经网络模型
class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.net(x)


# 初始化模型、损失函数和优化器
model = RegressionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练循环
epochs = 1000
for epoch in range(epochs):
    # 前向传播
    predictions = model(X)

    # 计算损失
    loss = criterion(predictions, Y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每100次epoch打印损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 可视化结果
with torch.no_grad():
    predicted = model(X).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X.numpy(), Y.numpy(), alpha=0.6, label='Original Data')
plt.plot(X.numpy(), predicted, 'r', linewidth=2, label='Model Prediction')
plt.title('PyTorch Regression Demo')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()