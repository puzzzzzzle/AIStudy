import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchmetrics import Accuracy

# 定义 PyTorch Lightning 模型
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.layer2 = nn.Linear(128, 10)
        self.accuracy = Accuracy(task='multiclass', num_classes=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# 定义数据模块
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        # 下载数据集
        MNIST('./data', train=True, download=True)
        MNIST('./data', train=False, download=True)

    def setup(self, stage=None):
        # 分配数据集
        if stage == 'fit' or stage is None:
            self.mnist_train = MNIST('./data', train=True, transform=self.transform)
            self.mnist_val = MNIST('./data', train=False, transform=self.transform)
        if stage == 'test':
            self.mnist_test = MNIST('./data', train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

# 训练模型
if __name__ == '__main__':
    # 初始化数据模块和模型
    dm = MNISTDataModule(batch_size=32)
    model = LitModel()

    # 初始化训练器
    trainer = Trainer(
        max_epochs=3,
        accelerator='auto',
        devices='auto'
    )

    # 训练和验证
    trainer.fit(model, dm)

    # 测试
    trainer.test(model, datamodule=dm)