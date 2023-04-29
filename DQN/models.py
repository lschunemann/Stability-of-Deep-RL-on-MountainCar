from torch import nn
import torch


class DQN(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=2, padding=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=(0, 1))
        self.batch2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=(1, 2), padding=(0, 1))
        self.batch3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=(1, 2), padding=(0, 1))
        self.batch4 = nn.BatchNorm2d(32)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.linear = nn.Linear(192, 32)     # (2880, 256)
        self.out = nn.Linear(32, outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.pool1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.pool2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.pool3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.pool4(x)
        x = nn.ReLU()(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = nn.ReLU()(x)
        x = self.out(x)
        return x


class DQN_paper(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear = nn.Linear(4224, 512)     # (2880, 256) for old 2 layer, 2592 for paper conv
        self.out = nn.Linear(512, outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = nn.ReLU()(x)
        x = self.out(x)
        return x


class DQN_square(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear = nn.Linear(3136, 512)
        self.out = nn.Linear(512, outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = nn.ReLU()(x)
        x = self.out(x)
        return x


class Combine(nn.Module):
    def __call__(self, val, adv):
        adv = adv - torch.mean(adv, dim=1, keepdim=True)
        return val + adv


class DQN_dueling(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.value = nn.Linear(3136, 512)
        self.advantage = nn.Linear(3136, 512)
        self.value_out = nn.Linear(512, 1)
        self.advantage_out = nn.Linear(512, outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.flatten(x)
        val = self.value(x)
        val = nn.ReLU()(val)
        val = self.value_out(val)
        adv = self.advantage(x)
        adv = nn.ReLU()(adv)
        adv = self.advantage_out(adv)
        x = Combine()(val, adv)
        return x


class DQN_paper_old(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.linear = nn.Linear(3328, 256)     # (2880, 256) for old 2 layer, 2592 for paper conv
        self.out = nn.Linear(256, outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = nn.ReLU()(x)
        x = self.out(x)
        return x
