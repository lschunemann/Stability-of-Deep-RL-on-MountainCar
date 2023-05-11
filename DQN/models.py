import math
import torch.nn.functional as F
import torch
import torch.nn as nn


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
        self.linear = nn.Linear(192, 512)     # (2880, 256)
        self.out = nn.Linear(512, outputs)

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


class MLP(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28224, 512)  # 903168
        self.linear2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, outputs)

    def forward(self, x):
        x = self.flatten(x)
        x = nn.ReLU(self.linear1(x))
        x = nn.ReLU(self.linear2(x))
        x = self.out(x)
        return x


class MLP_state(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.linear1 = nn.Linear(2, 512)
        self.linear2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, outputs)

    def forward(self, x):
        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))
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


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_0=0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_0 = std_0

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_0 / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_0 / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def sample_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)


class NoisyNet_Dueling(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.value = NoisyLinear(3136, 512)
        self.advantage = NoisyLinear(3136, 512)
        self.value_out = NoisyLinear(512, 1)
        self.advantage_out = NoisyLinear(512, outputs)

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

    def act(self, state):
        q = self.forward(state)
        action = q.max(1)[1].item()
        return action

    def reset_noise(self):
        self.value.reset_noise()
        self.value_out.reset_noise()
        self.advantage.reset_noise()
        self.advantage_out.reset_noise()

    def sample_noise(self):
        self.value.sample_noise()
        self.value_out.sample_noise()
        self.advantage.sample_noise()
        self.advantage_out.sample_noise()


class NoisyNet(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.noisy1 = NoisyLinear(3136, 512)
        self.noisy2 = NoisyLinear(512, outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.flatten(x)
        x = self.noisy1(x)
        x = nn.ReLU()(x)
        x = self.noisy2(x)
        return x

    def act(self, state):
        q = self.forward(state)
        action = q.max(1)[1].item()
        return action

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

    def sample_noise(self):
        self.noisy1.sample_noise()
        self.noisy2.sample_noise()


class Categorical_DQN(nn.Module):
    def __init__(self, outputs, atoms=51):
        super().__init__()
        self.atoms = atoms
        self.outputs = outputs
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear = nn.Linear(3136, 512)
        self.out = nn.Linear(512,  self.outputs * self.atoms)

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
        x = F.softmax(x.view(-1, self.outputs, self.atoms), dim=2)
        return x


class Rainbow_DQN(nn.Module):
    def __init__(self, outputs, atoms=51):
        super().__init__()
        self.atoms = atoms
        self.outputs = outputs
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.value = NoisyLinear(3136, 512)
        self.advantage = NoisyLinear(3136, 512)
        self.value_out = NoisyLinear(512, self.atoms)
        self.advantage_out = NoisyLinear(512, self.outputs * self.atoms)

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
        val = self.value_out(val).view(-1, 1, self.atoms)
        adv = self.advantage(x)
        adv = nn.ReLU()(adv)
        adv = self.advantage_out(adv).view(-1, self.outputs, self.atoms)
        x = Combine()(val, adv).view(-1, 1, self.atoms)
        x = F.softmax(x.view(-1, self.num_actions, self.atoms), dim=2)
        return x

    def sample_noise(self):
        self.value.sample_noise()
        self.value_out.sample_noise()
        self.advantage.sample_noise()
        self.advantage_out.sample_noise()


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


class DRQN(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(10, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.lstm = nn.LSTM(3136, 512, bidirectional=False)
        self.out = nn.Linear(512, outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.flatten(x)
        x, _ = self.lstm(x)
        x = self.out(x)
        return x
