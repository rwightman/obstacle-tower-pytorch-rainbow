import math
import torch
from torch import nn
from torch.nn import functional as F


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(self, args, action_space, act_fn=F.relu):
        super().__init__()
        self.atoms = args.atoms
        self.action_space = action_space
        self.act_fn = act_fn

        self.conv1 = nn.Conv2d(3, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc_h_v = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, action_space.n * self.atoms, std_init=args.noisy_std)

    def forward(self, x, log=False):
        x = self.act_fn(self.conv1(x))
        x = self.act_fn(self.conv2(x))
        x = self.act_fn(self.conv3(x))
        x = x.view(-1, 3136)

        v = self.fc_z_v(self.act_fn(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(self.act_fn(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space.n, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams

        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.drop_rate > 0.:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResDQN(nn.Module):
    def __init__(self, args, action_space, act_fn=F.relu):
        super().__init__()
        self.atoms = args.atoms
        self.action_space = action_space
        self.act_fn = act_fn
        self.inplanes = 64
        self.linear_size = 6400

        self.conv1 = nn.Conv2d(3, 32, 8, stride=4, padding=3)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        #self.conv3 = nn.Conv2d(64, 64, 4, stride=2)
        #self.conv4 = nn.Conv2d(64, 64, 3)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=2)


        self.fc_h_v = NoisyLinear(self.linear_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(self.linear_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, action_space.n * self.atoms, std_init=args.noisy_std)

    def _make_layer(self, block, planes, blocks, stride=1, drop_rate=0.):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, drop_rate)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, log=False):
        x = self.act_fn(self.conv1(x))
        x = self.act_fn(self.conv2(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, self.linear_size)

        v = self.fc_z_v(self.act_fn(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(self.act_fn(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space.n, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams

        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()
