import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride):
    # keeps the original dimentions
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def shortcut(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_channels))


class fc_relu_block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(*args, **kwargs), nn.LeakyReLU())

    def forward(self, x):
        x = self.block(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # conv layer 1
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.LeakyReLU()
        # conv layer 2 [stride=1]
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.LeakyReLU()

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=(1, 1),
                          stride=1,
                          bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out += self.shortcut(x)

        out = self.activation2(out)
        return out

    def _apply_downsample(self):
        return self.in_channels != self.out_channels


class Resnet18(nn.Module):
    def __init__(self, history_length=0, n_classes=4):
        super().__init__()
        w_after_conv = 25  #50
        h_after_conv = 37  #75
        self.init_block = nn.Sequential(
            nn.Conv2d(in_channels=history_length + 1,
                      out_channels=8,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.BatchNorm2d(8))
        self.block1 = BasicBlock(8, 16)
        self.block2 = BasicBlock(16, 32)
        self.block3 = BasicBlock(32, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = fc_relu_block(w_after_conv * h_after_conv * 64, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.fcnorm = nn.BatchNorm1d(32)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.init_block(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        # print(f'SHAPE OF X BEFORE FC: {x.shape}')
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fcnorm(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    net = Resnet18(history_length=3, n_classes=4)
    data = torch.ones((100, 4, 200, 300))
    net(data)
