import torch.nn as nn
import torch
import torch.nn.functional as F


class NormActConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.norm_act_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.norm_act_conv(x)


class NormActLin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm_act_conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, out_channels)
        )

    def forward(self, x):
        return self.norm_act_conv(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            NormActConv(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            NormActConv(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.linear = nn.Sequential(
            nn.Linear(32*96*96, 1024),
            NormActLin(1024, 256),
            NormActLin(256, 32),
            NormActLin(32, 8)
        )


    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return F.softmax(x, -1)