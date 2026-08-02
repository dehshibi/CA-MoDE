import torch
from torch import nn
from .blocks import ConvBNAct


class SharedFeatureEncoder(nn.Module):
    def __init__(self, out_channels=528):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBNAct(64, 128, 3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBNAct(128, 256, 3, stride=1, padding=1),
            ConvBNAct(256, out_channels, 3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((14, 14)),
        )

    def forward(self, x):
        return self.stem(x)
