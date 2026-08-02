import torch
from torch import nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, apply_softmax=False):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.apply_softmax = apply_softmax

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        if self.apply_softmax:
            x = torch.softmax(x, dim=-1)
        return x
