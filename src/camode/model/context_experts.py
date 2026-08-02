import torch
from torch import nn
from .blocks import ConvBNAct, MLPHead


class ExpertTower(nn.Module):
    def __init__(self, in_channels, mid_channels=384):
        super().__init__()
        self.tower = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, 3, 1, 1),
            ConvBNAct(mid_channels, mid_channels, 3, 1, 1),
            ConvBNAct(mid_channels, mid_channels, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.tower(x)


class SceneContextExpert(nn.Module):
    def __init__(self, in_channels, num_scene_classes=365, kappa=56, dropout=0.2):
        super().__init__()
        self.tower = ExpertTower(in_channels)
        self.classifier = MLPHead(384, num_scene_classes, dropout=dropout, apply_softmax=True)
        self.projector = nn.Linear(num_scene_classes, kappa)

    def forward(self, feat):
        z = self.classifier(self.tower(feat))
        y = self.projector(z)
        return z, y


class ObjectContextExpert(nn.Module):
    def __init__(self, in_channels, num_object_classes=80, kappa=56, dropout=0.2):
        super().__init__()
        self.tower = ExpertTower(in_channels)
        self.classifier = MLPHead(384, num_object_classes, dropout=dropout, apply_softmax=True)
        self.projector = nn.Linear(num_object_classes, kappa)

    def forward(self, feat):
        z = self.classifier(self.tower(feat))
        y = self.projector(z)
        return z, y


class EmotionExpert(nn.Module):
    def __init__(self, in_channels, num_emotions=29, dropout=0.2):
        super().__init__()
        self.tower = ExpertTower(in_channels)
        self.logit_head = MLPHead(384, num_emotions, dropout=dropout, apply_softmax=False)
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    def forward(self, feat):
        logits = self.logit_head(self.tower(feat))
        temperature = torch.exp(self.log_sigma).clamp(min=1e-4)
        probs = torch.softmax(logits / temperature, dim=-1)
        return logits, probs
