from dataclasses import dataclass


@dataclass
class ModelConfig:
    image_size: int = 224
    num_emotions: int = 29
    num_scene_classes: int = 365
    num_object_classes: int = 80
    shared_channels: int = 528
    kappa: int = 56
    lambda_scale: float = 0.2
    gate_threshold: float = 0.5
    gate_sharpness: float = 10.0
    dropout: float = 0.2


@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 5e-3
    epochs: int = 90
    device: str = "cuda"
    seed: int = 42
