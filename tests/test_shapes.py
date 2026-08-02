import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch

from camode.config import ModelConfig
from camode.model.camode import CAMoDE


def test_model_output_shapes():
    cfg = ModelConfig()
    model = CAMoDE(cfg)
    model.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        out = model(x)

    assert out["feat"].shape == (2, 528, 14, 14)
    assert out["z_scene"].shape == (2, 365)
    assert out["z_object"].shape == (2, 80)
    assert out["emotion_logits"].shape == (2, 29)
    assert out["y_hat"].shape == (2, 29)

    assert torch.isfinite(out["y_hat"]).all()
    assert torch.isfinite(out["Q"]).all()