import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

import json
import torch
from torch.utils.data import DataLoader
from camode.config import ModelConfig, TrainConfig
from camode.data.synthetic_dataset import SyntheticBodilyEmotionDataset
from camode.model.camode import CAMoDE
from camode.training.trainer import Trainer
from camode.utils.seed import seed_everything


def run_case(batch_size, n_samples, device):
    mcfg = ModelConfig()
    tcfg = TrainConfig(batch_size=batch_size, epochs=1)

    dataset = SyntheticBodilyEmotionDataset(
        n_samples=n_samples,
        image_size=mcfg.image_size,
        num_emotions=mcfg.num_emotions,
        seed=40 + batch_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=tcfg.batch_size,
        shuffle=False,
    )

    model = CAMoDE(mcfg)

    trainer = Trainer(
        model=model,
        device=device,
        lr=tcfg.lr,
        momentum=tcfg.momentum,
        weight_decay=tcfg.weight_decay,
    )

    trainer.fit_priors(loader)

    train_loss = trainer.train_one_epoch(loader)
    eval_loss = trainer.evaluate(loader)

    sample = next(iter(loader))
    trainer.model.eval()

    with torch.no_grad():
        out = trainer.model(sample["image"].to(device))

    checkpoint_path = Path("camode_tmp_state.pt")
    torch.save(trainer.model.state_dict(), checkpoint_path)

    reloaded = CAMoDE(mcfg).to(device)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    reloaded.load_state_dict(state_dict)
    reloaded.eval()

    with torch.no_grad():
        out_reloaded = reloaded(sample["image"].to(device))

    max_difference = (
        out["y_hat"] - out_reloaded["y_hat"]
    ).abs().max().item()

    checkpoint_path.unlink(missing_ok=True)

    return {
        "batch_size": batch_size,
        "n_samples": n_samples,
        "train_loss": float(train_loss),
        "eval_loss": float(eval_loss),
        "output_shape": list(out["y_hat"].shape),
        "reload_max_abs_diff": float(max_difference),
        "finite_output": bool(
            torch.isfinite(out["y_hat"]).all().item()
        ),
        "finite_gate": bool(
            torch.isfinite(out["Q"]).all().item()
        ),
    }


def main():
    seed_everything(123)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cases = [(1, 8), (2, 10), (4, 16), (8, 24)]
    results = [run_case(bs, ns, device) for bs, ns in cases]
    report = {
        'device': device,
        'all_passed': all(r['finite_output'] and r['finite_gate'] and r['reload_max_abs_diff'] < 1e-5 for r in results),
        'cases': results,
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
