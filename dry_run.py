import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

import torch
from torch.utils.data import DataLoader
from camode.config import ModelConfig, TrainConfig
from camode.data.synthetic_dataset import SyntheticBodilyEmotionDataset
from camode.model.camode import CAMoDE
from camode.training.trainer import Trainer
from camode.utils.seed import seed_everything


def main():
    seed_everything(42)
    mcfg = ModelConfig()
    tcfg = TrainConfig(batch_size=4, epochs=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_ds = SyntheticBodilyEmotionDataset(n_samples=24, image_size=mcfg.image_size, num_emotions=mcfg.num_emotions, seed=42)
    val_ds = SyntheticBodilyEmotionDataset(n_samples=8, image_size=mcfg.image_size, num_emotions=mcfg.num_emotions, seed=123)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False)

    model = CAMoDE(mcfg)
    trainer = Trainer(
        model=model,
        device=device,
        lr=tcfg.lr,
        momentum=tcfg.momentum,
        weight_decay=tcfg.weight_decay,
    )
    trainer.fit_priors(train_loader)
    train_loss = trainer.train_one_epoch(train_loader)
    val_loss = trainer.evaluate(val_loader)

    batch = next(iter(val_loader))
    with torch.no_grad():
        out = trainer.model(batch['image'].to(device))

    print({
        'device': device,
        'train_loss': round(train_loss, 6),
        'val_loss': round(val_loss, 6),
        'output_shape': tuple(out['y_hat'].shape),
        'priors_fitted': trainer.model.fusion.priors.fitted,
    })


if __name__ == '__main__':
    main()
