import torch

from .losses import CAMoDELoss


class Trainer:
    """
    Default optimisation settings:
    - SGD
    - Initial learning rate: 1e-2
    - Momentum: 0.9
    - Weight decay: 5e-3
    - LR decay: multiply by 0.1 every 45 epochs
    """

    def __init__(
        self,
        model,
        device="cpu",
        lr=1e-2,
        momentum=0.9,
        weight_decay=5e-3,
        lr_step_size=45,
        lr_gamma=0.1,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.criterion = CAMoDELoss()

        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        if not trainable_parameters:
            raise ValueError(
                "No trainable parameters were found in the model."
            )

        self.optimizer = torch.optim.SGD(
            trainable_parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_step_size,
            gamma=lr_gamma,
        )

    def fit_priors(self, train_loader):
        """
        Fit P+, P-, pi, q exclusively on the training split.

        Call once before training. Do not fit priors with validation
        or test samples, since that would leak evaluation information.
        """
        self.model.fusion.priors.fit_from_loader(
            self.model,
            train_loader,
            self.device,
        )

        if not bool(self.model.fusion.priors.fitted.item()):
            raise RuntimeError("Context priors were not fitted.")

    def train_one_epoch(self, loader):
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in loader:
            x = batch["image"].to(
                self.device,
                non_blocking=True,
            )
            y = batch["emotion"].to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad(set_to_none=True)

            output = self.model(x)
            loss = self.criterion(output["y_hat"], y)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss detected: {loss.item()}"
                )

            loss.backward()
            self.optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size

        return total_loss / max(total_samples, 1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        for batch in loader:
            x = batch["image"].to(
                self.device,
                non_blocking=True,
            )
            y = batch["emotion"].to(
                self.device,
                non_blocking=True,
            )

            output = self.model(x)
            loss = self.criterion(output["y_hat"], y)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite validation loss detected: {loss.item()}"
                )

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / max(total_samples, 1)

    def step_scheduler(self):
        """
        Call exactly once after each completed training epoch.

        With the default StepLR configuration, the learning rate changes:
        - Epochs 1-45:  1e-2
        - Epochs 46-90: 1e-3
        """
        self.scheduler.step()

    def current_lr(self):
        """Return the current optimizer learning rate."""
        return self.optimizer.param_groups[0]["lr"]
