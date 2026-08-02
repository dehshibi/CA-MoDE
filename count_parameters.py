import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import torch

from camode.config import ModelConfig
from camode.model.camode import CAMoDE


def count_parameters(module):
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    frozen = total - trainable
    return total, trainable, frozen


def format_count(value):
    return f"{value:,}"


def format_millions(value):
    return f"{value / 1_000_000:.3f} M"


def print_row(name, total, trainable, frozen):
    print(
        f"{name:<24}"
        f"{format_count(total):>15}  "
        f"{format_count(trainable):>15}  "
        f"{format_count(frozen):>15}"
    )


def main():
    cfg = ModelConfig()
    model = CAMoDE(cfg)

    total, trainable, frozen = count_parameters(model)

    print("\nCA-MoDE parameter report")
    print("=" * 72)
    print(
        f"{'Module':<24}"
        f"{'Total':>15}  "
        f"{'Trainable':>15}  "
        f"{'Frozen':>15}"
    )
    print("-" * 72)

    for name, module in model.named_children():
        module_total, module_trainable, module_frozen = count_parameters(
            module
        )
        print_row(
            name,
            module_total,
            module_trainable,
            module_frozen,
        )

    print("-" * 72)
    print_row("CA-MoDE total", total, trainable, frozen)

    print("\nSummary")
    print(f"Total parameters:     {format_count(total)} ({format_millions(total)})")
    print(
        f"Trainable parameters: {format_count(trainable)} "
        f"({format_millions(trainable)})"
    )
    print(
        f"Frozen parameters:    {format_count(frozen)} "
        f"({format_millions(frozen)})"
    )

    if total > 0:
        trainable_percent = 100.0 * trainable / total
        frozen_percent = 100.0 * frozen / total

        print(f"Trainable share:      {trainable_percent:.2f}%")
        print(f"Frozen share:         {frozen_percent:.2f}%")

    print("\nTrainable parameter tensors")
    print("-" * 72)

    for name, parameter in model.named_parameters():
        status = "trainable" if parameter.requires_grad else "frozen"

        print(
            f"{status:<10} "
            f"{name:<60} "
            f"shape={str(tuple(parameter.shape)):<20} "
            f"numel={parameter.numel():,}"
        )

    print("\nNon-parameter buffers")
    print("-" * 72)

    for name, buffer in model.named_buffers():
        print(
            f"{name:<60} "
            f"shape={str(tuple(buffer.shape)):<20} "
            f"numel={buffer.numel():,}"
        )


if __name__ == "__main__":
    main()