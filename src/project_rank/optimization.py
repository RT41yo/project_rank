"""Explicit, config-driven training recipes; legacy configs still use SGD."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "sgd")).lower()
    lr = float(config.get("learning_rate", 0.1))
    decay = float(config.get("weight_decay", 5e-4))
    exclusions = bool(config.get("exclude_bias_norm_tokens_from_weight_decay", False))
    if name == "sgd":
        if exclusions:
            raise ValueError("Weight-decay exclusions are supported only for AdamW")
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=float(config.get("momentum", 0.9)),
            weight_decay=decay, nesterov=bool(config.get("nesterov", True)),
        )
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer: {name}")
    if exclusions:
        decayed, exempt = [], []
        for parameter_name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            # All biases and LayerNorm affine weights are one-dimensional.
            group = exempt if parameter.ndim <= 1 or parameter_name in {
                "cls_token", "pos_embed",
            } else decayed
            group.append(parameter)
        parameters = [
            {"params": decayed, "weight_decay": decay},
            {"params": exempt, "weight_decay": 0.0},
        ]
    else:
        parameters = model.parameters()
    return torch.optim.AdamW(
        parameters, lr=lr, weight_decay=decay,
        betas=tuple(config.get("adam_betas", (0.9, 0.999))),
        eps=float(config.get("adam_eps", 1e-8)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: Mapping[str, Any], epochs: int,
) -> Any:
    name = str(config.get("scheduler", "cosine"))
    warmup = config.get("warmup_epochs", 0)
    if type(warmup) is not int or not 0 <= warmup < epochs:
        raise ValueError("warmup_epochs must be an integer in [0, epochs)")
    if name == "none":
        if warmup:
            raise ValueError("Warmup requires the cosine scheduler")
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name}")
    if warmup == 0:
        # Keep the original scheduler and its exact LR sequence for ResNet/SGD.
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start = float(config.get("warmup_start_factor", 0.01))
    if not 0 < start <= 1:
        raise ValueError("warmup_start_factor must be in (0, 1]")

    def multiplier(epoch_index: int) -> float:
        # Epochs 1..warmup use a linear ramp; epoch warmup+1 reaches peak LR.
        if epoch_index < warmup:
            return start + (1 - start) * epoch_index / warmup
        progress = min(1.0, (epoch_index - warmup) / (epochs - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
