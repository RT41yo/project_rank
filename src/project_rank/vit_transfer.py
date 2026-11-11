"""ViT-specific physical neuron transfer; widths are chosen separately."""
from __future__ import annotations

import random
from collections.abc import Mapping

import torch

from project_rank.models.vit_tiny import ViTTinyCIFAR


@torch.no_grad()
def transfer_vit_neurons(
    source: ViTTinyCIFAR, target: ViTTinyCIFAR,
    selected: Mapping[str, list[int]],
) -> None:
    if source.spec != target.spec:
        raise ValueError("Transfer requires identical specs except MLP widths")
    expected = {f"block{i + 1}" for i in range(source.spec.depth)}
    if set(selected) != expected:
        raise ValueError("Expected selected indices for every block")
    # Validate every block BEFORE modifying the target.
    for i, (old, new) in enumerate(zip(source.inner_widths, target.inner_widths, strict=True)):
        indices = selected[f"block{i + 1}"]
        if (new > old or len(indices) != new or len(set(indices)) != new
                or indices != sorted(indices)
                or any(type(j) is not int or not 0 <= j < old for j in indices)):
            raise ValueError(f"Invalid neuron indices in block{i + 1}")
    state = {}
    for name, value in source.state_dict().items():
        if name.startswith("blocks."):
            parts = name.split(".")
            indices = selected[f"block{int(parts[1]) + 1}"]
            if parts[2] == "fc1":
                value = value[indices]
            elif parts[2:] == ["fc2", "weight"]:
                value = value[:, indices]
        state[name] = value
    target.load_state_dict(state)


def select_vit_neurons(
    source: ViTTinyCIFAR, target_widths: tuple[int, ...] | list[int],
    method: str = "l1", seed: int = 2001,
    covariances: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, list[int]]:
    widths = source.spec.widths(target_widths)
    if method not in {"l1", "random", "erank_subspace"}:
        raise ValueError(f"Unknown transfer method: {method}")
    rng = random.Random(seed)
    selected = {}
    for i, (block, keep) in enumerate(zip(source.blocks, widths, strict=True)):
        name = f"block{i + 1}"
        old = source.inner_widths[i]
        if keep > old:
            raise ValueError("Cannot restore removed neurons")
        if keep == old:
            indices = list(range(old))
        elif method == "random":
            indices = sorted(rng.sample(range(old), keep))
        else:
            if method == "l1":
                scores = block.fc1.weight.detach().abs().sum(dim=1).cpu()
            else:
                if covariances is None or name not in covariances:
                    raise ValueError(f"Missing covariance for {name}")
                covariance = covariances[name].to(device="cpu", dtype=torch.float64)
                if covariance.shape != (old, old) or not torch.isfinite(covariance).all():
                    raise ValueError(f"Invalid covariance for {name}")
                _, vectors = torch.linalg.eigh((covariance + covariance.T) / 2)
                scores = vectors[:, -keep:].square().sum(dim=1)
            # Stable ties use the original neuron order.
            indices = scores.argsort(descending=True, stable=True)[:keep].sort().values.tolist()
        selected[name] = indices
    return selected
