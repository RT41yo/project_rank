from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from project_rank.numerics import effective_rank_numpy


def effective_rank_from_singular_values(
    singular_values: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute effective rank along the last dimension."""
    singular_values = singular_values.clamp_min(0)
    totals = singular_values.sum(dim=-1, keepdim=True)
    probabilities = singular_values / totals.clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
    ranks = entropy.exp()
    return torch.where(totals.squeeze(-1) > eps, ranks, torch.zeros_like(ranks))


def effective_rank_torch(matrix: torch.Tensor, center: bool = True) -> torch.Tensor:
    values = matrix.to(dtype=torch.float64)
    if values.ndim < 2:
        raise ValueError("matrix must have at least two dimensions")
    if center:
        values = values - values.mean(dim=-2, keepdim=True)
    singular_values = torch.linalg.svdvals(values)
    return effective_rank_from_singular_values(singular_values)


@dataclass
class _LayerStatistics:
    channels: int
    count: int = 0
    sum_x: torch.Tensor | None = None
    sum_xtx: torch.Tensor | None = None
    image_rank_sum: float = 0.0
    image_count: int = 0

    def update(self, activation: torch.Tensor) -> None:
        if activation.ndim != 4:
            raise ValueError("Expected activation with shape B x C x H x W")
        batch, channels, height, width = activation.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")

        per_image = activation.detach().to(dtype=torch.float64)
        per_image = per_image.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        centered_per_image = per_image - per_image.mean(dim=1, keepdim=True)
        per_image_ranks = effective_rank_from_singular_values(
            torch.linalg.svdvals(centered_per_image)
        )
        self.image_rank_sum += float(per_image_ranks.sum().cpu())
        self.image_count += batch

        observations = per_image.reshape(-1, channels)
        batch_sum = observations.sum(dim=0).cpu()
        batch_xtx = (observations.T @ observations).cpu()
        if self.sum_x is None:
            self.sum_x = batch_sum
            self.sum_xtx = batch_xtx
        else:
            self.sum_x += batch_sum
            assert self.sum_xtx is not None
            self.sum_xtx += batch_xtx
        self.count += observations.shape[0]

    def finalize(self) -> dict[str, float | int]:
        if self.count == 0 or self.sum_x is None or self.sum_xtx is None:
            raise RuntimeError("No activations were collected")
        gram_centered = self.sum_xtx - torch.outer(self.sum_x, self.sum_x) / self.count
        eigenvalues = torch.linalg.eigvalsh(gram_centered).clamp_min(0)
        singular_values = eigenvalues.sqrt()
        pooled_rank = float(effective_rank_from_singular_values(singular_values))
        return {
            "channels": self.channels,
            "images": self.image_count,
            "observations": self.count,
            "mean_per_image_effective_rank": self.image_rank_sum / self.image_count,
            "pooled_effective_rank": pooled_rank,
            "pooled_rank_ratio": pooled_rank / self.channels,
        }


class ActivationRankCollector:
    """Forward-hook collector for per-image and pooled activation ERank."""

    def __init__(self, modules: Mapping[str, nn.Module], widths: Mapping[str, int]):
        if set(modules) != set(widths):
            raise ValueError("modules and widths must have identical keys")
        self.statistics = {
            name: _LayerStatistics(channels=int(widths[name])) for name in modules
        }
        self.handles = [
            module.register_forward_hook(self._make_hook(name))
            for name, module in modules.items()
        ]

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            self.statistics[name].update(output)

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def results(self) -> dict[str, dict[str, float | int]]:
        return {name: stats.finalize() for name, stats in self.statistics.items()}

    def __enter__(self) -> "ActivationRankCollector":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


@torch.inference_mode()
def collect_activation_ranks(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, dict[str, float | int]]:
    if not hasattr(model, "activation_modules") or not hasattr(model, "inner_widths"):
        raise TypeError("Model must expose activation_modules and inner_widths")
    modules = model.activation_modules
    widths = {
        f"block{index + 1}": width
        for index, width in enumerate(model.inner_widths)
    }
    model.eval()
    seen = 0
    with ActivationRankCollector(modules, widths) as collector:
        for images, _targets in loader:
            if max_samples is not None:
                remaining = max_samples - seen
                if remaining <= 0:
                    break
                images = images[:remaining]
            images = images.to(device, non_blocking=True)
            model(images)
            seen += images.shape[0]
    return collector.results()


def collect_weight_ranks(model: nn.Module) -> dict[str, float]:
    if not hasattr(model, "blocks"):
        raise TypeError("Model must expose residual blocks")
    results: dict[str, float] = {}
    for index, block in enumerate(model.blocks):
        weight = block.conv1.weight.detach().to(dtype=torch.float64)
        matrix = weight.reshape(weight.shape[0], -1)
        rank = effective_rank_torch(matrix, center=False)
        results[f"block{index + 1}"] = float(rank.cpu())
    return results

