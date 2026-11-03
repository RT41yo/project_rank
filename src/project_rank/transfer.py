from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from project_rank.data import build_loaders
from project_rank.models import ResNet20
from project_rank.utils import load_json, resolve_device, seed_everything


def _copy_batch_norm(source: nn.BatchNorm2d, target: nn.BatchNorm2d, indices=None) -> None:
    if indices is None:
        target.load_state_dict(source.state_dict())
        return
    for name in ("weight", "bias", "running_mean", "running_var"):
        source_value = getattr(source, name)
        target_value = getattr(target, name)
        target_value.copy_(source_value[indices])
    target.num_batches_tracked.copy_(source.num_batches_tracked)


def _validate_indices(indices: list[int], source_width: int, keep: int) -> None:
    if len(indices) != keep or len(set(indices)) != keep:
        raise ValueError("Every selected channel index must be unique")
    if indices != sorted(indices):
        raise ValueError("Selected channel indices must be sorted")
    if any(index < 0 or index >= source_width for index in indices):
        raise ValueError("Selected channel index is outside the source block")


@torch.no_grad()
def transfer_selected_channels(
    source: ResNet20,
    target: ResNet20,
    selected: dict[str, list[int]],
) -> None:
    """Copy a full model into a narrower one using explicit channel indices."""
    if source.num_classes != target.num_classes:
        raise ValueError("Source and target must have the same number of classes")
    target.stem.load_state_dict(source.stem.state_dict())
    target.fc.load_state_dict(source.fc.state_dict())

    for index, (source_block, target_block) in enumerate(
        zip(source.blocks, target.blocks, strict=True)
    ):
        name = f"block{index + 1}"
        keep = target_block.inner_channels
        if keep > source_block.inner_channels:
            raise ValueError(
                f"Target block {index + 1} is wider than source: "
                f"{keep} > {source_block.inner_channels}"
            )
        if name not in selected:
            raise ValueError(f"Missing selected channels for {name}")
        raw_indices = [int(value) for value in selected[name]]
        _validate_indices(raw_indices, source_block.inner_channels, keep)
        indices = torch.tensor(
            raw_indices,
            dtype=torch.long,
            device=source_block.conv1.weight.device,
        )

        target_block.conv1.weight.copy_(source_block.conv1.weight[indices])
        _copy_batch_norm(source_block.bn1, target_block.bn1, indices)
        target_block.conv2.weight.copy_(source_block.conv2.weight[:, indices])
        _copy_batch_norm(source_block.bn2, target_block.bn2)
        target_block.shortcut.load_state_dict(source_block.shortcut.state_dict())


@torch.no_grad()
def transfer_by_l1(source: ResNet20, target: ResNet20) -> dict[str, list[int]]:
    """Transfer channels with the largest Conv1 filter L1 norms."""
    selected: dict[str, list[int]] = {}
    for index, (source_block, target_block) in enumerate(
        zip(source.blocks, target.blocks, strict=True)
    ):
        keep = target_block.inner_channels
        importance = source_block.conv1.weight.abs().sum(dim=(1, 2, 3))
        indices = importance.topk(keep, largest=True).indices.sort().values
        selected[f"block{index + 1}"] = indices.cpu().tolist()
    transfer_selected_channels(source, target, selected)
    return selected


@torch.no_grad()
def transfer_by_random(
    source: ResNet20,
    target: ResNet20,
    seed: int,
) -> dict[str, list[int]]:
    """Transfer a reproducible random subset of channels in every block."""
    rng = random.Random(int(seed))
    selected: dict[str, list[int]] = {}
    for index, (source_block, target_block) in enumerate(
        zip(source.blocks, target.blocks, strict=True)
    ):
        keep = target_block.inner_channels
        indices = sorted(rng.sample(range(source_block.inner_channels), keep))
        selected[f"block{index + 1}"] = indices
    transfer_selected_channels(source, target, selected)
    return selected


@dataclass
class _ChannelMoments:
    channels: int
    count: int = 0
    sum_x: torch.Tensor | None = None
    sum_xtx: torch.Tensor | None = None

    def update(self, activation: torch.Tensor) -> None:
        if activation.ndim != 4 or activation.shape[1] != self.channels:
            raise ValueError("Expected activation with shape B x C x H x W")
        observations = activation.detach().to(dtype=torch.float64)
        observations = observations.permute(0, 2, 3, 1).reshape(-1, self.channels)
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

    def covariance(self) -> torch.Tensor:
        if self.count < 2 or self.sum_x is None or self.sum_xtx is None:
            raise RuntimeError("Not enough activations to calculate covariance")
        centered = self.sum_xtx - torch.outer(self.sum_x, self.sum_x) / self.count
        covariance = centered / (self.count - 1)
        return (covariance + covariance.T) / 2


@torch.inference_mode()
def collect_activation_covariances(
    model: ResNet20,
    loader: Any,
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, torch.Tensor]:
    """Collect channel covariances from the same activations used by ERank."""
    moments = {
        name: _ChannelMoments(int(model.inner_widths[index]))
        for index, name in enumerate(model.activation_modules)
    }
    handles = []
    for name, module in model.activation_modules.items():
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, layer=name: moments[layer].update(output)
            )
        )

    model.eval()
    seen = 0
    try:
        for images, _targets in loader:
            if max_samples is not None:
                remaining = max_samples - seen
                if remaining <= 0:
                    break
                images = images[:remaining]
            images = images.to(device, non_blocking=True)
            model(images)
            seen += images.shape[0]
    finally:
        for handle in handles:
            handle.remove()
    if seen == 0:
        raise ValueError("The loader did not produce any samples")
    return {name: value.covariance() for name, value in moments.items()}


def activation_subspace_selection(
    covariances: dict[str, torch.Tensor],
    target_widths: list[int] | tuple[int, ...],
) -> tuple[dict[str, list[int]], dict[str, list[float]]]:
    """Select channels that best represent the leading activation subspace.

    For a target width k, the top-k eigenvectors of the channel covariance are
    calculated. A channel receives a high score when it contributes strongly
    to that leading subspace. The k highest-scoring channels are retained.
    """
    if len(target_widths) != 9:
        raise ValueError("Expected nine target widths")
    selected: dict[str, list[int]] = {}
    scores: dict[str, list[float]] = {}
    for index, keep_value in enumerate(target_widths):
        name = f"block{index + 1}"
        if name not in covariances:
            raise ValueError(f"Missing activation covariance for {name}")
        covariance = covariances[name].to(dtype=torch.float64, device="cpu")
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError(f"Covariance for {name} must be square")
        keep = int(keep_value)
        channels = covariance.shape[0]
        if not 0 < keep <= channels:
            raise ValueError(f"Invalid target width for {name}: {keep}")
        if keep == channels:
            leverage = torch.ones(channels, dtype=torch.float64)
            indices = torch.arange(channels)
        else:
            _eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            leading_basis = eigenvectors[:, -keep:]
            leverage = leading_basis.square().sum(dim=1)
            indices = leverage.topk(keep, largest=True).indices.sort().values
        selected[name] = indices.tolist()
        scores[name] = leverage.tolist()
    return selected, scores


@torch.no_grad()
def transfer_by_activation_subspace(
    source: ResNet20,
    target: ResNet20,
    covariances: dict[str, torch.Tensor],
) -> tuple[dict[str, list[int]], dict[str, list[float]]]:
    """Transfer channels selected from the leading activation subspace."""
    selected, scores = activation_subspace_selection(
        covariances, list(target.inner_widths)
    )
    transfer_selected_channels(source, target, selected)
    return selected, scores


def _find_architecture(data: dict[str, Any], name: str) -> list[int]:
    for item in data.get("architectures", []):
        if item.get("name") == name:
            return [int(value) for value in item["inner_widths"]]
    raise KeyError(f"Architecture {name!r} was not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a transferred checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--architectures", required=True)
    parser.add_argument("--architecture", default="erank")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--method",
        choices=("l1", "random", "erank_subspace"),
        default="l1",
    )
    parser.add_argument("--config")
    parser.add_argument("--selection-seed", type=int, default=2001)
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    source_checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    device = torch.device("cpu")
    if args.method == "erank_subspace":
        if not args.config:
            parser.error("--config is required for erank_subspace")
        config = load_json(args.config)
        seed_everything(int(config.get("seed", 1)))
        device = resolve_device(str(config.get("device", "auto")))

    source = ResNet20(
        source_checkpoint["inner_widths"],
        source_checkpoint.get("num_classes", 10),
    ).to(device)
    source.load_state_dict(source_checkpoint["model_state"])

    widths = _find_architecture(load_json(args.architectures), args.architecture)
    target = ResNet20(widths, source.num_classes).to(device)
    scores = None
    if args.method == "l1":
        selected = transfer_by_l1(source, target)
        transfer_method = "conv1_l1"
    elif args.method == "random":
        selected = transfer_by_random(source, target, args.selection_seed)
        transfer_method = "random_channels"
    else:
        assert args.config is not None
        _, validation_loader = build_loaders(load_json(args.config))
        covariances = collect_activation_covariances(
            source, validation_loader, device, max_samples=args.max_samples
        )
        selected, scores = transfer_by_activation_subspace(
            source, target, covariances
        )
        transfer_method = "erank_activation_subspace"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 0,
            "model_state": target.state_dict(),
            "inner_widths": widths,
            "num_classes": source.num_classes,
            "source_checkpoint": str(args.checkpoint),
            "transfer_method": transfer_method,
            "selection_seed": (
                args.selection_seed if args.method == "random" else None
            ),
            "selected_channels": selected,
            "channel_scores": scores,
        },
        output,
    )
    print(json.dumps({"output": str(output), "inner_widths": widths}, indent=2))


if __name__ == "__main__":
    main()

