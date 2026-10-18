from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from project_rank.data import build_loaders
from project_rank.models import DEFAULT_INNER_WIDTHS, ResNet20
from project_rank.utils import load_json, resolve_device, save_json, seed_everything


def l2_normalize(values: Sequence[float]) -> list[float]:
    """Normalize one layer's channel scores for global comparison."""
    scores = [float(value) for value in values]
    if not scores or any(value < 0 or not math.isfinite(value) for value in scores):
        raise ValueError("Scores must be a non-empty list of finite non-negative values")
    norm = math.sqrt(sum(value * value for value in scores))
    if norm == 0.0:
        return [1.0 / math.sqrt(len(scores))] * len(scores)
    return [value / norm for value in scores]


@torch.no_grad()
def collect_l1_weight_scores(model: ResNet20) -> dict[str, list[float]]:
    """Return the L1 norm of every Conv1 output filter in every block."""
    result: dict[str, list[float]] = {}
    for index, block in enumerate(model.blocks, start=1):
        scores = block.conv1.weight.detach().abs().sum(dim=(1, 2, 3))
        result[f"block{index}"] = scores.cpu().double().tolist()
    return result


def collect_taylor_scores(
    model: ResNet20,
    loader: Any,
    device: torch.device,
    max_samples: int,
) -> tuple[dict[str, list[float]], int]:
    """Estimate channel importance with first-order Taylor scores.

    For every image and channel, the spatial sum of activation times loss
    gradient is calculated. Its absolute value is averaged over images.
    No optimizer step is performed, so this function does not train the model.
    """
    if max_samples < 1:
        raise ValueError("max_samples must be positive")

    model.eval()
    totals = {
        f"block{index + 1}": torch.zeros(block.inner_channels, dtype=torch.float64)
        for index, block in enumerate(model.blocks)
    }
    handles: list[Any] = []

    def make_forward_hook(name: str):
        def forward_hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            activation = output.detach()

            def gradient_hook(gradient: torch.Tensor) -> None:
                contribution = (activation * gradient.detach()).sum(dim=(2, 3)).abs()
                totals[name].add_(contribution.sum(dim=0).cpu().double())

            output.register_hook(gradient_hook)

        return forward_hook

    for name, module in model.activation_modules.items():
        handles.append(module.register_forward_hook(make_forward_hook(name)))

    criterion = nn.CrossEntropyLoss(reduction="sum")
    processed = 0
    try:
        for images, targets in loader:
            remaining = max_samples - processed
            if remaining <= 0:
                break
            if images.shape[0] > remaining:
                images = images[:remaining]
                targets = targets[:remaining]
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            processed += int(targets.shape[0])
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)

    if processed == 0:
        raise ValueError("The loader did not produce any samples")
    return {
        name: (values / processed).tolist() for name, values in totals.items()
    }, processed


def _criterion_payload(scores: dict[str, list[float]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, raw_scores in scores.items():
        normalized = l2_normalize(raw_scores)
        payload[name] = {
            "channels": len(raw_scores),
            "raw_scores": raw_scores,
            "normalized_scores": normalized,
            "raw_l2_norm": math.sqrt(sum(value * value for value in raw_scores)),
        }
    return payload


def analyze_pruning_criteria(
    model: ResNet20,
    loader: Any,
    device: torch.device,
    max_samples: int,
) -> dict[str, Any]:
    l1_scores = collect_l1_weight_scores(model)
    taylor_scores, processed = collect_taylor_scores(
        model, loader, device, max_samples=max_samples
    )
    return {
        "schema_version": 1,
        "samples": processed,
        "normalization": "layer_l2",
        "criteria": {
            "l1_weight": _criterion_payload(l1_scores),
            "taylor": _criterion_payload(taylor_scores),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate L1-weight and first-order Taylor channel scores"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    config = load_json(args.config)
    seed_everything(int(config.get("seed", 1)))
    device = resolve_device(str(config.get("device", "auto")))
    _, validation_loader = build_loaders(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    widths = tuple(int(value) for value in checkpoint["inner_widths"])
    if widths != tuple(DEFAULT_INNER_WIDTHS):
        raise ValueError("Pruning criteria must be measured on the full ResNet-20")
    model = ResNet20(widths, checkpoint.get("num_classes", 10)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    result = analyze_pruning_criteria(
        model, validation_loader, device, max_samples=args.max_samples
    )
    result["checkpoint"] = str(args.checkpoint)
    result["max_samples"] = int(args.max_samples)
    save_json(result, Path(args.output))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": result["samples"],
                "criteria": list(result["criteria"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
