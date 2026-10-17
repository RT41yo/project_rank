from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from project_rank.data import build_loaders, build_test_loader
from project_rank.models import ResNet20
from project_rank.train import _run_epoch
from project_rank.utils import load_json, resolve_device, save_json


def evaluate_checkpoint(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    split: str = "validation",
) -> dict[str, Any]:
    """Evaluate one checkpoint without changing model weights."""
    config = dict(config)
    device = resolve_device(str(config.get("device", "auto")))
    if split == "validation":
        _, loader = build_loaders(config)
    elif split == "test":
        loader = build_test_loader(config)
    else:
        raise ValueError("split must be 'validation' or 'test'")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ResNet20(
        checkpoint["inner_widths"], checkpoint.get("num_classes", 10)
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = _run_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device,
        optimizer=None,
        scaler=None,
        amp_enabled=bool(config.get("amp", True)) and device.type == "cuda",
    )
    return {"split": split, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved ResNet-20")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path; metrics are always printed as well",
    )
    args = parser.parse_args()

    config = load_json(args.config)
    metrics = evaluate_checkpoint(
        config,
        args.checkpoint,
        split=args.split,
    )
    if args.output:
        save_json(metrics, args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

