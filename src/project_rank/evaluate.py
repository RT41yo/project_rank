from __future__ import annotations

import argparse
import json

import torch
from torch import nn

from project_rank.data import build_loaders
from project_rank.models import ResNet20
from project_rank.train import _run_epoch
from project_rank.utils import load_json, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved ResNet-20")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    device = resolve_device(str(config.get("device", "auto")))
    _, val_loader = build_loaders(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ResNet20(
        checkpoint["inner_widths"], checkpoint.get("num_classes", 10)
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = _run_epoch(
        model,
        val_loader,
        nn.CrossEntropyLoss(),
        device,
        optimizer=None,
        scaler=None,
        amp_enabled=bool(config.get("amp", True)) and device.type == "cuda",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

