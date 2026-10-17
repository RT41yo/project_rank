from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from project_rank.data import build_loaders
from project_rank.models import DEFAULT_INNER_WIDTHS, ResNet20
from project_rank.utils import (
    environment_summary,
    load_json,
    resolve_device,
    save_json,
    seed_everything,
)


def _run_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    count = 0
    started = time.perf_counter()

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            batch_size = targets.shape[0]
            loss_sum += float(loss.detach()) * batch_size
            correct += int((logits.argmax(dim=1) == targets).sum())
            count += batch_size

    return {
        "loss": loss_sum / max(count, 1),
        "accuracy": correct / max(count, 1),
        "samples": float(count),
        "seconds": time.perf_counter() - started,
    }


def train_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(config)
    seed = int(config.get("seed", 1))
    seed_everything(seed, deterministic=bool(config.get("deterministic", False)))
    device = resolve_device(str(config.get("device", "auto")))
    train_loader, val_loader = build_loaders(config)

    model = ResNet20(
        inner_widths=config.get("inner_widths", DEFAULT_INNER_WIDTHS),
        num_classes=int(config.get("num_classes", 10)),
    ).to(device)

    resume_from = config.get("resume_from")
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.1)),
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=float(config.get("weight_decay", 5e-4)),
        nesterov=bool(config.get("nesterov", True)),
    )
    epochs = int(config.get("epochs", 160))
    scheduler_name = str(config.get("scheduler", "cosine"))
    if scheduler_name == "cosine":
        scheduler: Any = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    output_dir = Path(str(config.get("output_dir", "results/run")))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "config.json")
    save_json(environment_summary(), output_dir / "environment.json")

    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            amp_enabled,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
        )
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "inner_widths": list(model.inner_widths),
            "num_classes": model.num_classes,
            "validation_accuracy": val_metrics["accuracy"],
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            torch.save(checkpoint, output_dir / "best.pt")
        save_json(history, output_dir / "history.json")

    return {
        "best_validation_accuracy": best_accuracy,
        "checkpoint": str(output_dir / "best.pt"),
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR ResNet-20")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_from_config(load_json(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

