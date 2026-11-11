from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from project_rank.data import build_loaders
from project_rank.models.factory import build_model, model_specification
from project_rank.models.vit_tiny import ViTTinyCIFAR
from project_rank.optimization import build_optimizer, build_scheduler
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

    model = build_model(config).to(device)
    output_dir = Path(str(config.get("output_dir", "results/run")))
    if isinstance(model, ViTTinyCIFAR) and not config.get("overwrite", False):
        if any((output_dir / name).exists() for name in ("best.pt", "last.pt", "history.json")):
            raise FileExistsError(f"Refusing to overwrite ViT run: {output_dir}. Use a new output_dir.")

    resume_from = config.get("resume_from")
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        if checkpoint.get("model_spec", {"type": "resnet20"}) != model_specification(model):
            raise ValueError("Initialization checkpoint model_spec differs from target")
        model.load_state_dict(checkpoint["model_state"])

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    epochs = int(config.get("epochs", 160))
    rank_epochs = config.get("rank_checkpoint_epochs", [])
    if epochs < 1 or any(type(e) is not int or not 0 <= e <= epochs for e in rank_epochs):
        raise ValueError("epochs must be positive; rank checkpoint epochs must be in [0, epochs]")
    scheduler = build_scheduler(optimizer, config, epochs)

    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "config.json")
    save_json(environment_summary(), output_dir / "environment.json")

    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    if 0 in rank_epochs:
        torch.save({
            "epoch": 0, "model_state": model.state_dict(),
            "inner_widths": list(model.inner_widths), "num_classes": model.num_classes,
            "model_spec": model_specification(model), "config": config,
        }, output_dir / "epoch_000.pt")
    for epoch in range(1, epochs + 1):
        learning_rate_used = optimizer.param_groups[0]["lr"]
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
        if not all(math.isfinite(m["loss"]) for m in (train_metrics, val_metrics)):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}; run is incomplete")
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "learning_rate_used": learning_rate_used,
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
            "model_spec": model_specification(model),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if epoch in rank_epochs:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
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
    parser = argparse.ArgumentParser(description="Train CIFAR ResNet-20 or ViT")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_from_config(load_json(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
