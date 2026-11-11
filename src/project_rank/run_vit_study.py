"""Sequential validation-only runner for a frozen stage-D manifest (Linux)."""
from __future__ import annotations

import argparse
import fcntl
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from project_rank.evaluate import evaluate_checkpoint
from project_rank.prepare_vit_p060 import config_sha256, file_sha256
from project_rank.utils import load_json, save_json


def run_complete(config: dict[str, Any]) -> bool:
    """Only untouched or fully validated runs are resumable; never overwrite partials."""
    output = Path(config["output_dir"])
    names = ("config.json", "history.json", "best.pt", "last.pt")
    files = [output / name for name in names]
    if not any(p.exists() for p in files) and not list(output.glob("*.pt")):
        return False
    if not all(p.exists() for p in files):
        raise RuntimeError(f"Partial run in {output}; no automatic overwrite/restart")
    if load_json(files[0]) != config:
        raise ValueError(f"Result/config mismatch in {output}")
    history = load_json(files[1])
    epochs = int(config["epochs"])
    if [r["epoch"] for r in history] != list(range(1, epochs + 1)):
        raise RuntimeError(f"Incomplete epoch history in {output}")
    for row in history:
        for split in ("train", "validation"):
            if not math.isfinite(row[split]["loss"]) or not 0 <= row[split]["accuracy"] <= 1:
                raise ValueError(f"Invalid metrics in {output}")
            if config["dataset"] == "cifar10":
                expected = 50_000 - config.get("val_size", 5000) if split == "train" else config.get("val_size", 5000)
            else:
                expected = config.get("synthetic_train_size", 64) if split == "train" else config.get("synthetic_val_size", 32)
            if row[split]["samples"] != expected:
                raise ValueError(f"Unexpected {split} sample count in {output}")
    # train.py saves the FIRST maximum validation accuracy.
    best_row = max(history, key=lambda row: row["validation"]["accuracy"])
    for path, epoch in ((files[2], best_row["epoch"]), (files[3], epochs)):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (checkpoint["epoch"] != epoch or checkpoint["config"] != config
                or checkpoint["inner_widths"] != config["inner_widths"]
                or checkpoint["model_spec"] != config["model_spec"]
                or checkpoint["validation_accuracy"] != history[epoch - 1]["validation"]["accuracy"]):
            raise ValueError(f"Checkpoint/history mismatch: {path}")
    return True


def run_study(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    # A second runner cannot race the first even before the first epoch is saved.
    with manifest_path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This study already has an active runner") from exc
        return _run_locked(manifest_path)


def _run_locked(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "vit_stage_d_v1":
        raise ValueError("Unsupported study manifest")
    if file_sha256(manifest["source_checkpoint"]) != manifest["source_sha256"]:
        raise ValueError("Baseline checkpoint changed after preparation")
    experiments = manifest["experiments"]
    roots = {Path(e["result_dir"]).parent for e in experiments}
    if len(roots) != 1:
        raise ValueError("Stage-D runs must share one result root")
    result_root = next(iter(roots))
    if len({e["name"] for e in experiments}) != len(experiments):
        raise ValueError("Duplicate run names")
    configs = {}
    completed = {}
    # Preflight ALL runs before starting expensive work.
    for experiment in experiments:
        config = load_json(experiment["config"])
        if config_sha256(config) != experiment["config_sha256"]:
            raise ValueError(f"Frozen configuration changed: {experiment['config']}")
        if (config["output_dir"] != experiment["result_dir"]
                or config["inner_widths"] != experiment["inner_widths"]
                or config["epochs"] != experiment["epochs"]):
            raise ValueError("Manifest and configuration differ")
        if experiment["initial_checkpoint"]:
            if config.get("resume_from") != experiment["initial_checkpoint"]:
                raise ValueError("Fine-tune initialization path mismatch")
            if file_sha256(experiment["initial_checkpoint"]) != experiment["initial_sha256"]:
                raise ValueError("Fine-tune initialization changed after preparation")
        configs[experiment["name"]] = config
        completed[experiment["name"]] = run_complete(config)
    status_path = result_root / "run_status.json"
    status = {"manifest": str(manifest_path), "test_evaluated": False, "runs": {}}

    def record(name: str, state: str, **extra: Any) -> None:
        status["runs"][name] = {"state": state, **extra}
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_json(status, status_path)

    # Record the fixed masks' initial accuracy BEFORE any fine-tune.
    for experiment in experiments:
        name, config = experiment["name"], configs[experiment["name"]]
        record(name, "complete" if completed[name] else "pending")
        if not experiment["initial_checkpoint"]:
            continue
        initial_path = Path(experiment["initial_validation"])
        if initial_path.exists():
            initial = load_json(initial_path)
            if (initial["initial_sha256"] != experiment["initial_sha256"]
                    or initial["config_sha256"] != experiment["config_sha256"]
                    or initial["metrics"]["split"] != "validation"):
                raise ValueError("Existing initial evaluation has different provenance")
        else:
            print(f"Initial validation: {name}", flush=True)
            metrics = evaluate_checkpoint(config, experiment["initial_checkpoint"], split="validation")
            save_json({"initial_sha256": experiment["initial_sha256"],
                       "config_sha256": experiment["config_sha256"], "metrics": metrics}, initial_path)
    for experiment in experiments:
        name, config = experiment["name"], configs[experiment["name"]]
        if completed[name]:
            print(f"Verified complete; skipping {name}", flush=True)
            continue
        output = Path(config["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "train.log"
        record(name, "training", log=str(log_path))
        print(f"Training {name}; log: {log_path}", flush=True)
        try:
            with log_path.open("x") as log:
                subprocess.run([sys.executable, "-u", "-m", "project_rank.train",
                                "--config", experiment["config"]], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            if not run_complete(config):
                raise RuntimeError(f"Training returned without complete artifacts: {name}")
        except Exception as exc:
            record(name, "failed", error=str(exc), log=str(log_path))
            raise
        record(name, "complete", log=str(log_path))
    summary = {
        "schema": "vit_stage_d_validation_summary_v1", "manifest": str(manifest_path),
        "protocol_id": manifest["protocol_id"], "train_seed": manifest["train_seed"],
        "source_checkpoint": manifest["source_checkpoint"], "source_sha256": manifest["source_sha256"],
        "baseline_validation_accuracy": manifest["source_validation_accuracy"],
        "test_evaluated": False, "experiments": [],
    }
    for experiment in experiments:
        history = load_json(Path(experiment["result_dir"]) / "history.json")
        best = max(history, key=lambda row: row["validation"]["accuracy"])
        initial = load_json(experiment["initial_validation"])["metrics"] if experiment["initial_validation"] else None
        summary["experiments"].append({
            **experiment, "epochs_completed": len(history), "best_epoch": best["epoch"],
            "validation_accuracy": best["validation"]["accuracy"],
            "validation_loss": best["validation"]["loss"], "initial_validation": initial,
            "training_seconds": sum(row["train"]["seconds"] for row in history),
        })
    save_json(summary, result_root / "validation_summary.json")
    print(f"Completed {len(experiments)} runs; summary: {result_root / 'validation_summary.json'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen ViT stage-D study sequentially; validation only")
    parser.add_argument("--manifest", default="configs/vit/p060_stage_d_seed1/manifest.json")
    args = parser.parse_args()
    run_study(args.manifest)


if __name__ == "__main__":
    main()
