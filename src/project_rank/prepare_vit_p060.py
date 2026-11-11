"""Freeze the five-run stage-D design and create reproducible initial weights."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from project_rank.data import build_loaders
from project_rank.models.factory import model_from_checkpoint
from project_rank.models.vit_tiny import ViTTinyCIFAR
from project_rank.transfer import collect_activation_covariances
from project_rank.utils import load_json, resolve_device, save_json, seed_everything
from project_rank.vit_spec import count_vit_macs, count_vit_parameters
from project_rank.vit_transfer import select_vit_neurons, transfer_vit_neurons


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def prepare_study(
    base_config: dict[str, Any], architectures: dict[str, Any],
    checkpoint_path: str | Path, covariances: dict[str, torch.Tensor],
    config_dir: str | Path, result_dir: str | Path, *,
    train_seed: int = 1, architecture_seed: int = 1001, mask_seed: int = 2001,
    fine_tune_epochs: int = 30, fine_tune_lr_ratio: float = .1,
    fine_tune_warmup_epochs: int = 5, calibration_samples: int = 5000,
) -> dict[str, Any]:
    config_dir, result_dir = Path(config_dir), Path(result_dir)
    if config_dir.exists() and any(config_dir.iterdir()):
        raise FileExistsError(f"Config directory is not empty: {config_dir}; reuse its manifest instead")
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"Result directory is not empty: {result_dir}; do not overwrite an existing study")
    if base_config.get("optimizer") != "adamw":
        raise ValueError("This study requires the agreed AdamW recipe")
    if int(base_config.get("seed", 1)) != train_seed:
        raise ValueError("Stage D must use the baseline training/data seed")
    if train_seed != 1:
        raise ValueError("This first stage is frozen at train_seed=1")
    if not 0 < fine_tune_lr_ratio <= 1 or fine_tune_epochs < 1:
        raise ValueError("Invalid fine-tune duration or LR ratio")
    if not 0 <= fine_tune_warmup_epochs < fine_tune_epochs or calibration_samples < 1:
        raise ValueError("Invalid warmup or calibration sample count")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = model_from_checkpoint(checkpoint)
    if not isinstance(source, ViTTinyCIFAR) or source.inner_widths != source.spec.default_widths:
        raise ValueError("Expected a full ViT source checkpoint")
    source_config = checkpoint.get("config", {})
    if source_config != base_config:
        raise ValueError("Source checkpoint config does not match the base recipe")
    if architectures.get("model_spec") != source.spec.to_dict():
        raise ValueError("Architecture model specification differs from checkpoint")
    metadata = architectures.get("analysis_metadata") or {}
    if Path(metadata.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("Architecture ranks must originate from this checkpoint")
    if metadata.get("samples") != calibration_samples:
        raise ValueError("Calibration sample count differs from the architecture source")
    budget = next(b for b in architectures["budgets"] if b["label"] == "p060")
    erank = next(a for a in budget["architectures"] if a["name"] == "erank_per_image")
    random = next(a for a in budget["architectures"]
                  if a["name"] == "random" and a["architecture_seed"] == architecture_seed)
    spec = source.spec
    if architectures["full_parameters"] != count_vit_parameters(spec=spec):
        raise ValueError("Full parameter count mismatch")
    for architecture in (erank, random):
        widths = spec.widths(architecture["inner_widths"])
        if any(w < 4 or w % 2 or w > old for w, old in zip(widths, source.inner_widths, strict=True)):
            raise ValueError("Widths violate the pruning grid")
        if count_vit_parameters(widths, spec) != architecture["parameters"]:
            raise ValueError("Architecture parameter count mismatch")
    if erank["parameters"] != random["parameters"]:
        raise ValueError("Stage D requires EXACT parameter matching")
    if erank["inner_widths"] == random["inner_widths"]:
        raise ValueError("Coincident scratch profiles need an alias, not two physical runs")
    source_hash = file_sha256(checkpoint_path)
    configs, initializations, experiments = {}, {}, []
    selections = {}

    def add_run(name: str, widths: list[int], mode: str, method: str, init_path: Path | None = None) -> None:
        config = copy.deepcopy(base_config)
        config.pop("resume_from", None)
        config.pop("overwrite", None)
        config.update(protocol_id="vit_cifar10_adamw_v1", seed=train_seed,
                      inner_widths=list(widths), output_dir=str(result_dir / name))
        experiment = {
            "family": "vit_p060_stage_d", "budget": "p060", "mode": mode,
            "width_rule": method if mode == "scratch" else "erank_per_image",
            "transfer_method": method if mode == "finetune" else None,
            "architecture_seed": architecture_seed if method == "random" and mode == "scratch" else None,
            "mask_seed": mask_seed if method == "random" and mode == "finetune" else None,
        }
        config["experiment"] = experiment
        if init_path is not None:
            config.update(resume_from=str(init_path), epochs=fine_tune_epochs,
                          learning_rate=float(base_config["learning_rate"]) * fine_tune_lr_ratio,
                          warmup_epochs=fine_tune_warmup_epochs, rank_checkpoint_epochs=[])
        config_path = config_dir / f"{name}.json"
        configs[config_path] = config
        experiments.append({
            "name": name, **experiment, "config": str(config_path),
            "config_sha256": config_sha256(config), "result_dir": config["output_dir"],
            "epochs": config["epochs"], "inner_widths": list(widths),
            "parameters": count_vit_parameters(widths, spec), "macs_per_image": count_vit_macs(widths, spec),
            "initial_checkpoint": str(init_path) if init_path else None,
            "initial_validation": str(result_dir / f"{name}_initial_validation.json") if init_path else None,
        })

    add_run("p060_erank_per_image_scratch_seed1", erank["inner_widths"], "scratch", "erank_per_image")
    add_run(f"p060_random_arch{architecture_seed}_scratch_seed1", random["inner_widths"], "scratch", "random")
    for method in ("l1", "random", "erank_subspace"):
        widths = erank["inner_widths"]
        selected = select_vit_neurons(source, widths, method, mask_seed, covariances)
        target = ViTTinyCIFAR(widths, spec)
        transfer_vit_neurons(source, target, selected)
        init_path = result_dir / "initializations" / f"p060_{method}.pt"
        initializations[init_path] = {
            "epoch": 0, "model_state": target.state_dict(), "model_spec": spec.to_dict(),
            "inner_widths": widths, "num_classes": spec.num_classes,
            "source_checkpoint": str(checkpoint_path), "source_sha256": source_hash,
            "transfer_method": method, "mask_seed": mask_seed if method == "random" else None,
            "selected_neurons": selected,
        }
        selections[method] = selected
        add_run(f"p060_{method}_finetune_seed1", widths, "finetune", method, init_path)

    config_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "initializations").mkdir(parents=True, exist_ok=True)
    for path, config in configs.items():
        save_json(config, path)
    for path, initialization in initializations.items():
        torch.save(initialization, path)
    for experiment in experiments:
        if experiment["initial_checkpoint"]:
            experiment["initial_sha256"] = file_sha256(experiment["initial_checkpoint"])
    manifest = {
        "schema": "vit_stage_d_v1", "protocol_id": "vit_cifar10_adamw_v1",
        "source_protocol_id": base_config.get("protocol_id"),
        "source_checkpoint": str(checkpoint_path), "source_sha256": source_hash,
        "source_validation_accuracy": checkpoint["validation_accuracy"],
        "model_spec": spec.to_dict(), "train_seed": train_seed,
        "calibration_samples": calibration_samples, "calibration_excludes_cls": True,
        "fine_tune_lr_ratio": fine_tune_lr_ratio, "fine_tune_epochs": fine_tune_epochs,
        "fine_tune_warmup_epochs": fine_tune_warmup_epochs,
        "test_policy": "validation_only_until_experiment_design_frozen",
        "experiments": experiments,
    }
    save_json(selections, result_dir / "selected_neurons.json")
    save_json(manifest, config_dir / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the five-run ViT p060 stage; no training")
    parser.add_argument("--base-config", default="configs/vit/baseline_adamw_pilot.json")
    parser.add_argument("--architectures", default="reports/data/vit/architectures_p060_preview.json")
    parser.add_argument("--checkpoint", default="results/vit/baseline_adamw_pilot_seed1/best.pt")
    parser.add_argument("--config-dir", default="configs/vit/p060_stage_d_seed1")
    parser.add_argument("--result-dir", default="results/vit/p060_stage_d_seed1")
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()
    # Refuse reuse before doing the expensive covariance pass.
    for directory in (Path(args.config_dir), Path(args.result_dir)):
        if directory.exists() and any(directory.iterdir()):
            parser.error(f"Nonempty destination: {directory}; reuse its frozen manifest")
    config = load_json(args.base_config)
    seed_everything(int(config.get("seed", 1)))
    device = resolve_device(config["device"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint).to(device)
    _, loader = build_loaders(config)
    if len(loader.dataset) != args.max_samples:
        parser.error("Stage D calibration must use the entire validation split")
    print(f"Collecting covariance on {args.max_samples} validation images (no optimizer steps)", flush=True)
    covariances = collect_activation_covariances(model, loader, device, args.max_samples)
    del model, loader
    manifest = prepare_study(config, load_json(args.architectures), args.checkpoint, covariances,
                             args.config_dir, args.result_dir, calibration_samples=args.max_samples)
    print(f"Prepared {len(manifest['experiments'])} runs: {args.config_dir}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
