from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import torch

from project_rank.architectures import make_erank_architecture_for_budget
from project_rank.data import build_loaders
from project_rank.effective_rank import collect_activation_ranks, collect_weight_ranks
from project_rank.models import DEFAULT_INNER_WIDTHS, ResNet20
from project_rank.spec import count_resnet20_parameters
from project_rank.transfer import transfer_by_l1
from project_rank.utils import load_json, resolve_device, save_json, seed_everything


def _save_shell(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    path.chmod(0o755)


def _rank_rows(
    activation_results: dict[str, dict[str, float | int]],
    weight_results: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(1, 10):
        name = f"block{index}"
        if name not in activation_results or name not in weight_results:
            raise ValueError(f"Missing rank results for {name}")
        rows.append(
            {
                "block": name,
                **activation_results[name],
                "weight_effective_rank": float(weight_results[name]),
            }
        )
    return rows


def prepare_iterative_step(
    base_config: dict[str, Any],
    source_checkpoint: dict[str, Any],
    activation_results: dict[str, dict[str, float | int]],
    weight_results: dict[str, float],
    source_checkpoint_path: str,
    config_dir: str | Path,
    output_dir: str | Path,
    target_param_ratio: float,
    fine_tune_epochs: int = 30,
    fine_tune_lr: float = 0.01,
    max_samples: int = 5000,
) -> dict[str, Any]:
    """Prepare one ERank iteration and both required training branches."""
    if not 0 < target_param_ratio < 1:
        raise ValueError("target_param_ratio must be in (0, 1)")

    source_widths = tuple(int(value) for value in source_checkpoint["inner_widths"])
    if len(source_widths) != 9:
        raise ValueError("Source checkpoint must contain nine inner widths")

    full_parameters = count_resnet20_parameters(DEFAULT_INNER_WIDTHS)
    source_parameters = count_resnet20_parameters(source_widths)
    target_parameters = int(round(full_parameters * target_param_ratio))
    if target_parameters >= source_parameters:
        raise ValueError(
            "The next iterative budget must be smaller than the source model"
        )

    rows = _rank_rows(activation_results, weight_results)
    ranks = [float(row["mean_per_image_effective_rank"]) for row in rows]
    architecture = make_erank_architecture_for_budget(
        ranks,
        target_parameters,
        base_widths=source_widths,
        minimum=4,
        multiple=2,
    )
    target_widths = list(architecture.inner_widths)
    if any(target > source for target, source in zip(target_widths, source_widths)):
        raise RuntimeError("An iterative step cannot restore removed channels")

    config_path = Path(config_dir)
    result_path = Path(output_dir)
    config_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)

    ranks_path = result_path / "ranks.csv"
    with ranks_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    source_model = ResNet20(
        source_widths,
        int(source_checkpoint.get("num_classes", 10)),
    )
    source_model.load_state_dict(source_checkpoint["model_state"])
    target_model = ResNet20(target_widths, source_model.num_classes)
    selected_channels = transfer_by_l1(source_model, target_model)

    initialization_path = result_path / "l1_initialization.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state": target_model.state_dict(),
            "inner_widths": target_widths,
            "num_classes": source_model.num_classes,
            "source_checkpoint": source_checkpoint_path,
            "transfer_method": "conv1_l1",
            "selected_channels": selected_channels,
        },
        initialization_path,
    )

    label = f"p{int(round(target_param_ratio * 100)):03d}"
    scratch_result = result_path / "scratch"
    fine_tune_result = result_path / "finetune_l1"

    scratch_config = copy.deepcopy(base_config)
    scratch_config.pop("resume_from", None)
    scratch_config["inner_widths"] = target_widths
    scratch_config["output_dir"] = str(scratch_result)
    scratch_config["iterative_search"] = {
        "criterion": "erank_per_image",
        "source_checkpoint": source_checkpoint_path,
        "target_budget": label,
        "training_mode": "scratch",
    }

    fine_tune_config = copy.deepcopy(base_config)
    fine_tune_config["inner_widths"] = target_widths
    fine_tune_config["resume_from"] = str(initialization_path)
    fine_tune_config["epochs"] = int(fine_tune_epochs)
    fine_tune_config["learning_rate"] = float(fine_tune_lr)
    fine_tune_config["output_dir"] = str(fine_tune_result)
    fine_tune_config["iterative_search"] = {
        "criterion": "erank_per_image",
        "source_checkpoint": source_checkpoint_path,
        "target_budget": label,
        "training_mode": "finetune",
        "channel_transfer": "l1",
    }

    scratch_config_path = config_path / f"{label}_scratch.json"
    fine_tune_config_path = config_path / f"{label}_finetune_l1.json"
    save_json(scratch_config, scratch_config_path)
    save_json(fine_tune_config, fine_tune_config_path)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "criterion": "erank_per_image",
        "rank_samples": int(max_samples),
        "source": {
            "checkpoint": source_checkpoint_path,
            "inner_widths": list(source_widths),
            "parameters": source_parameters,
            "parameter_ratio_vs_full": source_parameters / full_parameters,
        },
        "target": {
            "label": label,
            "requested_parameter_ratio_vs_full": float(target_param_ratio),
            "requested_parameters": target_parameters,
            "inner_widths": target_widths,
            "parameters": architecture.parameters,
            "parameter_ratio_vs_full": architecture.parameters / full_parameters,
        },
        "ranks": str(ranks_path),
        "fine_tune": {
            "channel_transfer": "l1",
            "initialization": str(initialization_path),
            "config": str(fine_tune_config_path),
            "output_dir": str(fine_tune_result),
        },
        "scratch": {
            "config": str(scratch_config_path),
            "output_dir": str(scratch_result),
        },
    }
    save_json(manifest, result_path / "manifest.json")

    _save_shell(
        config_path / "train_commands.sh",
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"python -m project_rank.train --config {fine_tune_config_path}",
            f"python -m project_rank.train --config {scratch_config_path}",
        ],
    )
    _save_shell(
        config_path / "test_commands.sh",
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "python -m project_rank.evaluate \\",
            f"  --config {fine_tune_config_path} \\",
            f"  --checkpoint {fine_tune_result / 'best.pt'} \\",
            "  --split test \\",
            f"  --output {fine_tune_result / 'test.json'}",
            "",
            "python -m project_rank.evaluate \\",
            f"  --config {scratch_config_path} \\",
            f"  --checkpoint {scratch_result / 'best.pt'} \\",
            "  --split test \\",
            f"  --output {scratch_result / 'test.json'}",
        ],
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare one iterative Effective-Rank architecture-search step"
    )
    parser.add_argument("--base-config", default="configs/baseline.json")
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--target-param-ratio", type=float, required=True)
    parser.add_argument(
        "--config-dir",
        default="configs/iterative_erank_seed1/step_p080_to_p060",
    )
    parser.add_argument(
        "--output-dir",
        default="results/iterative_erank_seed1/step_p080_to_p060",
    )
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--fine-tune-epochs", type=int, default=30)
    parser.add_argument("--fine-tune-lr", type=float, default=0.01)
    args = parser.parse_args()

    config = load_json(args.base_config)
    seed_everything(int(config.get("seed", 1)))
    device = resolve_device(str(config.get("device", "auto")))
    _, validation_loader = build_loaders(config)
    checkpoint = torch.load(
        args.source_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = ResNet20(
        checkpoint["inner_widths"],
        checkpoint.get("num_classes", 10),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    activation_results = collect_activation_ranks(
        model,
        validation_loader,
        device,
        max_samples=args.max_samples,
    )
    weight_results = collect_weight_ranks(model)
    manifest = prepare_iterative_step(
        base_config=config,
        source_checkpoint=checkpoint,
        activation_results=activation_results,
        weight_results=weight_results,
        source_checkpoint_path=args.source_checkpoint,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        target_param_ratio=args.target_param_ratio,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_lr=args.fine_tune_lr,
        max_samples=args.max_samples,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
