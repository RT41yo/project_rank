from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from project_rank.models import DEFAULT_INNER_WIDTHS, ResNet20
from project_rank.transfer import transfer_by_l1
from project_rank.utils import load_json, save_json


def _group_by_widths(
    architectures: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], list[str]]]:
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for item in architectures:
        widths = tuple(int(value) for value in item["inner_widths"])
        groups.setdefault(widths, []).append(item)
    return [(items[0], [str(item["name"]) for item in items]) for items in groups.values()]


def prepare_experiments(
    base_config: dict[str, Any],
    architecture_report: dict[str, Any],
    baseline_checkpoint: dict[str, Any],
    output_dir: str | Path,
    initializations_dir: str | Path,
    fine_tune_epochs: int,
    fine_tune_lr: float,
) -> dict[str, Any]:
    if architecture_report.get("schema_version") != 2:
        raise ValueError("Expected architecture report schema_version=2")
    if tuple(baseline_checkpoint["inner_widths"]) != tuple(DEFAULT_INNER_WIDTHS):
        raise ValueError("Baseline checkpoint must contain the full ResNet-20")

    config_dir = Path(output_dir)
    init_dir = Path(initializations_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    init_dir.mkdir(parents=True, exist_ok=True)

    source = ResNet20(
        baseline_checkpoint["inner_widths"],
        baseline_checkpoint.get("num_classes", 10),
    )
    source.load_state_dict(baseline_checkpoint["model_state"])
    seed = int(base_config.get("seed", 1))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "budgets": [],
    }

    for budget in architecture_report["budgets"]:
        label = str(budget["label"])
        budget_manifest: dict[str, Any] = {
            "label": label,
            "requested_parameter_ratio": budget["requested_parameter_ratio"],
            "experiments": [],
            "aliases": [],
        }
        for representative, aliases in _group_by_widths(budget["architectures"]):
            name = str(representative["name"])
            widths = [int(value) for value in representative["inner_widths"]]
            if len(aliases) > 1:
                budget_manifest["aliases"].append(
                    {"representative": name, "equivalent_names": aliases}
                )

            scratch = copy.deepcopy(base_config)
            scratch["inner_widths"] = widths
            scratch.pop("resume_from", None)
            scratch["output_dir"] = f"results/{label}_{name}_scratch_seed{seed}"
            scratch_path = config_dir / f"{label}_{name}_scratch.json"
            save_json(scratch, scratch_path)
            budget_manifest["experiments"].append(
                {
                    "name": name,
                    "aliases": aliases,
                    "mode": "scratch",
                    "config": str(scratch_path),
                    "parameters": representative["parameters"],
                }
            )

            target = ResNet20(widths, source.num_classes)
            selected = transfer_by_l1(source, target)
            init_path = init_dir / f"{label}_{name}_l1_init.pt"
            torch.save(
                {
                    "epoch": 0,
                    "model_state": target.state_dict(),
                    "inner_widths": widths,
                    "num_classes": source.num_classes,
                    "transfer_method": "conv1_l1",
                    "selected_channels": selected,
                    "architecture_names": aliases,
                },
                init_path,
            )

            fine_tune = copy.deepcopy(base_config)
            fine_tune["inner_widths"] = widths
            fine_tune["resume_from"] = str(init_path)
            fine_tune["epochs"] = int(fine_tune_epochs)
            fine_tune["learning_rate"] = float(fine_tune_lr)
            fine_tune["output_dir"] = f"results/{label}_{name}_finetune_seed{seed}"
            fine_tune_path = config_dir / f"{label}_{name}_finetune.json"
            save_json(fine_tune, fine_tune_path)
            budget_manifest["experiments"].append(
                {
                    "name": name,
                    "aliases": aliases,
                    "mode": "finetune",
                    "config": str(fine_tune_path),
                    "initial_checkpoint": str(init_path),
                    "parameters": representative["parameters"],
                }
            )

        manifest["budgets"].append(budget_manifest)

    save_json(manifest, config_dir / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare scratch/fine-tune configs and transferred checkpoints"
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--architectures", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output-dir", default="configs/generated_v2")
    parser.add_argument(
        "--initializations-dir", default="results/compact_initializations"
    )
    parser.add_argument("--fine-tune-epochs", type=int, default=30)
    parser.add_argument("--fine-tune-lr", type=float, default=0.01)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.baseline_checkpoint, map_location="cpu", weights_only=False
    )
    manifest = prepare_experiments(
        load_json(args.base_config),
        load_json(args.architectures),
        checkpoint,
        args.output_dir,
        args.initializations_dir,
        args.fine_tune_epochs,
        args.fine_tune_lr,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
