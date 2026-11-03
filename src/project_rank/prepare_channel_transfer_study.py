from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import torch

from project_rank.data import build_loaders
from project_rank.models import DEFAULT_INNER_WIDTHS, ResNet20
from project_rank.transfer import (
    collect_activation_covariances,
    transfer_by_activation_subspace,
    transfer_by_l1,
    transfer_by_random,
)
from project_rank.utils import load_json, resolve_device, save_json, seed_everything


METHODS = ("l1", "random", "erank_subspace")


def _find_widths(
    architecture_report: dict[str, Any], budget_label: str, architecture_name: str
) -> list[int]:
    for budget in architecture_report.get("budgets", []):
        if budget.get("label") != budget_label:
            continue
        for architecture in budget.get("architectures", []):
            if architecture.get("name") == architecture_name:
                return [int(value) for value in architecture["inner_widths"]]
    raise KeyError(
        f"Architecture {architecture_name!r} for {budget_label!r} was not found"
    )


def _shell_header() -> list[str]:
    return ["#!/usr/bin/env bash", "set -euo pipefail", ""]


def _save_shell(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    path.chmod(0o755)


def prepare_channel_transfer_study(
    base_config: dict[str, Any],
    architecture_report: dict[str, Any],
    baseline_checkpoint: dict[str, Any],
    covariances: dict[str, torch.Tensor],
    config_dir: str | Path,
    initialization_dir: str | Path,
    result_root: str | Path,
    budgets: list[str],
    architecture_name: str = "erank_per_image",
    fine_tune_epochs: int = 30,
    fine_tune_lr: float = 0.01,
    selection_seed: int = 2001,
    max_samples: int = 5000,
) -> dict[str, Any]:
    if tuple(baseline_checkpoint["inner_widths"]) != tuple(DEFAULT_INNER_WIDTHS):
        raise ValueError("Baseline checkpoint must contain the full ResNet-20")
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("Budgets must be a non-empty list without duplicates")

    config_path = Path(config_dir)
    init_path = Path(initialization_dir)
    results_path = Path(result_root)
    config_path.mkdir(parents=True, exist_ok=True)
    init_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    source = ResNet20(
        baseline_checkpoint["inner_widths"],
        baseline_checkpoint.get("num_classes", 10),
    )
    source.load_state_dict(baseline_checkpoint["model_state"])
    train_seed = int(base_config.get("seed", 1))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "architecture": architecture_name,
        "train_seed": train_seed,
        "selection_seed": int(selection_seed),
        "activation_samples": int(max_samples),
        "fine_tune_epochs": int(fine_tune_epochs),
        "fine_tune_learning_rate": float(fine_tune_lr),
        "budgets": [],
    }
    selection_report: dict[str, Any] = {
        "schema_version": 1,
        "architecture": architecture_name,
        "selection_seed": int(selection_seed),
        "budgets": {},
    }
    initial_lines = _shell_header()
    train_lines = _shell_header()
    test_lines = _shell_header()

    for budget_label in budgets:
        widths = _find_widths(
            architecture_report, budget_label, architecture_name
        )
        budget_data: dict[str, Any] = {
            "label": budget_label,
            "inner_widths": widths,
            "scratch_test": (
                f"results/{budget_label}_{architecture_name}_scratch_seed{train_seed}"
                "/test.json"
            ),
            "experiments": [],
        }
        selection_report["budgets"][budget_label] = {}

        for method in METHODS:
            target = ResNet20(widths, source.num_classes)
            scores = None
            if method == "l1":
                selected = transfer_by_l1(source, target)
                transfer_method = "conv1_l1"
            elif method == "random":
                selected = transfer_by_random(source, target, selection_seed)
                transfer_method = "random_channels"
            else:
                selected, scores = transfer_by_activation_subspace(
                    source, target, covariances
                )
                transfer_method = "erank_activation_subspace"

            init_file = init_path / f"{budget_label}_{method}_init.pt"
            torch.save(
                {
                    "epoch": 0,
                    "model_state": target.state_dict(),
                    "inner_widths": widths,
                    "num_classes": source.num_classes,
                    "source_checkpoint": "results/baseline_seed1/best.pt",
                    "transfer_method": transfer_method,
                    "selection_seed": (
                        int(selection_seed) if method == "random" else None
                    ),
                    "selected_channels": selected,
                    "channel_scores": scores,
                },
                init_file,
            )

            if budget_label == "p060" and method == "l1" and train_seed == 1:
                output_dir = Path(
                    "results/p060_erank_per_image_finetune_seed1"
                )
                reused_existing_result = True
            else:
                output_dir = results_path / (
                    f"{budget_label}_{method}_finetune_seed{train_seed}"
                )
                reused_existing_result = False

            config = copy.deepcopy(base_config)
            config["inner_widths"] = widths
            config["resume_from"] = str(init_file)
            config["epochs"] = int(fine_tune_epochs)
            config["learning_rate"] = float(fine_tune_lr)
            config["output_dir"] = str(output_dir)
            config["transfer_method"] = transfer_method
            if method == "random":
                config["channel_selection_seed"] = int(selection_seed)
            config_file = config_path / (
                f"{budget_label}_{method}_finetune_seed{train_seed}.json"
            )
            save_json(config, config_file)

            initial_test = results_path / f"{budget_label}_{method}_initial_test.json"
            final_test = output_dir / "test.json"
            experiment = {
                "method": method,
                "transfer_method": transfer_method,
                "initial_checkpoint": str(init_file),
                "initial_test": str(initial_test),
                "config": str(config_file),
                "output_dir": str(output_dir),
                "final_test": str(final_test),
                "reused_existing_result": reused_existing_result,
            }
            budget_data["experiments"].append(experiment)
            selection_report["budgets"][budget_label][method] = {
                "selected_channels": selected,
                "channel_scores": scores,
            }

            initial_lines.extend(
                [
                    f"python -m project_rank.evaluate \\",
                    f"  --config {config_file} \\",
                    f"  --checkpoint {init_file} \\",
                    "  --split test \\",
                    f"  --output {initial_test}",
                    "",
                ]
            )
            train_lines.extend(
                [
                    f"if [[ ! -s {output_dir / 'best.pt'} ]]; then",
                    f"  python -m project_rank.train --config {config_file}",
                    "fi",
                    "",
                ]
            )
            test_lines.extend(
                [
                    f"if [[ ! -s {final_test} ]]; then",
                    f"  python -m project_rank.evaluate \\",
                    f"    --config {config_file} \\",
                    f"    --checkpoint {output_dir / 'best.pt'} \\",
                    "    --split test \\",
                    f"    --output {final_test}",
                    "fi",
                    "",
                ]
            )

        manifest["budgets"].append(budget_data)

    save_json(manifest, config_path / "manifest.json")
    save_json(selection_report, results_path / "channel_selection.json")
    _save_shell(config_path / "initial_test_commands.sh", initial_lines)
    _save_shell(config_path / "train_commands.sh", train_lines)
    _save_shell(config_path / "test_commands.sh", test_lines)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed-width channel-transfer study"
    )
    parser.add_argument("--base-config", default="configs/baseline.json")
    parser.add_argument("--architectures", required=True)
    parser.add_argument(
        "--baseline-checkpoint", default="results/baseline_seed1/best.pt"
    )
    parser.add_argument(
        "--config-dir", default="configs/channel_transfer_seed1"
    )
    parser.add_argument(
        "--initialization-dir",
        default="results/channel_transfer_initializations_seed1",
    )
    parser.add_argument(
        "--result-root", default="results/channel_transfer_seed1"
    )
    parser.add_argument("--budgets", nargs="+", default=["p060", "p030"])
    parser.add_argument("--architecture", default="erank_per_image")
    parser.add_argument("--fine-tune-epochs", type=int, default=30)
    parser.add_argument("--fine-tune-lr", type=float, default=0.01)
    parser.add_argument("--selection-seed", type=int, default=2001)
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    base_config = load_json(args.base_config)
    seed_everything(int(base_config.get("seed", 1)))
    device = resolve_device(str(base_config.get("device", "auto")))
    checkpoint = torch.load(
        args.baseline_checkpoint, map_location="cpu", weights_only=False
    )
    source = ResNet20(
        checkpoint["inner_widths"], checkpoint.get("num_classes", 10)
    ).to(device)
    source.load_state_dict(checkpoint["model_state"])
    _, validation_loader = build_loaders(base_config)
    covariances = collect_activation_covariances(
        source,
        validation_loader,
        device,
        max_samples=args.max_samples,
    )

    manifest = prepare_channel_transfer_study(
        base_config,
        load_json(args.architectures),
        checkpoint,
        covariances,
        args.config_dir,
        args.initialization_dir,
        args.result_root,
        budgets=list(args.budgets),
        architecture_name=args.architecture,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_lr=args.fine_tune_lr,
        selection_seed=args.selection_seed,
        max_samples=args.max_samples,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
