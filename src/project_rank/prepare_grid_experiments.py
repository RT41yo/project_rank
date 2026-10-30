from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from project_rank.architectures import make_random_architecture
from project_rank.utils import load_json, save_json


DEFAULT_METHODS = (
    "erank_per_image",
    "erank_pooled",
    "erank_weight",
    "l1_weight",
    "taylor",
    "uniform",
)
DEFAULT_BUDGETS = tuple(f"p{value:03d}" for value in range(90, 20, -10))
DEFAULT_RANDOM_ARCHITECTURE_SEEDS = tuple(range(1001, 1011))


def _has_history(path: Path) -> bool:
    history = path / "history.json"
    if not history.is_file():
        return False
    try:
        data = load_json(history)
    except (OSError, ValueError, TypeError):
        return False
    return isinstance(data, list) and bool(data)


def _legacy_random_result(
    results_dir: Path,
    budget: str,
    train_seed: int,
    inner_widths: Sequence[int],
) -> Path | None:
    candidate = results_dir / f"{budget}_random_scratch_seed{train_seed}"
    config_path = candidate / "config.json"
    if not config_path.is_file() or not _has_history(candidate):
        return None
    config = load_json(config_path)
    existing = tuple(int(value) for value in config.get("inner_widths", []))
    return candidate if existing == tuple(inner_widths) else None


def _write_commands(manifest: dict[str, Any], destination: Path) -> None:
    train_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated commands: train only experiments that are not already complete."
    ]
    test_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated commands: evaluate every available grid checkpoint on test."
    ]
    for experiment in manifest["experiments"]:
        config = str(experiment["config"])
        result_dir = str(experiment["effective_result_dir"])
        if experiment["needs_training"]:
            train_lines.append(
                f"python -m project_rank.train --config {config}"
            )
        test_lines.append(
            "python -m project_rank.evaluate "
            f"--config {result_dir}/config.json "
            f"--checkpoint {result_dir}/best.pt "
            f"--split test --output {result_dir}/test.json"
        )
    destination.mkdir(parents=True, exist_ok=True)
    destination.joinpath("train_commands.txt").write_text(
        "\n".join(train_lines) + "\n", encoding="utf-8"
    )
    destination.joinpath("test_commands.txt").write_text(
        "\n".join(test_lines) + "\n", encoding="utf-8"
    )


def prepare_grid_experiments(
    base_config: dict[str, Any],
    architecture_report: dict[str, Any],
    output_dir: str | Path,
    results_dir: str | Path,
    budgets: Sequence[str] = DEFAULT_BUDGETS,
    methods: Sequence[str] = DEFAULT_METHODS,
    train_seed: int = 1,
    random_architecture_seeds: Sequence[int] = DEFAULT_RANDOM_ARCHITECTURE_SEEDS,
) -> dict[str, Any]:
    """Create a seed-1 scratch grid and mark reusable completed runs.

    Deterministic methods are deduplicated by their complete nine-width profile.
    Random-width architectures are generated independently for each explicit
    architecture seed. Training randomness is kept separate as ``train_seed``.
    """
    if architecture_report.get("schema_version") != 3:
        raise ValueError("Expected architecture report schema_version=3")
    if train_seed < 0:
        raise ValueError("train_seed must be non-negative")
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("budgets must be a non-empty sequence without duplicates")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be a non-empty sequence without duplicates")
    if not random_architecture_seeds:
        raise ValueError("At least one random architecture seed is required")
    if len(set(random_architecture_seeds)) != len(random_architecture_seeds):
        raise ValueError("random architecture seeds must be unique")

    destination = Path(output_dir)
    results_root = Path(results_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_budgets = {
        str(item["label"]): item for item in architecture_report.get("budgets", [])
    }
    missing_budgets = set(budgets) - set(report_budgets)
    if missing_budgets:
        raise ValueError(f"Missing budgets: {sorted(missing_budgets)}")

    full_parameters = int(architecture_report["full_architecture"]["parameters"])
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment_family": "one_shot_scratch_grid",
        "train_seed": int(train_seed),
        "budgets": list(budgets),
        "methods": list(methods),
        "random_architecture_seeds_requested": [
            int(seed) for seed in random_architecture_seeds
        ],
        "full_parameters": full_parameters,
        "experiments": [],
    }

    for budget_label in budgets:
        budget = report_budgets[budget_label]
        available = {
            str(item["name"]): item for item in budget.get("architectures", [])
        }
        missing_methods = set(methods) - set(available)
        if missing_methods:
            raise ValueError(
                f"Missing methods in {budget_label}: {sorted(missing_methods)}"
            )

        width_groups: dict[tuple[int, ...], list[str]] = {}
        for method in methods:
            widths = tuple(int(value) for value in available[method]["inner_widths"])
            width_groups.setdefault(widths, []).append(method)

        for widths, aliases in width_groups.items():
            representative = aliases[0]
            architecture = available[representative]
            run_name = (
                f"{budget_label}_{representative}_scratch_seed{train_seed}"
            )
            result_dir = results_root / run_name
            config_path = destination / f"{run_name}.json"
            config = copy.deepcopy(base_config)
            config["seed"] = int(train_seed)
            config["inner_widths"] = list(widths)
            config.pop("resume_from", None)
            config["output_dir"] = str(result_dir)
            save_json(config, config_path)
            complete = _has_history(result_dir)
            manifest["experiments"].append(
                {
                    "run": run_name,
                    "budget": budget_label,
                    "method": representative,
                    "method_family": "deterministic_width_rule",
                    "aliases": aliases,
                    "mode": "scratch",
                    "train_seed": int(train_seed),
                    "architecture_seed": None,
                    "parameters": int(architecture["parameters"]),
                    "parameter_ratio_vs_full": (
                        int(architecture["parameters"]) / full_parameters
                    ),
                    "inner_widths": list(widths),
                    "config": str(config_path),
                    "result_dir": str(result_dir),
                    "effective_result_dir": str(result_dir),
                    "existing_legacy_run": None,
                    "needs_training": not complete,
                }
            )

        reference_parameters = int(budget["reference_parameters"])
        seen_random_widths: dict[tuple[int, ...], int] = {}
        for architecture_seed in random_architecture_seeds:
            architecture = make_random_architecture(
                reference_parameters,
                seed=int(architecture_seed),
                multiple=2,
            )
            widths = tuple(architecture.inner_widths)
            if widths in seen_random_widths:
                raise ValueError(
                    "Random architecture seeds produced a duplicate width profile: "
                    f"{seen_random_widths[widths]} and {architecture_seed}"
                )
            seen_random_widths[widths] = int(architecture_seed)
            run_name = (
                f"{budget_label}_random_arch{architecture_seed}"
                f"_scratch_seed{train_seed}"
            )
            result_dir = results_root / run_name
            config_path = destination / f"{run_name}.json"
            config = copy.deepcopy(base_config)
            config["seed"] = int(train_seed)
            config["inner_widths"] = list(widths)
            config.pop("resume_from", None)
            config["output_dir"] = str(result_dir)
            config["architecture_seed"] = int(architecture_seed)
            save_json(config, config_path)

            legacy = _legacy_random_result(
                results_root,
                budget_label,
                train_seed,
                widths,
            )
            effective_result = legacy or result_dir
            complete = _has_history(effective_result)
            manifest["experiments"].append(
                {
                    "run": run_name,
                    "budget": budget_label,
                    "method": "random",
                    "method_family": "random_width_rule",
                    "aliases": ["random"],
                    "mode": "scratch",
                    "train_seed": int(train_seed),
                    "architecture_seed": int(architecture_seed),
                    "parameters": architecture.parameters,
                    "parameter_ratio_vs_full": (
                        architecture.parameters / full_parameters
                    ),
                    "inner_widths": list(widths),
                    "config": str(config_path),
                    "result_dir": str(result_dir),
                    "effective_result_dir": str(effective_result),
                    "existing_legacy_run": (
                        str(legacy) if legacy is not None else None
                    ),
                    "needs_training": not complete,
                }
            )

    manifest["counts"] = {
        "total": len(manifest["experiments"]),
        "already_complete": sum(
            not bool(item["needs_training"]) for item in manifest["experiments"]
        ),
        "needs_training": sum(
            bool(item["needs_training"]) for item in manifest["experiments"]
        ),
    }
    save_json(manifest, destination / "manifest.json")
    _write_commands(manifest, destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a multi-budget seed-1 scratch experiment grid"
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--architectures", required=True)
    parser.add_argument("--output-dir", default="configs/generated_v4")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--budgets", nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument(
        "--random-architecture-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_RANDOM_ARCHITECTURE_SEEDS),
    )
    args = parser.parse_args()
    manifest = prepare_grid_experiments(
        load_json(args.base_config),
        load_json(args.architectures),
        args.output_dir,
        args.results_dir,
        budgets=args.budgets,
        methods=args.methods,
        train_seed=args.train_seed,
        random_architecture_seeds=args.random_architecture_seeds,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
