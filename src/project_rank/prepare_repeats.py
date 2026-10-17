from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from project_rank.utils import load_json, save_json


DEFAULT_METHODS = ("erank_pooled", "erank_per_image", "uniform")


def _resolve_config(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def prepare_repeat_configs(
    base_config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    project_root: str | Path,
    output_dir: str | Path,
    budget_label: str,
    methods: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    """Create scratch configs for fixed architectures and additional seeds."""
    unique_seeds = list(dict.fromkeys(int(seed) for seed in seeds))
    if not unique_seeds or any(seed < 0 for seed in unique_seeds):
        raise ValueError("At least one non-negative seed is required")
    if not methods:
        raise ValueError("At least one compact method is required")

    budget = next(
        (
            item
            for item in manifest.get("budgets", [])
            if item.get("label") == budget_label
        ),
        None,
    )
    if budget is None:
        raise ValueError(f"Budget {budget_label!r} was not found in the manifest")

    scratch_by_name: dict[str, dict[str, Any]] = {}
    for experiment in budget.get("experiments", []):
        if experiment.get("mode") != "scratch":
            continue
        names = [str(experiment["name"]), *map(str, experiment.get("aliases", []))]
        for name in names:
            scratch_by_name[name] = experiment

    missing = [method for method in methods if method not in scratch_by_name]
    if missing:
        raise ValueError(f"Scratch methods are missing from the manifest: {missing}")

    root = Path(project_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []

    for seed in unique_seeds:
        baseline = copy.deepcopy(base_config)
        baseline["seed"] = seed
        baseline["output_dir"] = f"results/baseline_seed{seed}"
        baseline.pop("resume_from", None)
        baseline_path = destination / f"baseline_seed{seed}.json"
        save_json(baseline, baseline_path)
        generated.append(
            {
                "budget": "full",
                "method": "baseline",
                "mode": "scratch",
                "seed": seed,
                "config": str(baseline_path),
            }
        )

        for method in methods:
            experiment = scratch_by_name[method]
            source = load_json(
                _resolve_config(root, str(experiment["config"]))
            )
            source["seed"] = seed
            source["output_dir"] = (
                f"results/{budget_label}_{method}_scratch_seed{seed}"
            )
            source.pop("resume_from", None)
            config_path = destination / (
                f"{budget_label}_{method}_scratch_seed{seed}.json"
            )
            save_json(source, config_path)
            generated.append(
                {
                    "budget": budget_label,
                    "method": method,
                    "mode": "scratch",
                    "seed": seed,
                    "config": str(config_path),
                    "parameters": experiment.get("parameters"),
                }
            )

    repeat_manifest = {
        "schema_version": 1,
        "source_manifest_seed": manifest.get("seed"),
        "fixed_architectures": True,
        "budget": budget_label,
        "methods": methods,
        "seeds": unique_seeds,
        "experiments": generated,
    }
    save_json(repeat_manifest, destination / "repeat_manifest.json")
    return repeat_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare repeat scratch configs for fixed architectures"
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="configs/repeats")
    parser.add_argument("--budget", default="p060")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2, 3])
    args = parser.parse_args()

    result = prepare_repeat_configs(
        load_json(args.base_config),
        load_json(args.manifest),
        project_root=args.project_root,
        output_dir=args.output_dir,
        budget_label=args.budget,
        methods=args.methods,
        seeds=args.seeds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
