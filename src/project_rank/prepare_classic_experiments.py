from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from project_rank.utils import load_json, save_json


DEFAULT_RUNS = {
    "p080": {"l1_weight": [1], "taylor": [1]},
    "p060": {"l1_weight": [1, 2, 3], "taylor": [1, 2, 3]},
}


def prepare_classic_experiments(
    base_config: dict[str, Any],
    architecture_report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    if architecture_report.get("schema_version") != 3:
        raise ValueError("Expected architecture report schema_version=3")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    budgets = {
        str(item["label"]): item for item in architecture_report.get("budgets", [])
    }
    manifest: dict[str, Any] = {"schema_version": 1, "experiments": []}

    for budget_label, methods in DEFAULT_RUNS.items():
        if budget_label not in budgets:
            raise ValueError(f"Missing budget {budget_label!r}")
        architectures = {
            str(item["name"]): item for item in budgets[budget_label]["architectures"]
        }
        for method, seeds in methods.items():
            if method not in architectures:
                raise ValueError(f"Missing architecture {method!r} in {budget_label}")
            architecture = architectures[method]
            widths = [int(value) for value in architecture["inner_widths"]]
            for seed in seeds:
                config = copy.deepcopy(base_config)
                config["seed"] = seed
                config["inner_widths"] = widths
                config.pop("resume_from", None)
                run_name = f"{budget_label}_{method}_scratch_seed{seed}"
                config["output_dir"] = f"results/{run_name}"
                path = destination / f"{run_name}.json"
                save_json(config, path)
                manifest["experiments"].append(
                    {
                        "run": run_name,
                        "budget": budget_label,
                        "method": method,
                        "mode": "scratch",
                        "seed": seed,
                        "parameters": architecture["parameters"],
                        "inner_widths": widths,
                        "config": str(path),
                    }
                )

    save_json(manifest, destination / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the eight v06 L1/Taylor scratch experiments"
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--architectures", required=True)
    parser.add_argument("--output-dir", default="configs/generated_v3")
    args = parser.parse_args()
    manifest = prepare_classic_experiments(
        load_json(args.base_config),
        load_json(args.architectures),
        args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
