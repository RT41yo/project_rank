from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from project_rank.utils import load_json, save_json


def _architecture_map(data: dict) -> dict[str, list[int]]:
    return {
        str(item["name"]): [int(value) for value in item["inner_widths"]]
        for item in data["architectures"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare configs for compact models")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--architectures", required=True)
    parser.add_argument("--transferred-checkpoint", required=True)
    parser.add_argument("--output-dir", default="configs/generated")
    parser.add_argument("--fine-tune-epochs", type=int, default=30)
    parser.add_argument("--fine-tune-lr", type=float, default=0.01)
    args = parser.parse_args()

    base = load_json(args.base_config)
    architectures = _architecture_map(load_json(args.architectures))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[str] = []
    for name in ("erank", "random", "uniform"):
        config = copy.deepcopy(base)
        config["inner_widths"] = architectures[name]
        config["output_dir"] = f"results/{name}_scratch_seed{config.get('seed', 1)}"
        config.pop("resume_from", None)
        path = output_dir / f"{name}_scratch.json"
        save_json(config, path)
        created.append(str(path))

    fine_tune = copy.deepcopy(base)
    fine_tune["inner_widths"] = architectures["erank"]
    fine_tune["resume_from"] = args.transferred_checkpoint
    fine_tune["epochs"] = args.fine_tune_epochs
    fine_tune["learning_rate"] = args.fine_tune_lr
    fine_tune["output_dir"] = f"results/erank_finetune_seed{fine_tune.get('seed', 1)}"
    fine_tune_path = output_dir / "erank_finetune.json"
    save_json(fine_tune, fine_tune_path)
    created.append(str(fine_tune_path))

    print(json.dumps({"created": created}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

