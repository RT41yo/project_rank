from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from project_rank.architectures import architecture_report
from project_rank.data import build_loaders
from project_rank.effective_rank import collect_activation_ranks, collect_weight_ranks
from project_rank.models import DEFAULT_INNER_WIDTHS, ResNet20
from project_rank.utils import load_json, resolve_device, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure ERank and propose widths")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--target-param-ratio", type=float, default=0.60)
    parser.add_argument("--random-seed", type=int, default=1001)
    args = parser.parse_args()

    config = load_json(args.config)
    seed_everything(int(config.get("seed", 1)))
    device = resolve_device(str(config.get("device", "auto")))
    _, val_loader = build_loaders(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    widths = tuple(int(value) for value in checkpoint["inner_widths"])
    if widths != tuple(DEFAULT_INNER_WIDTHS):
        raise ValueError("The first architecture search must use the full ResNet-20")

    model = ResNet20(widths, checkpoint.get("num_classes", 10)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    activation_results = collect_activation_ranks(
        model, val_loader, device, max_samples=args.max_samples
    )
    weight_results = collect_weight_ranks(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index in range(9):
        name = f"block{index + 1}"
        row = {
            "block": name,
            **activation_results[name],
            "weight_effective_rank": weight_results[name],
        }
        rows.append(row)

    csv_path = output_dir / "ranks.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ranks = [float(row["pooled_effective_rank"]) for row in rows]
    report = architecture_report(ranks, args.target_param_ratio, args.random_seed)
    report["rank_source"] = "pooled_effective_rank"
    report["max_validation_samples"] = args.max_samples
    save_json(report, output_dir / "architectures.json")
    print(
        json.dumps(
            {
                "ranks": str(csv_path),
                "architectures": str(output_dir / "architectures.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

