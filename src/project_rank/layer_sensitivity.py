from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from project_rank.build_pruning_architectures import read_pruning_scores
from project_rank.data import build_loaders
from project_rank.models import DEFAULT_INNER_WIDTHS, ResNet20
from project_rank.spec import count_resnet20_parameters
from project_rank.train import _run_epoch
from project_rank.transfer import transfer_by_l1
from project_rank.utils import load_json, resolve_device, save_json, seed_everything


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[order[position]] = average
        index = end
    return ranks


def spearman_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    x = _rankdata(left)
    y = _rankdata(right)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x)
        * sum((b - y_mean) ** 2 for b in y)
    )
    return None if denominator == 0.0 else numerator / denominator


def _reduced_width(width: int, reduction_ratio: float) -> int:
    proposed = int(round(width * (1.0 - reduction_ratio) / 2.0) * 2)
    return min(width - 2, max(4, proposed))


def _low_score_fraction(scores: list[float], removed: int) -> float:
    total = sum(scores)
    if total <= 0.0:
        return 0.0
    return sum(sorted(scores)[:removed]) / total


def analyze_layer_sensitivity(
    source: ResNet20,
    loader: Any,
    device: torch.device,
    ranks: list[dict[str, str]],
    pruning_scores: dict[str, list[list[float]]],
    reduction_ratio: float,
    amp_enabled: bool,
) -> dict[str, Any]:
    if not 0.0 < reduction_ratio < 1.0:
        raise ValueError("reduction_ratio must be in (0, 1)")
    criterion = nn.CrossEntropyLoss()
    baseline = _run_epoch(
        source, loader, criterion, device, optimizer=None, scaler=None,
        amp_enabled=amp_enabled,
    )
    rows: list[dict[str, Any]] = []
    for index, base_width in enumerate(DEFAULT_INNER_WIDTHS):
        reduced = _reduced_width(base_width, reduction_ratio)
        widths = list(DEFAULT_INNER_WIDTHS)
        widths[index] = reduced
        target = ResNet20(widths, source.num_classes).to(device)
        transfer_by_l1(source, target)
        metrics = _run_epoch(
            target, loader, criterion, device, optimizer=None, scaler=None,
            amp_enabled=amp_enabled,
        )
        rank_row = ranks[index]
        removed = base_width - reduced
        row = {
            "block": f"block{index + 1}",
            "base_width": base_width,
            "reduced_width": reduced,
            "removed_channels": removed,
            "parameters": count_resnet20_parameters(widths),
            "validation_accuracy": metrics["accuracy"],
            "accuracy_drop": baseline["accuracy"] - metrics["accuracy"],
            "erank_per_image_ratio": float(
                rank_row["mean_per_image_effective_rank"]
            ) / base_width,
            "erank_pooled_ratio": float(rank_row["pooled_effective_rank"]) / base_width,
            "erank_weight_ratio": float(rank_row["weight_effective_rank"]) / base_width,
            "l1_removed_score_fraction": _low_score_fraction(
                pruning_scores["l1_weight"][index], removed
            ),
            "taylor_removed_score_fraction": _low_score_fraction(
                pruning_scores["taylor"][index], removed
            ),
        }
        rows.append(row)

    drops = [float(row["accuracy_drop"]) for row in rows]
    predictors = [
        "erank_per_image_ratio",
        "erank_pooled_ratio",
        "erank_weight_ratio",
        "l1_removed_score_fraction",
        "taylor_removed_score_fraction",
    ]
    correlations = {
        name: spearman_correlation(
            [float(row[name]) for row in rows], drops
        )
        for name in predictors
    }
    return {
        "schema_version": 1,
        "reduction_ratio": reduction_ratio,
        "transfer_method": "conv1_l1",
        "fine_tuning": False,
        "baseline_validation": baseline,
        "blocks": rows,
        "spearman_vs_accuracy_drop": correlations,
    }


def _read_ranks(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if [row.get("block") for row in rows] != [f"block{i}" for i in range(1, 10)]:
        raise ValueError("Expected ordered rows block1 ... block9 in ranks.csv")
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure validation sensitivity when one block is narrowed"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ranks", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reduction-ratio", type=float, default=0.25)
    args = parser.parse_args()

    config = load_json(args.config)
    seed_everything(int(config.get("seed", 1)))
    device = resolve_device(str(config.get("device", "auto")))
    _, validation_loader = build_loaders(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(checkpoint["inner_widths"]) != tuple(DEFAULT_INNER_WIDTHS):
        raise ValueError("Layer sensitivity requires the full ResNet-20 checkpoint")
    source = ResNet20(
        checkpoint["inner_widths"], checkpoint.get("num_classes", 10)
    ).to(device)
    source.load_state_dict(checkpoint["model_state"])
    score_data = load_json(args.scores)
    result = analyze_layer_sensitivity(
        source,
        validation_loader,
        device,
        _read_ranks(args.ranks),
        read_pruning_scores(score_data),
        reduction_ratio=args.reduction_ratio,
        amp_enabled=bool(config.get("amp", True)) and device.type == "cuda",
    )
    result["checkpoint"] = str(args.checkpoint)
    result["ranks"] = str(args.ranks)
    result["scores"] = str(args.scores)
    output_dir = Path(args.output_dir)
    save_json(result, output_dir / "sensitivity.json")
    _write_csv(result["blocks"], output_dir / "sensitivity.csv")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
