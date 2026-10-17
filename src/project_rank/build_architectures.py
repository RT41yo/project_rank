from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from project_rank.architectures import (
    RANK_SOURCE_COLUMNS,
    multi_budget_architecture_report,
)
from project_rank.utils import save_json


def read_rank_columns(path: str | Path) -> dict[str, list[float]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 9:
        raise ValueError(f"Expected ranks for 9 blocks, got {len(rows)}")
    expected_blocks = [f"block{index}" for index in range(1, 10)]
    actual_blocks = [row.get("block") for row in rows]
    if actual_blocks != expected_blocks:
        raise ValueError(
            f"Expected blocks {expected_blocks}, got {actual_blocks}"
        )

    result: dict[str, list[float]] = {}
    for column in RANK_SOURCE_COLUMNS.values():
        if column not in rows[0]:
            raise ValueError(f"Column {column!r} is missing from ranks CSV")
        values = [float(row[column]) for row in rows]
        if any(value <= 0 for value in values):
            raise ValueError(f"Column {column!r} contains non-positive ranks")
        result[column] = values
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build multi-budget architectures from an existing ranks.csv"
    )
    parser.add_argument("--ranks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--target-param-ratios",
        type=float,
        nargs="+",
        default=[0.80, 0.60],
    )
    parser.add_argument("--random-seed", type=int, default=1001)
    args = parser.parse_args()

    report = multi_budget_architecture_report(
        read_rank_columns(args.ranks),
        args.target_param_ratios,
        args.random_seed,
    )
    report["source_ranks"] = str(args.ranks)
    save_json(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
