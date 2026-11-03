from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from project_rank.utils import load_json, save_json


def _accuracy(path: str | Path) -> float:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Missing experiment result: {source}")
    value = float(load_json(source)["accuracy"])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid accuracy in {source}: {value}")
    return value


def summarize_channel_transfer(manifest: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for budget in manifest.get("budgets", []):
        label = str(budget["label"])
        scratch_accuracy = _accuracy(budget["scratch_test"])
        for experiment in budget.get("experiments", []):
            initial_accuracy = _accuracy(experiment["initial_test"])
            final_accuracy = _accuracy(experiment["final_test"])
            rows.append(
                {
                    "budget": label,
                    "method": str(experiment["method"]),
                    "scratch_test_accuracy": scratch_accuracy,
                    "initial_test_accuracy": initial_accuracy,
                    "final_test_accuracy": final_accuracy,
                    "fine_tune_gain_pp": 100.0
                    * (final_accuracy - initial_accuracy),
                    "delta_vs_scratch_pp": 100.0
                    * (final_accuracy - scratch_accuracy),
                    "reused_existing_result": bool(
                        experiment.get("reused_existing_result", False)
                    ),
                }
            )
    return {
        "schema_version": 1,
        "architecture": manifest.get("architecture"),
        "train_seed": manifest.get("train_seed"),
        "selection_seed": manifest.get("selection_seed"),
        "rows": rows,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Сравнение способов переноса каналов",
        "",
        "| Бюджет | Способ | До fine-tune, % | После fine-tune, % | "
        "Изменение, п.п. | Scratch, % | Разница со scratch, п.п. |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {budget} | `{method}` | {initial:.2f} | {final:.2f} | "
            "{gain:+.2f} | {scratch:.2f} | {delta:+.2f} |".format(
                budget=row["budget"],
                method=row["method"],
                initial=100.0 * row["initial_test_accuracy"],
                final=100.0 * row["final_test_accuracy"],
                gain=row["fine_tune_gain_pp"],
                scratch=100.0 * row["scratch_test_accuracy"],
                delta=row["delta_vs_scratch_pp"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the fixed-width channel-transfer study"
    )
    parser.add_argument(
        "--manifest", default="configs/channel_transfer_seed1/manifest.json"
    )
    parser.add_argument(
        "--output-dir", default="results/channel_transfer_seed1/summary"
    )
    args = parser.parse_args()

    report = summarize_channel_transfer(load_json(args.manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, output_dir / "summary.json")
    _write_csv(report["rows"], output_dir / "summary.csv")
    _write_markdown(report, output_dir / "summary.md")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
