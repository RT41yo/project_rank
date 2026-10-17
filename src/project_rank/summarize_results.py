from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from project_rank.spec import count_resnet20_parameters
from project_rank.utils import load_json, save_json


COMPACT_PATTERN = re.compile(
    r"^(p\d{3})_(.+)_(scratch|finetune)_seed(\d+)$"
)
BASELINE_PATTERN = re.compile(r"^baseline_seed(\d+)$")


def _identity(run_name: str) -> dict[str, Any] | None:
    baseline = BASELINE_PATTERN.fullmatch(run_name)
    if baseline:
        return {
            "budget": "full",
            "method": "baseline",
            "mode": "scratch",
            "seed": int(baseline.group(1)),
        }
    compact = COMPACT_PATTERN.fullmatch(run_name)
    if compact:
        return {
            "budget": compact.group(1),
            "method": compact.group(2),
            "mode": compact.group(3),
            "seed": int(compact.group(4)),
        }
    return None


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def collect_results(results_dir: str | Path) -> dict[str, Any]:
    root = Path(results_dir)
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        identity = _identity(run_dir.name)
        config_path = run_dir / "config.json"
        history_path = run_dir / "history.json"
        if identity is None or not config_path.is_file() or not history_path.is_file():
            continue

        config = load_json(config_path)
        with history_path.open("r", encoding="utf-8") as file:
            history = json.load(file)
        if not history:
            continue
        best = max(
            history,
            key=lambda item: (
                float(item["validation"]["accuracy"]),
                -float(item["validation"]["loss"]),
            ),
        )
        final = history[-1]
        row: dict[str, Any] = {
            "run": run_dir.name,
            **identity,
            "parameters": count_resnet20_parameters(
                config["inner_widths"], int(config.get("num_classes", 10))
            ),
            "epochs_completed": len(history),
            "best_epoch": int(best["epoch"]),
            "best_validation_accuracy": float(best["validation"]["accuracy"]),
            "best_validation_loss": float(best["validation"]["loss"]),
            "final_validation_accuracy": float(final["validation"]["accuracy"]),
            "final_validation_loss": float(final["validation"]["loss"]),
            "test_accuracy": None,
            "test_loss": None,
            "macs_per_batch": None,
            "latency_median_ms": None,
            "images_per_second": None,
            "peak_cuda_memory_bytes": None,
        }

        test_path = run_dir / "test.json"
        if test_path.is_file():
            test = load_json(test_path)
            row["test_accuracy"] = float(test["accuracy"])
            row["test_loss"] = float(test["loss"])

        benchmark_path = run_dir / "benchmark.json"
        if benchmark_path.is_file():
            benchmark = load_json(benchmark_path)
            for key in (
                "macs_per_batch",
                "latency_median_ms",
                "images_per_second",
                "peak_cuda_memory_bytes",
            ):
                row[key] = benchmark.get(key)
        rows.append(row)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["budget"], row["method"], row["mode"])].append(row)

    aggregates: list[dict[str, Any]] = []
    for (budget, method, mode), items in sorted(grouped.items()):
        validation = [float(item["best_validation_accuracy"]) for item in items]
        test = [
            float(item["test_accuracy"])
            for item in items
            if item["test_accuracy"] is not None
        ]
        aggregates.append(
            {
                "budget": budget,
                "method": method,
                "mode": mode,
                "seeds": sorted(int(item["seed"]) for item in items),
                "runs": len(items),
                "best_validation_accuracy_mean": _mean(validation),
                "best_validation_accuracy_stdev": _stdev(validation),
                "test_runs": len(test),
                "test_accuracy_mean": _mean(test),
                "test_accuracy_stdev": _stdev(test),
            }
        )
    return {"schema_version": 1, "runs": rows, "aggregates": aggregates}


def _percent(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["run"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Сводка экспериментов",
        "",
        "## Отдельные запуски",
        "",
        "| Запуск | Лучшая эпоха | Validation accuracy | Test accuracy | Параметры |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        lines.append(
            f"| {row['run']} | {row['best_epoch']} | "
            f"{_percent(row['best_validation_accuracy'])} | "
            f"{_percent(row['test_accuracy'])} | {row['parameters']} |"
        )
    lines.extend(
        [
            "",
            "## Средние значения по seed",
            "",
            "| Бюджет | Метод | Режим | Seed | Validation mean ± std | Test mean ± std |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for item in report["aggregates"]:
        val_std = item["best_validation_accuracy_stdev"]
        test_std = item["test_accuracy_stdev"]
        validation = _percent(item["best_validation_accuracy_mean"])
        if val_std is not None:
            validation += f" ± {100.0 * val_std:.2f} п.п."
        test = _percent(item["test_accuracy_mean"])
        if test_std is not None:
            test += f" ± {100.0 * test_std:.2f} п.п."
        lines.append(
            f"| {item['budget']} | {item['method']} | {item['mode']} | "
            f"{', '.join(map(str, item['seeds']))} | {validation} | {test} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: dict[str, Any], output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    save_json(report, destination / "summary.json")
    _write_csv(report["runs"], destination / "runs.csv")
    _write_markdown(report, destination / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect training, test, and benchmark metrics"
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/summary")
    args = parser.parse_args()
    report = collect_results(args.results_dir)
    write_report(report, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
