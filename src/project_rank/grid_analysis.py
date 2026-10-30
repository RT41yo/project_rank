from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from project_rank.spec import (
    DEFAULT_INNER_WIDTHS,
    count_resnet20_macs,
    count_resnet20_parameters,
)


METHOD_ORDER = (
    "erank_per_image",
    "erank_pooled",
    "erank_weight",
    "l1_weight",
    "taylor",
    "uniform",
    "random",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _best_validation(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        raise ValueError("history.json is empty")
    best = max(
        history,
        key=lambda item: (
            float(item["validation"]["accuracy"]),
            -float(item["validation"]["loss"]),
        ),
    )
    return {
        "epoch": int(best["epoch"]),
        "accuracy": float(best["validation"]["accuracy"]),
        "loss": float(best["validation"]["loss"]),
    }


def _result_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _read_run(
    project_root: Path,
    experiment: dict[str, Any],
    full_parameters: int,
    full_macs: int,
) -> dict[str, Any]:
    result_dir = _result_path(
        project_root,
        str(experiment.get("effective_result_dir") or experiment["result_dir"]),
    )
    config = _load_json(result_dir / "config.json")
    history = _load_json(result_dir / "history.json")
    test = _load_json(result_dir / "test.json")
    widths = [int(value) for value in experiment["inner_widths"]]
    config_widths = [int(value) for value in config["inner_widths"]]
    if config_widths != widths:
        raise ValueError(
            f"Width mismatch for {experiment['run']}: {widths} != {config_widths}"
        )

    num_classes = int(config.get("num_classes", 10))
    parameters = count_resnet20_parameters(widths, num_classes)
    if parameters != int(experiment["parameters"]):
        raise ValueError(
            f"Parameter mismatch for {experiment['run']}: "
            f"{parameters} != {experiment['parameters']}"
        )
    macs = count_resnet20_macs(widths, num_classes)
    samples = int(float(test["samples"]))
    if samples != 10_000:
        raise ValueError(
            f"Expected 10000 test samples for {experiment['run']}, got {samples}"
        )

    return {
        "run": str(experiment["run"]),
        "effective_result_dir": str(
            experiment.get("effective_result_dir") or experiment["result_dir"]
        ),
        "budget": str(experiment["budget"]),
        "method": str(experiment["method"]),
        "aliases": [str(value) for value in experiment.get("aliases", [])],
        "method_family": str(experiment["method_family"]),
        "mode": str(experiment["mode"]),
        "train_seed": int(experiment["train_seed"]),
        "architecture_seed": (
            int(experiment["architecture_seed"])
            if experiment.get("architecture_seed") is not None
            else None
        ),
        "inner_widths": widths,
        "parameters": parameters,
        "parameter_ratio_vs_full": parameters / full_parameters,
        "macs_per_image": macs,
        "mac_ratio_vs_full": macs / full_macs,
        "epochs_completed": len(history),
        "best_validation": _best_validation(history),
        "test_accuracy": float(test["accuracy"]),
        "test_loss": float(test["loss"]),
        "test_samples": samples,
    }


def _read_baseline(project_root: Path, baseline_dir: str | Path) -> dict[str, Any]:
    directory = _result_path(project_root, str(baseline_dir))
    config = _load_json(directory / "config.json")
    history = _load_json(directory / "history.json")
    test = _load_json(directory / "test.json")
    widths = [int(value) for value in config.get("inner_widths", DEFAULT_INNER_WIDTHS)]
    num_classes = int(config.get("num_classes", 10))
    parameters = count_resnet20_parameters(widths, num_classes)
    macs = count_resnet20_macs(widths, num_classes)
    result = {
        "run": directory.name,
        "train_seed": int(config.get("seed", 1)),
        "inner_widths": widths,
        "parameters": parameters,
        "parameter_ratio_vs_full": 1.0,
        "macs_per_image": macs,
        "mac_ratio_vs_full": 1.0,
        "epochs_completed": len(history),
        "best_validation": _best_validation(history),
        "test_accuracy": float(test["accuracy"]),
        "test_loss": float(test["loss"]),
        "test_samples": int(float(test["samples"])),
    }
    benchmark_path = directory / "benchmark.json"
    if benchmark_path.is_file():
        result["benchmark"] = _load_json(benchmark_path)
    environment_path = directory / "environment.json"
    if environment_path.is_file():
        result["environment"] = _load_json(environment_path)
    return result


def _logical_points(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for run in runs:
        aliases = run["aliases"] or [run["method"]]
        for method in aliases:
            point = {
                key: value
                for key, value in run.items()
                if key not in {"aliases", "method"}
            }
            point["method"] = method
            point["physical_method"] = run["method"]
            point["shared_physical_run"] = len(aliases) > 1
            points.append(point)
    return points


def _random_summary(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        if point["method"] == "random":
            groups[point["budget"]].append(point)

    result = []
    for budget, items in groups.items():
        accuracy = [float(item["test_accuracy"]) for item in items]
        parameters = [int(item["parameters"]) for item in items]
        macs = [int(item["macs_per_image"]) for item in items]
        result.append(
            {
                "budget": budget,
                "count": len(items),
                "architecture_seeds": sorted(
                    int(item["architecture_seed"]) for item in items
                ),
                "test_accuracy_mean": statistics.mean(accuracy),
                "test_accuracy_stdev": (
                    statistics.stdev(accuracy) if len(accuracy) > 1 else 0.0
                ),
                "test_accuracy_min": min(accuracy),
                "test_accuracy_max": max(accuracy),
                "parameters_mean": statistics.mean(parameters),
                "parameter_ratio_mean": statistics.mean(
                    float(item["parameter_ratio_vs_full"]) for item in items
                ),
                "macs_per_image_mean": statistics.mean(macs),
                "mac_ratio_mean": statistics.mean(
                    float(item["mac_ratio_vs_full"]) for item in items
                ),
            }
        )
    return sorted(result, key=lambda item: item["budget"], reverse=True)


def _budget_summary(
    points: list[dict[str, Any]],
    random_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    random_by_budget = {item["budget"]: item for item in random_summary}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        if point["method"] != "random":
            groups[point["budget"]].append(point)

    summaries = []
    for budget, items in groups.items():
        winner = max(items, key=lambda item: float(item["test_accuracy"]))
        random = random_by_budget[budget]
        summaries.append(
            {
                "budget": budget,
                "winner_method": winner["method"],
                "winner_test_accuracy": winner["test_accuracy"],
                "winner_parameters": winner["parameters"],
                "winner_parameter_ratio": winner["parameter_ratio_vs_full"],
                "winner_macs_per_image": winner["macs_per_image"],
                "winner_mac_ratio": winner["mac_ratio_vs_full"],
                "winner_minus_random_mean": (
                    float(winner["test_accuracy"])
                    - float(random["test_accuracy_mean"])
                ),
            }
        )
    return sorted(summaries, key=lambda item: item["budget"], reverse=True)


def collect_grid_results(
    manifest_path: str | Path,
    project_root: str | Path = ".",
    baseline_dir: str | Path = "results/baseline_seed1",
) -> dict[str, Any]:
    root = Path(project_root)
    manifest_file = _result_path(root, str(manifest_path))
    manifest = _load_json(manifest_file)
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected grid manifest schema_version=1")
    if manifest.get("counts", {}).get("needs_training") != 0:
        raise ValueError("Grid is incomplete: needs_training must be zero")

    baseline = _read_baseline(root, baseline_dir)
    full_parameters = int(baseline["parameters"])
    full_macs = int(baseline["macs_per_image"])
    if full_parameters != int(manifest["full_parameters"]):
        raise ValueError("Baseline and manifest parameter counts differ")

    runs = [
        _read_run(root, item, full_parameters, full_macs)
        for item in manifest["experiments"]
    ]
    points = _logical_points(runs)
    random = _random_summary(points)
    budgets = _budget_summary(points, random)
    return {
        "schema_version": 1,
        "experiment_family": "expanded_one_shot_scratch_grid",
        "manifest": str(manifest_path),
        "protocol": {
            "dataset": "CIFAR-10",
            "model": "ResNet-20",
            "budgets": list(manifest["budgets"]),
            "deterministic_methods": list(manifest["methods"]),
            "train_seed": int(manifest["train_seed"]),
            "random_architecture_seeds": list(
                manifest["random_architecture_seeds_requested"]
            ),
            "physical_runs": len(runs),
            "logical_method_points": len(points),
        },
        "baseline": baseline,
        "runs": runs,
        "method_points": points,
        "random_summary": random,
        "budget_summary": budgets,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "budget",
        "method",
        "physical_method",
        "train_seed",
        "architecture_seed",
        "inner_widths",
        "parameters",
        "parameter_ratio_vs_full",
        "macs_per_image",
        "mac_ratio_vs_full",
        "best_validation_accuracy",
        "test_accuracy",
        "test_loss",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run": row["run"],
                    "budget": row["budget"],
                    "method": row["method"],
                    "physical_method": row["physical_method"],
                    "train_seed": row["train_seed"],
                    "architecture_seed": row["architecture_seed"],
                    "inner_widths": " ".join(map(str, row["inner_widths"])),
                    "parameters": row["parameters"],
                    "parameter_ratio_vs_full": row["parameter_ratio_vs_full"],
                    "macs_per_image": row["macs_per_image"],
                    "mac_ratio_vs_full": row["mac_ratio_vs_full"],
                    "best_validation_accuracy": row["best_validation"]["accuracy"],
                    "test_accuracy": row["test_accuracy"],
                    "test_loss": row["test_loss"],
                }
            )


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Сводка расширенной одношаговой сетки",
        "",
        "Все компактные модели обучены с нуля при `train_seed=1`. Для `random` ",
        "использованы десять разных `architecture_seed` на каждый бюджет.",
        "",
        "## Победители среди детерминированных методов",
        "",
        "| Бюджет | Метод | Test accuracy | Параметры | MACs на изображение | Превышение среднего random |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in report["budget_summary"]:
        lines.append(
            f"| `{item['budget']}` | `{item['winner_method']}` | "
            f"{_format_percent(item['winner_test_accuracy'])} | "
            f"{item['winner_parameters']:,} | {item['winner_macs_per_image']:,} | "
            f"{100.0 * item['winner_minus_random_mean']:.2f} п.п. |"
        )
    lines.extend(
        [
            "",
            "## Случайные архитектуры",
            "",
            "| Бюджет | Число архитектур | Среднее ± отклонение | Минимум | Максимум |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in report["random_summary"]:
        lines.append(
            f"| `{item['budget']}` | {item['count']} | "
            f"{100.0 * item['test_accuracy_mean']:.2f} ± "
            f"{100.0 * item['test_accuracy_stdev']:.2f}% | "
            f"{_format_percent(item['test_accuracy_min'])} | "
            f"{_format_percent(item['test_accuracy_max'])} |"
        )
    lines.extend(
        [
            "",
            "## Контроль полноты",
            "",
            f"- Физических обучений: {report['protocol']['physical_runs']}.",
            f"- Логических точек методов с учётом aliases: "
            f"{report['protocol']['logical_method_points']}.",
            "- Все результаты относятся к официальным 10 000 test-изображениям.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_grid_results(
    report: dict[str, Any],
    output_dir: str | Path,
    report_data: str | Path | None = None,
) -> None:
    destination = Path(output_dir)
    _save_json(report, destination / "summary.json")
    _write_csv(report["method_points"], destination / "method_points.csv")
    _write_markdown(report, destination / "summary.md")
    if report_data is not None:
        _save_json(report, Path(report_data))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect the completed expanded one-shot experiment grid"
    )
    parser.add_argument(
        "--manifest", default="configs/generated_v4/manifest.json"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--baseline-dir", default="results/baseline_seed1")
    parser.add_argument("--output-dir", default="results/summary_v09")
    parser.add_argument(
        "--report-data", default="reports/data/grid_results_v09.json"
    )
    args = parser.parse_args()
    report = collect_grid_results(
        args.manifest,
        project_root=args.project_root,
        baseline_dir=args.baseline_dir,
    )
    write_grid_results(report, args.output_dir, args.report_data)
    print(
        json.dumps(
            {
                "physical_runs": report["protocol"]["physical_runs"],
                "logical_method_points": report["protocol"][
                    "logical_method_points"
                ],
                "budgets": len(report["budget_summary"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
