from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COLORS = {
    "baseline": "#5F6368",
    "erank_per_image": "#1565C0",
    "uniform": "#2E7D32",
    "erank_pooled": "#EF6C00",
    "random": "#7B1FA2",
    "l1_weight": "#00838F",
    "taylor": "#C62828",
}


def load_report_data(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data.get("schema_version") != 2:
        raise ValueError("Ожидалась вторая версия формата итоговых данных")
    return data


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "grid.color": "#DADCE0",
            "grid.alpha": 0.65,
        }
    )
    return plt


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")


def plot_accuracy(data: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    keys = ["baseline", "taylor", "erank_per_image", "l1_weight", "uniform", "erank_pooled"]
    items = {
        "baseline": data["baseline"],
        **data["p060"],
    }
    labels = ["baseline", "taylor", "erank_\nper_image", "l1_weight", "uniform", "erank_\npooled"]
    means = [100.0 * items[key]["test_mean"] for key in keys]
    errors = [100.0 * items[key]["test_stdev"] for key in keys]

    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    bars = ax.bar(
        labels,
        means,
        yerr=errors,
        capsize=5,
        color=[COLORS[key] for key in keys],
        width=0.68,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_ylim(90.0, 93.1)
    ax.set_ylabel("Точность на тестовой выборке, %")
    ax.set_title("Точность при сохранении около 60% параметров")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    for bar, value in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.2f}%".replace(".", ","),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    fig.text(
        0.5,
        -0.01,
        "Столбики ошибок показывают стандартное отклонение по трём запускам.",
        ha="center",
        fontsize=9,
        color="#4A4A4A",
    )
    _save(fig, output)
    plt.close(fig)


def plot_seed_results(data: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    keys = ["baseline", "taylor", "erank_per_image", "l1_weight", "uniform", "erank_pooled"]
    items = {
        "baseline": data["baseline"],
        **data["p060"],
    }
    seeds = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(9.8, 5.1))
    for key in keys:
        values = [100.0 * value for value in items[key]["test_by_seed"]]
        ax.plot(
            seeds,
            values,
            marker="o",
            markersize=6,
            linewidth=2.2,
            color=COLORS[key],
            label=key,
        )
    ax.set_xticks(seeds, ["Запуск 1", "Запуск 2", "Запуск 3"])
    ax.set_ylim(90.3, 92.8)
    ax.set_ylabel("Точность на тестовой выборке, %")
    ax.set_title("Результаты трёх независимых обучений")
    ax.grid(axis="y")
    ax.legend(frameon=False, ncols=3, loc="lower left")
    _save(fig, output)
    plt.close(fig)


def plot_resources(data: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    baseline = data["baseline"]
    keys = ["taylor", "erank_per_image", "l1_weight", "uniform", "erank_pooled"]
    metrics = [
        ("parameters", "Параметры"),
        ("macs_per_batch", "MACs"),
        ("latency_median_ms", "Задержка"),
        ("peak_cuda_memory_bytes", "Память"),
    ]

    def value(item: dict[str, Any], key: str) -> float:
        if key == "parameters":
            return float(item[key])
        return float(item["benchmark"][key])

    base_values = {key: value(baseline, key) for key, _ in metrics}
    x = list(range(len(metrics)))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    for index, key in enumerate(keys):
        item = data["p060"][key]
        ratios = [100.0 * value(item, metric) / base_values[metric] for metric, _ in metrics]
        positions = [position + (index - 2) * width for position in x]
        bars = ax.bar(
            positions,
            ratios,
            width=width,
            color=COLORS[key],
            label=item["label"],
        )
        for bar, ratio in zip(bars, ratios, strict=True):
            offset = 3.2 if ratio >= 96.0 else 1.3
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ratio + offset,
                f"{ratio:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.axhline(100.0, color=COLORS["baseline"], linestyle="--", linewidth=1.5)
    ax.text(3.48, 110.0, "Полная модель: 100%", ha="right", fontsize=8.5)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 116)
    ax.set_ylabel("Доля от полной модели, %")
    ax.set_title("Размер и затраты ресурсов при бюджете 60%")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        ncols=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    _save(fig, output)
    plt.close(fig)


def plot_widths(data: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    keys = ["baseline", "taylor", "erank_per_image", "l1_weight", "uniform", "erank_pooled"]
    items = {
        "baseline": data["baseline"],
        **data["p060"],
    }
    blocks = list(range(1, 10))
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.axvspan(0.5, 3.5, color="#E3F2FD", alpha=0.55)
    ax.axvspan(3.5, 6.5, color="#E8F5E9", alpha=0.55)
    ax.axvspan(6.5, 9.5, color="#FFF3E0", alpha=0.55)
    for key in keys:
        ax.plot(
            blocks,
            items[key]["widths"],
            marker="o",
            markersize=5,
            linewidth=2.2,
            color=COLORS[key],
            label=key,
        )
    ax.text(2, 67, "Группа 1", ha="center", fontsize=9)
    ax.text(5, 67, "Группа 2", ha="center", fontsize=9)
    ax.text(8, 67, "Группа 3", ha="center", fontsize=9)
    ax.set_xlim(0.7, 9.3)
    ax.set_ylim(0, 72)
    ax.set_xticks(blocks)
    ax.set_xlabel("Номер остаточного блока")
    ax.set_ylabel("Внутренняя ширина блока")
    ax.set_title("Как разные методы распределяют каналы между блоками")
    ax.grid(axis="y")
    ax.legend(
        frameon=False,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
    )
    _save(fig, output)
    plt.close(fig)


def _seed1_accuracy(item: dict[str, Any]) -> float:
    if "test_seed1" in item:
        return float(item["test_seed1"])
    values = item.get("test_by_seed")
    if values:
        return float(values[0])
    raise ValueError("Нет результата test для seed1")


def plot_quality_size(data: dict[str, Any], output: Path) -> None:
    """Plot the seed1 quality-size curves available in the report data."""
    plt = _pyplot()
    baseline = data["baseline"]
    baseline_parameters = float(baseline["parameters"])
    baseline_accuracy = _seed1_accuracy(baseline) * 100.0
    methods = [
        name for name in data["p060"]
        if name in data["p080"]
    ]
    preferred_order = [
        "erank_per_image",
        "erank_pooled",
        "l1_weight",
        "taylor",
        "uniform",
        "random",
    ]
    methods.sort(
        key=lambda name: (
            preferred_order.index(name) if name in preferred_order else 999,
            name,
        )
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for method in methods:
        points = [data["p060"][method], data["p080"][method]]
        x = [100.0 * float(item["parameters"]) / baseline_parameters for item in points]
        y = [100.0 * _seed1_accuracy(item) for item in points]
        x.append(100.0)
        y.append(baseline_accuracy)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=5.5,
            linewidth=2.0,
            color=COLORS.get(method),
            label=method,
        )
    ax.scatter(
        [100.0], [baseline_accuracy], color=COLORS["baseline"], s=55,
        zorder=5, label="baseline",
    )
    ax.set_xlabel("Параметры относительно baseline, %")
    ax.set_ylabel("Точность на test, %")
    ax.set_title("Кривые качество–размер для seed1")
    ax.set_xlim(56, 103)
    ax.grid(True)
    ax.legend(frameon=False, ncols=2)
    _save(fig, output)
    plt.close(fig)


def plot_layer_sensitivity(data: dict[str, Any], output: Path) -> None:
    """Show measured block sensitivity and criterion correlations."""
    plt = _pyplot()
    sensitivity = data["layer_sensitivity"]
    blocks = sensitivity["blocks"]
    correlations = sensitivity["spearman_vs_accuracy_drop"]

    fig, (ax_drop, ax_corr) = plt.subplots(1, 2, figsize=(11.2, 4.8))
    names = [item["name"] for item in blocks]
    drops = [100.0 * float(item["accuracy_drop"]) for item in blocks]
    colors = ["#90CAF9" if value < 2.0 else "#EF5350" for value in drops]
    bars = ax_drop.bar(names, drops, color=colors)
    ax_drop.set_ylabel("Падение validation accuracy, п.п.")
    ax_drop.set_title("Сужение одного блока на 25%")
    ax_drop.tick_params(axis="x", rotation=45)
    ax_drop.grid(axis="y")
    ax_drop.set_axisbelow(True)
    for bar, value in zip(bars, drops, strict=True):
        ax_drop.text(bar.get_x() + bar.get_width() / 2, value + 0.08,
                     f"{value:.2f}".replace(".", ","), ha="center", fontsize=8)

    criterion_names = list(correlations)
    values = [float(correlations[name]) for name in criterion_names]
    corr_colors = [COLORS.get(name, "#5F6368") for name in criterion_names]
    bars = ax_corr.barh(criterion_names, values, color=corr_colors)
    ax_corr.axvline(0.0, color="#5F6368", linewidth=1)
    ax_corr.set_xlim(-0.8, 0.8)
    ax_corr.set_xlabel("Корреляция Спирмена")
    ax_corr.set_title("Связь критерия с падением accuracy")
    ax_corr.grid(axis="x")
    ax_corr.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        offset = 0.025 if value >= 0 else -0.025
        ax_corr.text(value + offset, bar.get_y() + bar.get_height() / 2,
                     f"{value:+.2f}".replace(".", ","),
                     ha="left" if value >= 0 else "right", va="center", fontsize=9)
    fig.suptitle("Чувствительность блоков без дообучения", fontsize=14)
    fig.tight_layout()
    _save(fig, output)
    plt.close(fig)


def generate_plots(data_path: str | Path, output_dir: str | Path) -> list[Path]:
    data = load_report_data(data_path)
    destination = Path(output_dir)
    outputs = [
        destination / "accuracy_p060.png",
        destination / "accuracy_by_seed.png",
        destination / "resources_p060.png",
        destination / "width_profiles_p060.png",
        destination / "quality_size_seed1.png",
        destination / "layer_sensitivity.png",
    ]
    plot_accuracy(data, outputs[0])
    plot_seed_results(data, outputs[1])
    plot_resources(data, outputs[2])
    plot_widths(data, outputs[3])
    plot_quality_size(data, outputs[4])
    plot_layer_sensitivity(data, outputs[5])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Построить итоговые графики проекта"
    )
    parser.add_argument(
        "--data",
        default="reports/data/final_results.json",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/figures",
    )
    args = parser.parse_args()
    for path in generate_plots(args.data, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
