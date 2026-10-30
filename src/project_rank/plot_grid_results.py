from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHODS = (
    "erank_per_image",
    "erank_pooled",
    "erank_weight",
    "l1_weight",
    "taylor",
    "uniform",
)
COLORS = {
    "baseline": "#424242",
    "erank_per_image": "#1565C0",
    "erank_pooled": "#EF6C00",
    "erank_weight": "#6A1B9A",
    "l1_weight": "#00838F",
    "taylor": "#C62828",
    "uniform": "#2E7D32",
    "random": "#757575",
}


def load_grid_data(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data.get("schema_version") != 1:
        raise ValueError("Ожидалась первая версия сводки расширенной сетки")
    if data.get("experiment_family") != "expanded_one_shot_scratch_grid":
        raise ValueError("Передан файл другого семейства экспериментов")
    return data


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 12.5,
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


def _budget_value(label: str) -> int:
    return int(label.removeprefix("p"))


def _deterministic_map(data: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (item["budget"], item["method"]): item
        for item in data["method_points"]
        if item["method"] != "random"
    }


def _curve_plot(
    data: dict[str, Any],
    output: Path,
    x_key: str,
    random_x_key: str,
    xlabel: str,
    title: str,
) -> None:
    plt = _pyplot()
    points = _deterministic_map(data)
    budgets = sorted(data["protocol"]["budgets"], key=_budget_value)
    baseline = data["baseline"]

    fig, ax = plt.subplots(figsize=(9.4, 5.7))
    for method in METHODS:
        items = [points[(budget, method)] for budget in budgets]
        x = [100.0 * float(item[x_key]) for item in items]
        y = [100.0 * float(item["test_accuracy"]) for item in items]
        x.append(100.0)
        y.append(100.0 * float(baseline["test_accuracy"]))
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            color=COLORS[method],
            label=method,
        )

    random_items = sorted(
        data["random_summary"], key=lambda item: _budget_value(item["budget"])
    )
    random_x = [100.0 * float(item[random_x_key]) for item in random_items]
    random_y = [100.0 * float(item["test_accuracy_mean"]) for item in random_items]
    random_low = [
        100.0 * (
            float(item["test_accuracy_mean"])
            - float(item["test_accuracy_stdev"])
        )
        for item in random_items
    ]
    random_high = [
        100.0 * (
            float(item["test_accuracy_mean"])
            + float(item["test_accuracy_stdev"])
        )
        for item in random_items
    ]
    ax.plot(
        random_x,
        random_y,
        marker="s",
        markersize=4.5,
        linewidth=1.8,
        linestyle="--",
        color=COLORS["random"],
        label="random: среднее",
    )
    ax.fill_between(
        random_x,
        random_low,
        random_high,
        color=COLORS["random"],
        alpha=0.16,
        label="random: ±1 отклонение",
    )
    ax.scatter(
        [100.0],
        [100.0 * float(baseline["test_accuracy"])],
        color=COLORS["baseline"],
        marker="*",
        s=95,
        zorder=6,
        label="baseline",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Точность на test, %")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(frameon=False, ncols=3, fontsize=8.3, loc="lower right")
    _save(fig, output)
    plt.close(fig)


def plot_quality_parameters(data: dict[str, Any], output: Path) -> None:
    _curve_plot(
        data,
        output,
        "parameter_ratio_vs_full",
        "parameter_ratio_mean",
        "Параметры относительно baseline, %",
        "Кривые качество–размер: семь бюджетов",
    )


def plot_quality_macs(data: dict[str, Any], output: Path) -> None:
    _curve_plot(
        data,
        output,
        "mac_ratio_vs_full",
        "mac_ratio_mean",
        "MACs относительно baseline, %",
        "Качество и вычислительная стоимость",
    )


def plot_random_architectures(data: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    random_runs: dict[str, list[float]] = {}
    for item in data["method_points"]:
        if item["method"] == "random":
            random_runs.setdefault(item["budget"], []).append(
                100.0 * float(item["test_accuracy"])
            )
    budgets = sorted(random_runs, key=_budget_value, reverse=True)
    values = [random_runs[budget] for budget in budgets]
    winners = {item["budget"]: item for item in data["budget_summary"]}

    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    boxes = ax.boxplot(
        values,
        tick_labels=budgets,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#4A148C"},
        medianprops={"color": "#4A148C", "linewidth": 1.8},
    )
    for box in boxes["boxes"]:
        box.set(facecolor="#CE93D8", alpha=0.55, edgecolor="#7B1FA2")
    x = list(range(1, len(budgets) + 1))
    winner_y = [100.0 * float(winners[b]["winner_test_accuracy"]) for b in budgets]
    ax.plot(
        x,
        winner_y,
        marker="o",
        color="#C62828",
        linewidth=1.8,
        label="лучший детерминированный метод",
    )
    ax.set_xlabel("Бюджет параметров")
    ax.set_ylabel("Точность на test, %")
    ax.set_title("Разброс десяти random-архитектур на каждом бюджете")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, output)
    plt.close(fig)


def plot_advantage_vs_random(data: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    points = _deterministic_map(data)
    random = {item["budget"]: item for item in data["random_summary"]}
    budgets = sorted(data["protocol"]["budgets"], key=_budget_value, reverse=True)
    matrix = [
        [
            100.0 * (
                float(points[(budget, method)]["test_accuracy"])
                - float(random[budget]["test_accuracy_mean"])
            )
            for budget in budgets
        ]
        for method in METHODS
    ]

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(budgets)), budgets)
    ax.set_yticks(range(len(METHODS)), METHODS)
    ax.set_xlabel("Бюджет параметров")
    ax.set_title("Преимущество над средним random, процентные пункты")
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            ax.text(
                column,
                row,
                f"{value:+.2f}".replace(".", ","),
                ha="center",
                va="center",
                fontsize=8.5,
                color="black",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Разница, п.п.")
    _save(fig, output)
    plt.close(fig)


def plot_width_profiles(data: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    points = _deterministic_map(data)
    budgets = sorted(data["protocol"]["budgets"], key=_budget_value, reverse=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), sharex=True, sharey=True)
    last_image = None
    for ax, method in zip(axes.flat, METHODS, strict=True):
        matrix = [points[(budget, method)]["inner_widths"] for budget in budgets]
        last_image = ax.imshow(matrix, cmap="YlGnBu", vmin=4, vmax=64, aspect="auto")
        ax.set_title(method)
        ax.set_xticks(range(9), [str(index) for index in range(1, 10)])
        ax.set_yticks(range(len(budgets)), budgets)
        for row, values in enumerate(matrix):
            for column, value in enumerate(values):
                ax.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color="white" if value >= 38 else "black",
                )
    fig.supxlabel("Номер остаточного блока")
    fig.supylabel("Бюджет параметров")
    fig.suptitle("Внутренние ширины девяти блоков", fontsize=14)
    if last_image is not None:
        colorbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
        colorbar.set_label("Число каналов")
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.92, hspace=0.25, wspace=0.16)
    _save(fig, output)
    plt.close(fig)


def plot_winners(data: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    items = sorted(data["budget_summary"], key=lambda item: _budget_value(item["budget"]), reverse=True)
    budgets = [item["budget"] for item in items]
    values = [100.0 * float(item["winner_test_accuracy"]) for item in items]
    methods = [item["winner_method"] for item in items]
    colors = [COLORS[method] for method in methods]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    bars = ax.bar(budgets, values, color=colors, width=0.68)
    lower = min(values) - 0.8
    ax.set_ylim(lower, max(values) + 0.8)
    ax.set_ylabel("Точность на test, %")
    ax.set_xlabel("Бюджет параметров")
    ax.set_title("Лучший детерминированный метод меняется с бюджетом")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    for bar, value, method in zip(bars, values, methods, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.12,
            f"{method}\n{value:.2f}%".replace(".", ","),
            ha="center",
            va="bottom",
            fontsize=8.3,
        )
    _save(fig, output)
    plt.close(fig)


def generate_grid_plots(data_path: str | Path, output_dir: str | Path) -> list[Path]:
    data = load_grid_data(data_path)
    destination = Path(output_dir)
    outputs = [
        destination / "quality_parameters_grid.png",
        destination / "quality_macs_grid.png",
        destination / "random_architectures_grid.png",
        destination / "advantage_vs_random_grid.png",
        destination / "width_profiles_grid.png",
        destination / "winners_grid.png",
    ]
    plot_quality_parameters(data, outputs[0])
    plot_quality_macs(data, outputs[1])
    plot_random_architectures(data, outputs[2])
    plot_advantage_vs_random(data, outputs[3])
    plot_width_profiles(data, outputs[4])
    plot_winners(data, outputs[5])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the completed expanded one-shot experiment grid"
    )
    parser.add_argument(
        "--data", default="reports/data/grid_results_v09.json"
    )
    parser.add_argument("--output-dir", default="reports/figures")
    args = parser.parse_args()
    outputs = generate_grid_plots(args.data, args.output_dir)
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
