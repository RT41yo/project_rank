import json

import pytest

pytest.importorskip("matplotlib")

from project_rank.plot_results import generate_plots, load_report_data


def test_final_report_data_and_plots(tmp_path):
    data_path = "reports/data/final_results.json"
    data = load_report_data(data_path)

    assert data["dataset"] == "CIFAR-10"
    assert len(data["baseline"]["test_by_seed"]) == 3
    assert data["p060"]["taylor"]["test_mean"] > data["p060"]["erank_per_image"]["test_mean"]
    assert data["layer_sensitivity"]["spearman_vs_accuracy_drop"]["taylor"] > 0

    outputs = generate_plots(data_path, tmp_path)

    assert len(outputs) == 6
    assert all(path.is_file() and path.stat().st_size > 5_000 for path in outputs)
    with open(data_path, "r", encoding="utf-8") as file:
        assert json.load(file)["schema_version"] == 2
