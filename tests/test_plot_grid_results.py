import pytest

pytest.importorskip("matplotlib")

from project_rank.plot_grid_results import generate_grid_plots, load_grid_data


def test_grid_report_data_and_plots(tmp_path):
    data_path = "reports/data/grid_results_v09.json"
    data = load_grid_data(data_path)

    assert data["protocol"]["physical_runs"] == 106
    assert data["protocol"]["logical_method_points"] == 112
    assert len(data["budget_summary"]) == 7
    assert all(item["count"] == 10 for item in data["random_summary"])

    outputs = generate_grid_plots(data_path, tmp_path)

    assert len(outputs) == 6
    assert all(path.is_file() and path.stat().st_size > 5_000 for path in outputs)
