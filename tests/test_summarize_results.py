import json

import pytest

from project_rank.summarize_results import collect_results, write_report
from project_rank.utils import save_json


def _make_run(root, name, seed, accuracy, test_accuracy):
    run = root / name
    save_json(
        {
            "seed": seed,
            "inner_widths": [4] * 9,
            "num_classes": 10,
        },
        run / "config.json",
    )
    save_json(
        [
            {
                "epoch": 1,
                "validation": {"accuracy": accuracy - 0.01, "loss": 0.5},
            },
            {
                "epoch": 2,
                "validation": {"accuracy": accuracy, "loss": 0.4},
            },
        ],
        run / "history.json",
    )
    save_json(
        {"split": "test", "accuracy": test_accuracy, "loss": 0.45},
        run / "test.json",
    )


def test_summary_aggregates_multiple_seeds(tmp_path):
    results = tmp_path / "results"
    _make_run(
        results,
        "p060_erank_per_image_scratch_seed2",
        2,
        0.90,
        0.89,
    )
    _make_run(
        results,
        "p060_erank_per_image_scratch_seed3",
        3,
        0.92,
        0.91,
    )

    report = collect_results(results)
    aggregate = report["aggregates"][0]

    assert len(report["runs"]) == 2
    assert aggregate["seeds"] == [2, 3]
    assert aggregate["best_validation_accuracy_mean"] == pytest.approx(0.91)
    assert aggregate["test_accuracy_mean"] == pytest.approx(0.90)
    assert aggregate["best_validation_accuracy_stdev"] is not None

    output = tmp_path / "summary"
    write_report(report, output)
    assert json.loads((output / "summary.json").read_text())["schema_version"] == 1
    assert (output / "runs.csv").is_file()
    assert "Средние значения" in (output / "summary.md").read_text()
