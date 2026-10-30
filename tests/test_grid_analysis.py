import json

import pytest

from project_rank.grid_analysis import collect_grid_results, write_grid_results
from project_rank.spec import count_resnet20_macs, count_resnet20_parameters


FULL_WIDTHS = [16, 16, 16, 32, 32, 32, 64, 64, 64]


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_result(root, name, widths, accuracy):
    run = root / "results" / name
    _save(run / "config.json", {"inner_widths": widths, "num_classes": 10})
    _save(
        run / "history.json",
        [
            {
                "epoch": 1,
                "validation": {"accuracy": accuracy - 0.02, "loss": 0.5},
            },
            {
                "epoch": 2,
                "validation": {"accuracy": accuracy - 0.01, "loss": 0.4},
            },
        ],
    )
    _save(
        run / "test.json",
        {"split": "test", "accuracy": accuracy, "loss": 0.35, "samples": 10000},
    )


def test_analytical_macs_match_measured_baseline():
    assert count_resnet20_macs() == 40_813_184
    assert count_resnet20_macs() * 128 == 5_224_087_552


def test_grid_analysis_expands_aliases_and_summarizes_random(tmp_path):
    _make_result(tmp_path, "baseline_seed1", FULL_WIDTHS, 0.92)
    widths_a = [12, 12, 12, 24, 24, 24, 48, 48, 48]
    widths_b = [10, 12, 14, 20, 24, 28, 44, 48, 50]
    widths_c = [14, 10, 12, 26, 22, 24, 52, 44, 46]
    _make_result(tmp_path, "p080_erank_pooled_scratch_seed1", widths_a, 0.91)
    _make_result(tmp_path, "p080_random_arch1001_scratch_seed1", widths_b, 0.89)
    _make_result(tmp_path, "p080_random_arch1002_scratch_seed1", widths_c, 0.90)

    full_parameters = count_resnet20_parameters()

    def experiment(name, method, widths, aliases, architecture_seed=None):
        return {
            "run": name,
            "budget": "p080",
            "method": method,
            "method_family": (
                "random_width_rule" if method == "random" else "deterministic_width_rule"
            ),
            "aliases": aliases,
            "mode": "scratch",
            "train_seed": 1,
            "architecture_seed": architecture_seed,
            "parameters": count_resnet20_parameters(widths),
            "inner_widths": widths,
            "result_dir": f"results/{name}",
            "effective_result_dir": f"results/{name}",
        }

    manifest = {
        "schema_version": 1,
        "train_seed": 1,
        "budgets": ["p080"],
        "methods": ["erank_pooled", "erank_weight"],
        "random_architecture_seeds_requested": [1001, 1002],
        "full_parameters": full_parameters,
        "counts": {"total": 3, "already_complete": 3, "needs_training": 0},
        "experiments": [
            experiment(
                "p080_erank_pooled_scratch_seed1",
                "erank_pooled",
                widths_a,
                ["erank_pooled", "erank_weight"],
            ),
            experiment(
                "p080_random_arch1001_scratch_seed1",
                "random",
                widths_b,
                ["random"],
                1001,
            ),
            experiment(
                "p080_random_arch1002_scratch_seed1",
                "random",
                widths_c,
                ["random"],
                1002,
            ),
        ],
    }
    _save(tmp_path / "manifest.json", manifest)

    report = collect_grid_results(
        "manifest.json", project_root=tmp_path, baseline_dir="results/baseline_seed1"
    )

    assert report["protocol"]["physical_runs"] == 3
    assert report["protocol"]["logical_method_points"] == 4
    assert report["random_summary"][0]["count"] == 2
    assert report["random_summary"][0]["test_accuracy_mean"] == pytest.approx(0.895)
    assert report["budget_summary"][0]["winner_method"] in {
        "erank_pooled",
        "erank_weight",
    }

    output = tmp_path / "summary"
    write_grid_results(report, output, tmp_path / "report.json")
    assert (output / "summary.json").is_file()
    assert (output / "method_points.csv").is_file()
    assert (output / "summary.md").is_file()
    assert (tmp_path / "report.json").is_file()
