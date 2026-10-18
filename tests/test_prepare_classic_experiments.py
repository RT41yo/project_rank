from project_rank.architectures import (
    extend_with_pruning_architectures,
    multi_budget_architecture_report,
)
from project_rank.prepare_classic_experiments import prepare_classic_experiments
from project_rank.spec import DEFAULT_INNER_WIDTHS


def test_prepare_classic_experiments_creates_eight_scratch_configs(tmp_path):
    ranks = {
        "mean_per_image_effective_rank": [12, 14, 15, 26, 28, 28, 40, 40, 34],
        "pooled_effective_rank": [14, 15, 15, 29, 31, 31, 58, 62, 61],
        "weight_effective_rank": [14, 15, 15, 29, 31, 31, 58, 62, 60],
    }
    scores = [
        [float(index + 1) for index in range(width)]
        for width in DEFAULT_INNER_WIDTHS
    ]
    report = extend_with_pruning_architectures(
        multi_budget_architecture_report(ranks, [0.8, 0.6], 1001),
        {"l1_weight": scores, "taylor": scores},
    )
    manifest = prepare_classic_experiments(
        {"dataset": "synthetic", "epochs": 160, "seed": 1}, report, tmp_path
    )
    assert len(manifest["experiments"]) == 8
    assert len(list(tmp_path.glob("*.json"))) == 9
    assert all(item["mode"] == "scratch" for item in manifest["experiments"])
