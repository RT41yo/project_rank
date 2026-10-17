import pytest

torch = pytest.importorskip("torch")

from project_rank.architectures import multi_budget_architecture_report
from project_rank.models import ResNet20
from project_rank.prepare_experiments import prepare_experiments


def test_prepare_experiments_deduplicates_equivalent_erank(tmp_path):
    ranks = {
        "mean_per_image_effective_rank": [
            11.73, 14.07, 14.47, 26.65, 28.45, 28.28, 40.06, 40.05, 33.68
        ],
        "pooled_effective_rank": [
            14.04, 14.93, 14.88, 29.01, 30.96, 31.17, 57.79, 61.93, 60.82
        ],
        "weight_effective_rank": [
            14.04, 15.05, 14.56, 28.73, 30.51, 30.65, 57.72, 61.92, 60.30
        ],
    }
    report = multi_budget_architecture_report(ranks, [0.6], random_seed=1001)
    source = ResNet20()
    checkpoint = {
        "model_state": source.state_dict(),
        "inner_widths": list(source.inner_widths),
        "num_classes": 10,
    }
    config_dir = tmp_path / "configs"
    init_dir = tmp_path / "init"
    manifest = prepare_experiments(
        {
            "seed": 1,
            "dataset": "synthetic",
            "epochs": 1,
            "inner_widths": list(source.inner_widths),
        },
        report,
        checkpoint,
        config_dir,
        init_dir,
        fine_tune_epochs=1,
        fine_tune_lr=0.01,
    )
    budget = manifest["budgets"][0]
    assert any(
        set(item["equivalent_names"]) == {"erank_pooled", "erank_weight"}
        for item in budget["aliases"]
    )
    assert len(list(config_dir.glob("*.json"))) == 9  # eight configs + manifest
    assert len(list(init_dir.glob("*.pt"))) == 4
    assert config_dir.joinpath("p060_uniform_finetune.json").is_file()
    assert config_dir.joinpath("p060_random_finetune.json").is_file()
