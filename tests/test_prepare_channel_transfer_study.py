import json

import pytest

torch = pytest.importorskip("torch")

from project_rank.models import ResNet20
from project_rank.prepare_channel_transfer_study import (
    prepare_channel_transfer_study,
)


def test_prepare_channel_transfer_study_builds_two_budget_matrix(tmp_path):
    source = ResNet20()
    checkpoint = {
        "model_state": source.state_dict(),
        "inner_widths": list(source.inner_widths),
        "num_classes": 10,
    }
    report = {
        "budgets": [
            {
                "label": "p060",
                "architectures": [
                    {"name": "erank_per_image", "inner_widths": [8] * 9}
                ],
            },
            {
                "label": "p030",
                "architectures": [
                    {"name": "erank_per_image", "inner_widths": [4] * 9}
                ],
            },
        ]
    }
    covariances = {
        f"block{index + 1}": torch.diag(
            torch.arange(width, 0, -1, dtype=torch.float64)
        )
        for index, width in enumerate(source.inner_widths)
    }
    config_dir = tmp_path / "configs"
    init_dir = tmp_path / "initializations"
    result_dir = tmp_path / "results"
    manifest = prepare_channel_transfer_study(
        {
            "seed": 1,
            "dataset": "synthetic",
            "epochs": 1,
            "inner_widths": list(source.inner_widths),
        },
        report,
        checkpoint,
        covariances,
        config_dir,
        init_dir,
        result_dir,
        budgets=["p060", "p030"],
        fine_tune_epochs=1,
        max_samples=8,
    )

    assert len(manifest["budgets"]) == 2
    assert all(len(item["experiments"]) == 3 for item in manifest["budgets"])
    assert len(list(init_dir.glob("*.pt"))) == 6
    assert len(list(config_dir.glob("*_finetune_seed1.json"))) == 6
    assert config_dir.joinpath("train_commands.sh").is_file()
    selection = json.loads(
        result_dir.joinpath("channel_selection.json").read_text(encoding="utf-8")
    )
    assert set(selection["budgets"]) == {"p060", "p030"}
    assert len(selection["budgets"]["p030"]["erank_subspace"]["selected_channels"]) == 9
