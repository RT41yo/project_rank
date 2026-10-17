import pytest

pytest.importorskip("torch")

from project_rank.evaluate import evaluate_checkpoint
from project_rank.train import train_from_config


def test_evaluate_checkpoint_on_separate_synthetic_test_split(tmp_path):
    output_dir = tmp_path / "run"
    config = {
        "seed": 7,
        "dataset": "synthetic",
        "output_dir": str(output_dir),
        "num_classes": 10,
        "synthetic_train_size": 8,
        "synthetic_val_size": 4,
        "synthetic_test_size": 5,
        "batch_size": 4,
        "num_workers": 0,
        "epochs": 1,
        "learning_rate": 0.01,
        "scheduler": "none",
        "amp": False,
        "device": "cpu",
        "inner_widths": [4] * 9,
    }
    train_from_config(config)

    result = evaluate_checkpoint(config, output_dir / "best.pt", split="test")

    assert result["split"] == "test"
    assert result["samples"] == 5.0
    assert 0.0 <= result["accuracy"] <= 1.0
