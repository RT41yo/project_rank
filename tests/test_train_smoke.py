import pytest

pytest.importorskip("torch")

from project_rank.train import train_from_config


def test_one_epoch_synthetic_training_creates_checkpoint(tmp_path):
    output_dir = tmp_path / "smoke"
    result = train_from_config(
        {
            "seed": 3,
            "dataset": "synthetic",
            "output_dir": str(output_dir),
            "num_classes": 10,
            "synthetic_train_size": 8,
            "synthetic_val_size": 4,
            "batch_size": 4,
            "num_workers": 0,
            "epochs": 1,
            "learning_rate": 0.01,
            "scheduler": "none",
            "amp": False,
            "device": "cpu",
            "inner_widths": [4] * 9,
        }
    )
    assert output_dir.joinpath("best.pt").is_file()
    assert len(result["history"]) == 1

