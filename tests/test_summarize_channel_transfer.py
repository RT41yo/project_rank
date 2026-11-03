import json

import pytest

from project_rank.summarize_channel_transfer import summarize_channel_transfer


def _write_accuracy(path, accuracy):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"accuracy": accuracy}), encoding="utf-8")


def test_summarize_channel_transfer_calculates_gains(tmp_path):
    scratch = tmp_path / "scratch.json"
    initial = tmp_path / "initial.json"
    final = tmp_path / "final.json"
    _write_accuracy(scratch, 0.90)
    _write_accuracy(initial, 0.70)
    _write_accuracy(final, 0.91)
    manifest = {
        "architecture": "erank_per_image",
        "train_seed": 1,
        "selection_seed": 2001,
        "budgets": [
            {
                "label": "p060",
                "scratch_test": str(scratch),
                "experiments": [
                    {
                        "method": "random",
                        "initial_test": str(initial),
                        "final_test": str(final),
                    }
                ],
            }
        ],
    }

    report = summarize_channel_transfer(manifest)
    row = report["rows"][0]
    assert row["fine_tune_gain_pp"] == pytest.approx(21.0)
    assert row["delta_vs_scratch_pp"] == pytest.approx(1.0)
