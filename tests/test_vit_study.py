import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from project_rank.models import ViTTinyCIFAR
from project_rank.models.factory import model_from_checkpoint
from project_rank.prepare_vit_p060 import config_sha256, prepare_study
from project_rank.run_vit_study import run_complete, run_study
from project_rank.utils import load_json
from project_rank.vit_spec import ViTSpec, count_vit_parameters


@pytest.fixture
def study_inputs(tmp_path):
    spec = ViTSpec(embed_dim=12, depth=2)
    model = ViTTinyCIFAR(spec=spec)
    config = {
        "model_spec": spec.to_dict(), "inner_widths": list(spec.default_widths),
        "num_classes": 10, "protocol_id": "synthetic_pilot", "dataset": "synthetic",
        "seed": 1, "device": "cpu", "amp": False, "optimizer": "adamw",
        "learning_rate": .0003, "weight_decay": .05, "exclude_bias_norm_tokens_from_weight_decay": True,
        "scheduler": "cosine", "warmup_epochs": 0, "epochs": 1,
        "batch_size": 2, "num_workers": 0, "synthetic_train_size": 4, "synthetic_val_size": 3,
        "rank_checkpoint_epochs": [0, 1], "output_dir": str(tmp_path / "baseline"),
    }
    checkpoint = tmp_path / "source.pt"
    torch.save({"config": config, "model_state": model.state_dict(), "model_spec": spec.to_dict(),
                "inner_widths": list(spec.default_widths), "num_classes": 10,
                "validation_accuracy": .5, "epoch": 1}, checkpoint)
    architectures = {
        "model_spec": spec.to_dict(), "full_parameters": count_vit_parameters(spec=spec),
        "analysis_metadata": {"checkpoint": str(checkpoint), "samples": 3},
        "budgets": [{"label": "p060", "architectures": [
            {"name": "erank_per_image", "inner_widths": [8, 10], "parameters": count_vit_parameters([8, 10], spec)},
            {"name": "random", "architecture_seed": 1001, "inner_widths": [6, 12],
             "parameters": count_vit_parameters([6, 12], spec)},
        ]}],
    }
    covariance = {f"block{i + 1}": torch.diag(torch.arange(48, dtype=torch.float64)) for i in range(2)}
    return config, architectures, checkpoint, covariance


def prepare(tmp_path, inputs):
    return prepare_study(*inputs, tmp_path / "configs", tmp_path / "results",
                         fine_tune_epochs=1, fine_tune_warmup_epochs=0, calibration_samples=3)


def test_preparation_freezes_five_equal_budget_runs(tmp_path, study_inputs):
    manifest = prepare(tmp_path, study_inputs)
    runs = manifest["experiments"]
    assert len(runs) == 5
    assert len({r["parameters"] for r in runs}) == 1
    assert len({r["macs_per_image"] for r in runs}) == 1
    ft = [r for r in runs if r["mode"] == "finetune"]
    assert {r["transfer_method"] for r in ft} == {"l1", "random", "erank_subspace"}
    for r in runs:
        config = load_json(r["config"])
        assert config_sha256(config) == r["config_sha256"]
        assert config["seed"] == 1 and not config.get("overwrite")
        if r["mode"] == "scratch":
            assert "resume_from" not in config
            assert config["learning_rate"] == .0003
        else:
            assert config["learning_rate"] == pytest.approx(.00003)
            assert config["rank_checkpoint_epochs"] == []
            checkpoint = torch.load(r["initial_checkpoint"], weights_only=False)
            model = model_from_checkpoint(checkpoint)
            assert list(model.inner_widths) == [8, 10]
    with pytest.raises(FileExistsError):
        prepare(tmp_path, study_inputs)


def test_prepare_rejects_parameter_mismatch_before_writes(tmp_path, study_inputs):
    study_inputs[1]["budgets"][0]["architectures"][1]["parameters"] += 1
    with pytest.raises(ValueError, match="parameter count"):
        prepare(tmp_path, study_inputs)
    assert not (tmp_path / "configs").exists()


def test_partial_run_is_not_complete(tmp_path):
    config = {"output_dir": str(tmp_path), "epochs": 160}
    assert not run_complete(config)
    (tmp_path / "history.json").write_text('[{"epoch": 1}]')
    with pytest.raises(RuntimeError, match="Partial run"):
        run_complete(config)


def test_synthetic_end_to_end_and_reuse(tmp_path, study_inputs):
    manifest = prepare(tmp_path, study_inputs)
    path = tmp_path / "configs" / "manifest.json"
    summary = run_study(path)
    assert not summary["test_evaluated"]
    assert len(summary["experiments"]) == 5
    assert all(e["epochs_completed"] == 1 for e in summary["experiments"])
    assert all(e["initial_validation"]["split"] == "validation"
               for e in summary["experiments"] if e["mode"] == "finetune")
    for e in manifest["experiments"]:
        assert run_complete(load_json(e["config"]))
        assert not (Path(e["result_dir"]) / "test.json").exists()
    before = {e["name"]: (Path(e["result_dir"]) / "best.pt").stat().st_mtime_ns
              for e in manifest["experiments"]}
    repeated = run_study(path)
    assert [e["validation_accuracy"] for e in repeated["experiments"]] == [
        e["validation_accuracy"] for e in summary["experiments"]]
    assert before == {e["name"]: (Path(e["result_dir"]) / "best.pt").stat().st_mtime_ns
                      for e in manifest["experiments"]}
    config_path = Path(manifest["experiments"][0]["config"])
    changed = load_json(config_path)
    changed["learning_rate"] = .1
    config_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Frozen configuration"):
        run_study(path)
