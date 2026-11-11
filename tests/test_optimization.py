import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from project_rank.models import ViTTinyCIFAR
from project_rank.optimization import build_optimizer, build_scheduler
from project_rank.train import train_from_config
from project_rank.vit_spec import ViTSpec


def test_legacy_recipe_preserves_sgd_and_cosine():
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(model, {})
    scheduler = build_scheduler(optimizer, {}, epochs=4)
    assert isinstance(optimizer, torch.optim.SGD)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert optimizer.defaults["momentum"] == .9
    assert optimizer.defaults["nesterov"] is True
    rates = []
    for _ in range(4):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates == pytest.approx([.1, .08535533905932738, .05, .014644660940672627])


def test_adamw_parameter_groups():
    model = ViTTinyCIFAR(spec=ViTSpec(embed_dim=12, depth=1))
    optimizer = build_optimizer(model, {
        "optimizer": "adamw", "learning_rate": .0003, "weight_decay": .05,
        "exclude_bias_norm_tokens_from_weight_decay": True,
    })
    assert isinstance(optimizer, torch.optim.AdamW)
    groups = {group["weight_decay"]: {id(p) for p in group["params"]} for group in optimizer.param_groups}
    assert groups[.05].isdisjoint(groups[0.])
    assert groups[.05] | groups[0.] == {id(p) for p in model.parameters()}
    for name, parameter in model.named_parameters():
        expected = 0. if parameter.ndim <= 1 or name in {"cls_token", "pos_embed"} else .05
        assert id(parameter) in groups[expected]


def test_warmup_cosine_endpoints():
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=.0003)
    scheduler = build_scheduler(optimizer, {
        "warmup_epochs": 5, "warmup_start_factor": .01,
    }, epochs=160)
    rates = []
    for _ in range(160):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] == pytest.approx(3e-6)
    assert rates[5] == pytest.approx(3e-4)
    assert rates[:6] == sorted(rates[:6])
    assert rates[6:] == sorted(rates[6:], reverse=True)
    assert rates[-1] > 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.)


@pytest.mark.parametrize("config", [
    {"warmup_epochs": -1}, {"warmup_epochs": 4}, {"warmup_epochs": 1.5},
    {"warmup_epochs": 1, "scheduler": "none"},
    {"warmup_epochs": 1, "warmup_start_factor": 0}, {"scheduler": "unknown"},
])
def test_invalid_schedule(config):
    optimizer = torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=.1)
    with pytest.raises(ValueError):
        build_scheduler(optimizer, config, epochs=4)


def test_pilot_only_changes_training_recipe():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "configs/vit/baseline.json").read_text())
    new = json.loads((root / "configs/vit/baseline_adamw_pilot.json").read_text())
    for key in ("model_spec", "seed", "dataset", "data_dir", "num_classes", "val_size",
                "batch_size", "num_workers", "epochs", "rank_checkpoint_epochs",
                "amp", "device", "inner_widths"):
        assert new[key] == old[key]
    assert new["output_dir"] != old["output_dir"]
    assert "resume_from" not in new


def test_adamw_warmup_synthetic_training(tmp_path):
    result = train_from_config({
        "model_spec": {"type": "vit_tiny_cifar", "embed_dim": 12, "depth": 1},
        "inner_widths": [8], "dataset": "synthetic", "device": "cpu", "amp": False,
        "synthetic_train_size": 4, "synthetic_val_size": 3, "batch_size": 2,
        "num_workers": 0, "epochs": 3, "optimizer": "adamw", "learning_rate": .0003,
        "weight_decay": .05, "exclude_bias_norm_tokens_from_weight_decay": True,
        "warmup_epochs": 1, "warmup_start_factor": .01, "output_dir": str(tmp_path),
    })
    history = result["history"]
    assert len(history) == 3
    assert history[0]["learning_rate_used"] == pytest.approx(3e-6)
    assert history[1]["learning_rate_used"] == pytest.approx(3e-4)
    assert all(torch.isfinite(torch.tensor(row["train"]["loss"])) for row in history)
