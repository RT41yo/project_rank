import pytest

torch = pytest.importorskip("torch")

from project_rank.benchmark import count_macs
from project_rank.effective_rank import (
    ActivationRankCollector, collect_activation_ranks, collect_weight_ranks,
    effective_rank_torch,
)
from project_rank.evaluate import evaluate_checkpoint
from project_rank.models import ResNet20, ViTTinyCIFAR
from project_rank.models.factory import build_model, model_from_checkpoint, model_specification
from project_rank.train import train_from_config
from project_rank.vit_spec import ViTSpec, count_vit_macs, count_vit_parameters
from project_rank.vit_transfer import select_vit_neurons, transfer_vit_neurons


@pytest.fixture
def small_spec():
    return ViTSpec(embed_dim=12, depth=2, num_heads=3, num_classes=10)


def test_full_vit_counts():
    model = ViTTinyCIFAR()
    assert sum(p.numel() for p in model.parameters()) == count_vit_parameters() == 5_362_762
    assert count_vit_parameters([4] * 12) == 1_833_082
    assert count_vit_parameters([4] * 12) > 0.30 * count_vit_parameters()
    assert count_vit_macs() == 365_107_584
    model.eval()
    assert count_macs(model, torch.randn(1, 3, 32, 32)) == count_vit_macs()


def test_narrow_forward_backward_and_independent_mac_count(small_spec):
    model = ViTTinyCIFAR([8, 14], small_spec)
    images = torch.randn(2, 3, 32, 32)
    logits = model(images)
    assert logits.shape == (2, 10)
    logits.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == count_vit_parameters([8, 14], small_spec)
    assert count_macs(model.eval(), images) == 2 * count_vit_macs([8, 14], small_spec)
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 28, 28))


@pytest.mark.parametrize("kwargs", [
    {"embed_dim": 13}, {"depth": 0}, {"patch_size": 7}, {"embed_dim": 12.5},
])
def test_invalid_specs(kwargs):
    with pytest.raises(ValueError):
        ViTSpec(**kwargs)


def test_factory_legacy_and_vit(small_spec):
    old = ResNet20([4] * 9)
    restored = model_from_checkpoint({"inner_widths": [4] * 9, "model_state": old.state_dict()})
    assert isinstance(restored, ResNet20)
    model = build_model({"model_spec": small_spec.to_dict(), "inner_widths": [8, 10]})
    restored = model_from_checkpoint({
        "model_spec": model_specification(model), "inner_widths": [8, 10],
        "num_classes": 10, "model_state": model.state_dict(),
    })
    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(model(x), restored(x))
    with pytest.raises(ValueError):
        build_model({"model_spec": {"type": "unknown"}})


def test_streamed_token_ranks_match_direct_and_exclude_cls():
    torch.manual_seed(32)
    tokens = torch.randn(5, 7, 3, dtype=torch.float64)
    tokens[:, 0] = 1e6  # Must never enter the moments/SVD.
    module = torch.nn.Identity()
    with ActivationRankCollector({"block1": module}, {"block1": 3}, exclude_cls=True) as collector:
        module(tokens[:2])
        module(tokens[2:])
        result = collector.results()["block1"]
    expected = float(effective_rank_torch(tokens[:, 1:]).mean())
    pooled = float(effective_rank_torch(tokens[:, 1:].reshape(-1, 3)))
    assert result["mean_per_image_effective_rank"] == pytest.approx(expected, abs=1e-10)
    assert result["pooled_effective_rank"] == pytest.approx(pooled, abs=1e-10)
    assert result["observations"] == 30
    assert result["images"] == 5
    assert not module._forward_hooks


def test_rank_collectors_leave_weights_unchanged(small_spec):
    from torch.utils.data import DataLoader, TensorDataset

    model = ViTTinyCIFAR([8, 14], small_spec)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    loader = DataLoader(TensorDataset(torch.randn(4, 3, 32, 32), torch.zeros(4, dtype=torch.long)), batch_size=2)
    ranks = collect_activation_ranks(model, loader, torch.device("cpu"), max_samples=3)
    weights = collect_weight_ranks(model)
    assert list(ranks) == ["block1", "block2"]
    assert ranks["block1"]["images"] == 3
    assert ranks["block1"]["observations"] == 3 * 64
    assert 0 < weights["block2"] <= 12 + 1e-8
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])


@pytest.mark.parametrize("method", ["random", "l1"])
def test_transfer_matches_masked_source(small_spec, method):
    torch.manual_seed(5)
    source = ViTTinyCIFAR([18, 24], small_spec).eval()
    target = ViTTinyCIFAR([8, 10], small_spec).eval()
    selected = select_vit_neurons(source, target.inner_widths, method=method)
    transfer_vit_neurons(source, target, selected)
    handles = []
    for i, block in enumerate(source.blocks):
        mask = torch.zeros(source.inner_widths[i])
        mask[selected[f"block{i + 1}"]] = 1
        handles.append(block.act.register_forward_hook(lambda m, args, output, mask=mask: output * mask))
    try:
        images = torch.randn(2, 3, 32, 32)
        torch.testing.assert_close(target(images), source(images), atol=2e-6, rtol=2e-5)
    finally:
        for handle in handles:
            handle.remove()


def test_transfer_full_random_reproducibility_and_invalid_indices(small_spec):
    source = ViTTinyCIFAR([18, 24], small_spec).eval()
    target = ViTTinyCIFAR([18, 24], small_spec).eval()
    selected = select_vit_neurons(source, target.inner_widths)
    transfer_vit_neurons(source, target, selected)
    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(source(x), target(x), atol=0, rtol=0)
    a = select_vit_neurons(source, [8, 10], method="random", seed=2001)
    assert a == select_vit_neurons(source, [8, 10], method="random", seed=2001)
    assert a != select_vit_neurons(source, [8, 10], method="random", seed=2002)
    with pytest.raises(ValueError):
        transfer_vit_neurons(source, target, {"block1": [0], "block2": [0]})


def test_subspace_matches_leading_axes(small_spec):
    model = ViTTinyCIFAR([4, 4], small_spec)
    covariance = torch.diag(torch.tensor([9., 8., 1., .1]))
    selected = select_vit_neurons(model, [2, 2], "erank_subspace", covariances={
        "block1": covariance, "block2": covariance,
    })
    assert selected == {"block1": [0, 1], "block2": [0, 1]}


def test_vit_taylor_and_covariances_do_not_update_weights(small_spec):
    from torch.utils.data import DataLoader, TensorDataset
    from project_rank.pruning_criteria import analyze_pruning_criteria
    from project_rank.transfer import collect_activation_covariances

    model = ViTTinyCIFAR([8, 10], small_spec)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    images = torch.randn(3, 3, 32, 32)
    loader = DataLoader(TensorDataset(images, torch.tensor([0, 1, 2])), batch_size=2)
    result = analyze_pruning_criteria(model, loader, torch.device("cpu"), max_samples=3)
    assert result["samples"] == 3
    assert len(result["criteria"]["taylor"]["block2"]["raw_scores"]) == 10
    assert sum(result["criteria"]["taylor"]["block1"]["raw_scores"]) > 0
    covariance = collect_activation_covariances(model, loader, torch.device("cpu"), max_samples=3)
    captured = []
    handle = model.blocks[0].act.register_forward_hook(lambda m, a, out: captured.append(out.detach()))
    with torch.no_grad():
        model(images)
    handle.remove()
    observations = captured[0][:, 1:].double().reshape(-1, 8)
    torch.testing.assert_close(covariance["block1"], torch.cov(observations.T), atol=1e-10, rtol=1e-5)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_vit_synthetic_training_evaluation_and_finetune(tmp_path, small_spec):
    config = {
        "model_spec": small_spec.to_dict(), "inner_widths": [8, 10],
        "seed": 1, "dataset": "synthetic", "device": "cpu", "amp": False,
        "num_workers": 0, "batch_size": 2, "epochs": 1,
        "synthetic_train_size": 4, "synthetic_val_size": 3, "synthetic_test_size": 5,
        "learning_rate": .01, "output_dir": str(tmp_path / "baseline"),
        "rank_checkpoint_epochs": [0, 1],
    }
    result = train_from_config(config)
    checkpoint = torch.load(result["checkpoint"], weights_only=False)
    assert (tmp_path / "baseline" / "epoch_000.pt").is_file()
    assert (tmp_path / "baseline" / "epoch_001.pt").is_file()
    with pytest.raises(FileExistsError):
        train_from_config(config)
    assert checkpoint["model_spec"] == small_spec.to_dict()
    metrics = evaluate_checkpoint(config, result["checkpoint"], split="test")
    assert metrics["samples"] == 5
    source = model_from_checkpoint(checkpoint)
    target = ViTTinyCIFAR([4, 6], small_spec)
    transfer_vit_neurons(source, target, select_vit_neurons(source, [4, 6]))
    init = tmp_path / "init.pt"
    torch.save({"model_state": target.state_dict(), "model_spec": small_spec.to_dict(),
                "inner_widths": [4, 6], "num_classes": 10}, init)
    config.update(inner_widths=[4, 6], resume_from=str(init), output_dir=str(tmp_path / "ft"))
    ft = train_from_config(config)
    assert len(ft["history"]) == 1
    assert evaluate_checkpoint(config, ft["checkpoint"])["samples"] == 3
