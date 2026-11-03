import json

import torch

from project_rank.models import ResNet20
from project_rank.prepare_iterative_erank import prepare_iterative_step
from project_rank.spec import DEFAULT_INNER_WIDTHS, count_resnet20_parameters


def _fake_rank_results(widths):
    activations = {}
    weights = {}
    for index, width in enumerate(widths, start=1):
        name = f"block{index}"
        rank = max(2.0, width * (0.50 + index / 100.0))
        activations[name] = {
            "channels": width,
            "images": 100,
            "observations": 1000,
            "mean_per_image_effective_rank": rank,
            "pooled_effective_rank": min(float(width), rank + 1.0),
            "pooled_rank_ratio": min(1.0, (rank + 1.0) / width),
        }
        weights[name] = min(float(width), rank + 0.5)
    return activations, weights


def test_prepare_iterative_step_builds_erank_branches(tmp_path):
    source_widths = [14, 16, 16, 32, 32, 32, 50, 50, 42]
    source_model = ResNet20(source_widths)
    source_checkpoint = {
        "model_state": source_model.state_dict(),
        "inner_widths": source_widths,
        "num_classes": 10,
    }
    activations, weights = _fake_rank_results(source_widths)
    base_config = {
        "seed": 1,
        "epochs": 160,
        "learning_rate": 0.1,
        "output_dir": "unused",
    }
    config_dir = tmp_path / "configs"
    output_dir = tmp_path / "results"

    manifest = prepare_iterative_step(
        base_config=base_config,
        source_checkpoint=source_checkpoint,
        activation_results=activations,
        weight_results=weights,
        source_checkpoint_path="results/p080/best.pt",
        config_dir=config_dir,
        output_dir=output_dir,
        target_param_ratio=0.60,
    )

    target_widths = manifest["target"]["inner_widths"]
    assert manifest["criterion"] == "erank_per_image"
    assert all(target <= source for target, source in zip(target_widths, source_widths))
    full_parameters = count_resnet20_parameters(DEFAULT_INNER_WIDTHS)
    assert abs(manifest["target"]["parameters"] / full_parameters - 0.60) < 0.03

    scratch = json.loads((config_dir / "p060_scratch.json").read_text())
    fine_tune = json.loads((config_dir / "p060_finetune_l1.json").read_text())
    assert "resume_from" not in scratch
    assert scratch["inner_widths"] == target_widths
    assert fine_tune["resume_from"].endswith("l1_initialization.pt")
    assert fine_tune["iterative_search"]["criterion"] == "erank_per_image"
    assert fine_tune["iterative_search"]["channel_transfer"] == "l1"

    initialization = torch.load(
        output_dir / "l1_initialization.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert initialization["inner_widths"] == target_widths
    assert initialization["transfer_method"] == "conv1_l1"
    assert (config_dir / "train_commands.sh").exists()
    assert (config_dir / "test_commands.sh").exists()


def test_iterative_step_rejects_non_decreasing_budget(tmp_path):
    source_model = ResNet20(DEFAULT_INNER_WIDTHS)
    checkpoint = {
        "model_state": source_model.state_dict(),
        "inner_widths": list(DEFAULT_INNER_WIDTHS),
        "num_classes": 10,
    }
    activations, weights = _fake_rank_results(DEFAULT_INNER_WIDTHS)

    try:
        prepare_iterative_step(
            base_config={"seed": 1},
            source_checkpoint=checkpoint,
            activation_results=activations,
            weight_results=weights,
            source_checkpoint_path="baseline.pt",
            config_dir=tmp_path / "configs",
            output_dir=tmp_path / "results",
            target_param_ratio=1.0,
        )
    except ValueError as error:
        assert "target_param_ratio" in str(error)
    else:
        raise AssertionError("Expected a ValueError")
