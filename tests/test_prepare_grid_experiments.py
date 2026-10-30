import json

from project_rank.architectures import make_random_architecture
from project_rank.prepare_grid_experiments import prepare_grid_experiments
from project_rank.spec import count_resnet20_parameters


METHOD_WIDTHS = {
    "erank_per_image": [14, 16, 16, 30, 30, 30, 50, 50, 44],
    "erank_pooled": [12, 14, 14, 28, 28, 28, 52, 52, 50],
    "erank_weight": [12, 14, 14, 28, 28, 28, 52, 52, 50],
    "l1_weight": [16, 16, 16, 32, 32, 32, 64, 48, 44],
    "taylor": [16, 16, 16, 32, 32, 30, 62, 48, 44],
    "uniform": [14, 14, 14, 28, 28, 28, 54, 54, 54],
}


def _report():
    full_parameters = count_resnet20_parameters()
    budgets = []
    for label, ratio in (("p090", 0.9), ("p080", 0.8)):
        architectures = []
        for name, widths in METHOD_WIDTHS.items():
            parameters = count_resnet20_parameters(widths)
            architectures.append(
                {
                    "name": name,
                    "inner_widths": widths,
                    "parameters": parameters,
                    "parameter_ratio_vs_full": parameters / full_parameters,
                }
            )
        budgets.append(
            {
                "label": label,
                "requested_parameter_ratio": ratio,
                "reference_parameters": int(round(full_parameters * ratio)),
                "architectures": architectures,
            }
        )
    return {
        "schema_version": 3,
        "full_architecture": {
            "name": "full",
            "inner_widths": [16, 16, 16, 32, 32, 32, 64, 64, 64],
            "parameters": full_parameters,
        },
        "budgets": budgets,
    }


def _write_completed_result(path, widths):
    path.mkdir(parents=True)
    path.joinpath("config.json").write_text(
        json.dumps({"inner_widths": list(widths)}), encoding="utf-8"
    )
    path.joinpath("history.json").write_text(
        json.dumps([{"epoch": 1}]), encoding="utf-8"
    )


def test_prepare_grid_deduplicates_aliases_and_marks_existing_runs(tmp_path):
    results = tmp_path / "results"
    _write_completed_result(
        results / "p080_erank_per_image_scratch_seed1",
        METHOD_WIDTHS["erank_per_image"],
    )
    random_architecture = make_random_architecture(
        int(round(count_resnet20_parameters() * 0.8)), seed=1001
    )
    _write_completed_result(
        results / "p080_random_scratch_seed1",
        random_architecture.inner_widths,
    )

    output = tmp_path / "configs"
    manifest = prepare_grid_experiments(
        {"dataset": "synthetic", "epochs": 160, "seed": 99},
        _report(),
        output,
        results,
        budgets=["p090", "p080"],
        train_seed=1,
        random_architecture_seeds=[1001, 1002, 1003],
    )

    # Five unique deterministic profiles plus three random profiles per budget.
    assert manifest["counts"] == {
        "total": 16,
        "already_complete": 2,
        "needs_training": 14,
    }
    pooled = next(
        item
        for item in manifest["experiments"]
        if item["budget"] == "p080" and item["method"] == "erank_pooled"
    )
    assert pooled["aliases"] == ["erank_pooled", "erank_weight"]

    legacy = next(
        item
        for item in manifest["experiments"]
        if item["budget"] == "p080"
        and item["method"] == "random"
        and item["architecture_seed"] == 1001
    )
    assert legacy["needs_training"] is False
    assert legacy["existing_legacy_run"].endswith(
        "p080_random_scratch_seed1"
    )
    assert output.joinpath("manifest.json").is_file()
    assert output.joinpath("train_commands.txt").is_file()
    assert output.joinpath("test_commands.txt").is_file()


def test_prepare_grid_rejects_duplicate_architecture_seeds(tmp_path):
    try:
        prepare_grid_experiments(
            {"dataset": "synthetic", "epochs": 1},
            _report(),
            tmp_path / "configs",
            tmp_path / "results",
            budgets=["p090"],
            random_architecture_seeds=[1001, 1001],
        )
    except ValueError as error:
        assert "must be unique" in str(error)
    else:
        raise AssertionError("Duplicate architecture seeds must be rejected")
