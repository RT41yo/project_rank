from project_rank.prepare_repeats import prepare_repeat_configs
from project_rank.utils import load_json, save_json


def test_prepare_repeat_configs_creates_fixed_architectures(tmp_path):
    source_dir = tmp_path / "configs" / "generated_v2"
    source_dir.mkdir(parents=True)
    source_config = source_dir / "p060_erank_per_image_scratch.json"
    save_json(
        {
            "seed": 1,
            "dataset": "synthetic",
            "output_dir": "results/old",
            "epochs": 1,
            "inner_widths": [4] * 9,
        },
        source_config,
    )
    manifest = {
        "seed": 1,
        "budgets": [
            {
                "label": "p060",
                "experiments": [
                    {
                        "name": "erank_per_image",
                        "aliases": ["erank_per_image"],
                        "mode": "scratch",
                        "config": "configs/generated_v2/p060_erank_per_image_scratch.json",
                        "parameters": 123,
                    }
                ],
            }
        ],
    }
    output_dir = tmp_path / "configs" / "repeats"

    result = prepare_repeat_configs(
        {"seed": 1, "dataset": "synthetic", "inner_widths": [8] * 9},
        manifest,
        project_root=tmp_path,
        output_dir=output_dir,
        budget_label="p060",
        methods=["erank_per_image"],
        seeds=[2, 3],
    )

    assert len(result["experiments"]) == 4
    assert result["fixed_architectures"] is True
    compact = load_json(output_dir / "p060_erank_per_image_scratch_seed3.json")
    assert compact["seed"] == 3
    assert compact["inner_widths"] == [4] * 9
    assert compact["output_dir"] == "results/p060_erank_per_image_scratch_seed3"
    baseline = load_json(output_dir / "baseline_seed2.json")
    assert baseline["output_dir"] == "results/baseline_seed2"
