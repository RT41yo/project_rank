from project_rank.architectures import (
    architecture_report,
    multi_budget_architecture_report,
)
from project_rank.spec import DEFAULT_INNER_WIDTHS, count_resnet20_parameters


def test_parameter_counter_rejects_wrong_number_of_widths():
    try:
        count_resnet20_parameters([16, 16])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_architectures_respect_width_constraints_and_match_budget():
    ranks = [12.0, 10.0, 8.0, 25.0, 20.0, 18.0, 50.0, 40.0, 32.0]
    report = architecture_report(ranks, target_ratio=0.60, random_seed=17)
    items = {item["name"]: item for item in report["architectures"]}
    assert set(items) == {"full", "erank", "uniform", "random"}
    for name in ("erank", "uniform", "random"):
        widths = items[name]["inner_widths"]
        assert len(widths) == 9
        assert all(4 <= value <= maximum for value, maximum in zip(widths, DEFAULT_INNER_WIDTHS))
        assert all(value % 2 == 0 for value in widths)
    erank_parameters = items["erank"]["parameters"]
    random_parameters = items["random"]["parameters"]
    assert abs(random_parameters - erank_parameters) / erank_parameters <= 0.01


def test_multi_budget_report_detects_equivalent_rank_architectures():
    ranks = {
        "mean_per_image_effective_rank": [
            11.73, 14.07, 14.47, 26.65, 28.45, 28.28, 40.06, 40.05, 33.68
        ],
        "pooled_effective_rank": [
            14.04, 14.93, 14.88, 29.01, 30.96, 31.17, 57.79, 61.93, 60.82
        ],
        "weight_effective_rank": [
            14.04, 15.05, 14.56, 28.73, 30.51, 30.65, 57.72, 61.92, 60.30
        ],
    }
    report = multi_budget_architecture_report(ranks, [0.8, 0.6], random_seed=1001)
    assert report["schema_version"] == 2
    assert [budget["label"] for budget in report["budgets"]] == ["p080", "p060"]
    for budget in report["budgets"]:
        assert budget["max_absolute_difference_vs_reference_percent"] < 0.5
        assert ["erank_pooled", "erank_weight"] in budget[
            "equivalent_architectures"
        ]
        names = {item["name"] for item in budget["architectures"]}
        assert names == {
            "erank_pooled",
            "erank_per_image",
            "erank_weight",
            "uniform",
            "random",
        }

