from project_rank.architectures import (
    extend_with_pruning_architectures,
    make_pruning_architecture,
    multi_budget_architecture_report,
)
from project_rank.spec import DEFAULT_INNER_WIDTHS


def _channel_scores(reverse: bool = False):
    result = []
    for layer, width in enumerate(DEFAULT_INNER_WIDTHS):
        values = [float(index + 1 + layer) for index in range(width)]
        result.append(list(reversed(values)) if reverse else values)
    return result


def _rank_sources():
    return {
        "mean_per_image_effective_rank": [12, 14, 15, 26, 28, 28, 40, 40, 34],
        "pooled_effective_rank": [14, 15, 15, 29, 31, 31, 58, 62, 61],
        "weight_effective_rank": [14, 15, 15, 29, 31, 31, 58, 62, 60],
    }


def test_cost_aware_pruning_matches_budget_and_constraints():
    architecture = make_pruning_architecture(
        _channel_scores(), target_parameters=162_470, name="l1_weight"
    )
    assert architecture.name == "l1_weight"
    assert abs(architecture.parameters - 162_470) / 162_470 < 0.01
    assert all(
        4 <= width <= maximum and width % 2 == 0
        for width, maximum in zip(
            architecture.inner_widths, DEFAULT_INNER_WIDTHS, strict=True
        )
    )


def test_schema_v3_adds_both_classic_criteria():
    base = multi_budget_architecture_report(_rank_sources(), [0.8, 0.6], 1001)
    report = extend_with_pruning_architectures(
        base,
        {"l1_weight": _channel_scores(), "taylor": _channel_scores(reverse=True)},
    )
    assert report["schema_version"] == 3
    for budget in report["budgets"]:
        names = {item["name"] for item in budget["architectures"]}
        assert {"l1_weight", "taylor"} <= names
        assert budget["max_absolute_difference_vs_reference_percent"] < 1.0
