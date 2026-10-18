from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np

from project_rank.spec import (
    DEFAULT_INNER_WIDTHS,
    count_resnet20_parameters,
)
from project_rank.pruning_criteria import l2_normalize


@dataclass(frozen=True)
class Architecture:
    name: str
    inner_widths: tuple[int, ...]
    parameters: int

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["inner_widths"] = list(self.inner_widths)
        return data


def _round_width(value: float, minimum: int, maximum: int, multiple: int) -> int:
    rounded = int(round(value / multiple) * multiple)
    return min(max(rounded, minimum), maximum)


def _best_scaled_widths(
    scores: Sequence[float],
    base_widths: Sequence[int],
    target_parameters: int,
    minimum: int,
    multiple: int,
) -> tuple[int, ...]:
    if len(scores) != len(base_widths):
        raise ValueError("scores and base_widths must have equal length")
    if any(score <= 0 or not math.isfinite(score) for score in scores):
        raise ValueError("All scores must be positive finite numbers")

    scales = np.geomspace(0.05, 16.0, num=2000)
    candidates: set[tuple[int, ...]] = set()
    for scale in scales:
        candidate = tuple(
            _round_width(scale * score, minimum, maximum, multiple)
            for score, maximum in zip(scores, base_widths, strict=True)
        )
        candidates.add(candidate)
    return min(
        candidates,
        key=lambda widths: abs(count_resnet20_parameters(widths) - target_parameters),
    )


def make_erank_architecture(
    ranks: Sequence[float],
    target_ratio: float,
    base_widths: Sequence[int] = DEFAULT_INNER_WIDTHS,
    minimum: int = 4,
    multiple: int = 2,
) -> Architecture:
    if not 0 < target_ratio <= 1:
        raise ValueError("target_ratio must be in (0, 1]")
    full_parameters = count_resnet20_parameters(base_widths)
    target = int(round(full_parameters * target_ratio))
    widths = _best_scaled_widths(
        ranks, base_widths, target, minimum=minimum, multiple=multiple
    )
    return Architecture("erank", widths, count_resnet20_parameters(widths))


def make_uniform_architecture(
    target_parameters: int,
    base_widths: Sequence[int] = DEFAULT_INNER_WIDTHS,
    minimum: int = 4,
    multiple: int = 2,
) -> Architecture:
    widths = _best_scaled_widths(
        base_widths,
        base_widths,
        target_parameters,
        minimum=minimum,
        multiple=multiple,
    )
    return Architecture("uniform", widths, count_resnet20_parameters(widths))


def make_random_architecture(
    target_parameters: int,
    seed: int,
    base_widths: Sequence[int] = DEFAULT_INNER_WIDTHS,
    minimum: int = 4,
    multiple: int = 2,
    trials: int = 20_000,
) -> Architecture:
    rng = random.Random(seed)
    options = [
        list(range(minimum, maximum + 1, multiple)) for maximum in base_widths
    ]
    best: tuple[int, ...] | None = None
    best_difference = math.inf
    for _ in range(trials):
        candidate = tuple(rng.choice(layer_options) for layer_options in options)
        difference = abs(count_resnet20_parameters(candidate) - target_parameters)
        if difference < best_difference:
            best = candidate
            best_difference = difference
            if difference == 0:
                break
    assert best is not None
    return Architecture("random", best, count_resnet20_parameters(best))


def architecture_report(
    ranks: Sequence[float],
    target_ratio: float,
    random_seed: int,
) -> dict[str, object]:
    full = Architecture(
        "full",
        tuple(DEFAULT_INNER_WIDTHS),
        count_resnet20_parameters(DEFAULT_INNER_WIDTHS),
    )
    erank = make_erank_architecture(ranks, target_ratio)
    uniform = make_uniform_architecture(erank.parameters)
    random_arch = make_random_architecture(erank.parameters, random_seed)
    architectures = [full, erank, uniform, random_arch]
    return {
        "target_parameter_ratio": target_ratio,
        "architectures": [
            {
                **architecture.to_dict(),
                "parameter_ratio_vs_full": architecture.parameters / full.parameters,
                "difference_vs_erank_percent": (
                    100.0 * (architecture.parameters - erank.parameters) / erank.parameters
                ),
            }
            for architecture in architectures
        ],
    }


RANK_SOURCE_COLUMNS = {
    "erank_pooled": "pooled_effective_rank",
    "erank_per_image": "mean_per_image_effective_rank",
    "erank_weight": "weight_effective_rank",
}


def _rename(architecture: Architecture, name: str) -> Architecture:
    return replace(architecture, name=name)


def _best_uniform_for_budget(target_parameters: int) -> Architecture:
    """Choose the closest genuinely uniform candidate.

    Both integer and even-channel grids are considered. Coarser rounding can
    occasionally match a parameter budget better while preserving one common
    width multiplier for all stages.
    """
    candidates = [
        make_uniform_architecture(target_parameters, multiple=1),
        make_uniform_architecture(target_parameters, multiple=2),
    ]
    return min(
        candidates,
        key=lambda item: abs(item.parameters - target_parameters),
    )


def multi_budget_architecture_report(
    ranks_by_source: dict[str, Sequence[float]],
    target_ratios: Sequence[float],
    random_seed: int,
) -> dict[str, object]:
    """Build comparable architectures for several rank sources and budgets."""
    missing = set(RANK_SOURCE_COLUMNS.values()) - set(ranks_by_source)
    if missing:
        raise ValueError(f"Missing rank sources: {sorted(missing)}")
    if not target_ratios:
        raise ValueError("At least one target ratio is required")

    full_parameters = count_resnet20_parameters(DEFAULT_INNER_WIDTHS)
    budgets: list[dict[str, object]] = []
    for budget_index, ratio_value in enumerate(target_ratios):
        ratio = float(ratio_value)
        if not 0 < ratio <= 1:
            raise ValueError("Every target ratio must be in (0, 1]")

        rank_architectures: list[Architecture] = []
        for name, column in RANK_SOURCE_COLUMNS.items():
            architecture = make_erank_architecture(
                ranks_by_source[column],
                ratio,
                multiple=2,
            )
            rank_architectures.append(_rename(architecture, name))

        pooled = next(
            item for item in rank_architectures if item.name == "erank_pooled"
        )
        reference_parameters = pooled.parameters
        uniform = _best_uniform_for_budget(reference_parameters)
        random_arch = make_random_architecture(
            reference_parameters,
            seed=random_seed + budget_index,
            multiple=2,
        )
        architectures = [*rank_architectures, uniform, random_arch]

        width_groups: dict[tuple[int, ...], list[str]] = {}
        for architecture in architectures:
            width_groups.setdefault(architecture.inner_widths, []).append(
                architecture.name
            )
        equivalent = [
            names for names in width_groups.values() if len(names) > 1
        ]

        entries = []
        for architecture in architectures:
            entry = architecture.to_dict()
            entry["parameter_ratio_vs_full"] = (
                architecture.parameters / full_parameters
            )
            entry["difference_vs_reference_percent"] = (
                100.0
                * (architecture.parameters - reference_parameters)
                / reference_parameters
            )
            entry["rank_source"] = RANK_SOURCE_COLUMNS.get(architecture.name)
            entries.append(entry)

        budgets.append(
            {
                "label": f"p{int(round(ratio * 100)):03d}",
                "requested_parameter_ratio": ratio,
                "requested_parameters": int(round(full_parameters * ratio)),
                "reference_architecture": "erank_pooled",
                "reference_parameters": reference_parameters,
                "architectures": entries,
                "equivalent_architectures": equivalent,
                "max_absolute_difference_vs_reference_percent": max(
                    abs(float(item["difference_vs_reference_percent"]))
                    for item in entries
                ),
            }
        )

    return {
        "schema_version": 2,
        "full_architecture": {
            "name": "full",
            "inner_widths": list(DEFAULT_INNER_WIDTHS),
            "parameters": full_parameters,
        },
        "channel_rounding": {
            "erank": 2,
            "random": 2,
            "uniform": "best of integer and even grids",
            "minimum_width": 4,
        },
        "budgets": budgets,
    }


def make_pruning_architecture(
    channel_scores: Sequence[Sequence[float]],
    target_parameters: int,
    name: str,
    base_widths: Sequence[int] = DEFAULT_INNER_WIDTHS,
    minimum: int = 4,
    multiple: int = 2,
) -> Architecture:
    """Build a cost-aware structured-pruning architecture.

    Scores are normalized independently for every layer, matching the common
    global-pruning practice for criteria whose raw scales differ by layer.
    At every step the least important removable channel group per saved
    parameter is pruned. The closest state on that greedy path is returned.
    """
    if len(channel_scores) != len(base_widths):
        raise ValueError("Expected one channel-score vector for every block")
    if minimum < 1 or multiple < 1 or minimum % multiple != 0:
        raise ValueError("minimum must be positive and divisible by multiple")

    normalized: list[list[float]] = []
    for scores, width in zip(channel_scores, base_widths, strict=True):
        if len(scores) != width:
            raise ValueError(
                f"Expected {width} channel scores for a block, got {len(scores)}"
            )
        normalized.append(sorted(l2_normalize(scores)))

    widths = tuple(int(value) for value in base_widths)
    removed = [0] * len(widths)
    path = [widths]
    while any(width > minimum for width in widths):
        current_parameters = count_resnet20_parameters(widths)
        candidates: list[tuple[float, int, tuple[int, ...]]] = []
        for index, width in enumerate(widths):
            if width - multiple < minimum:
                continue
            candidate = list(widths)
            candidate[index] -= multiple
            candidate_widths = tuple(candidate)
            saving = current_parameters - count_resnet20_parameters(candidate_widths)
            start = removed[index]
            group_score = sum(normalized[index][start : start + multiple])
            candidates.append((group_score / saving, index, candidate_widths))
        if not candidates:
            break
        _, selected_index, widths = min(candidates, key=lambda item: (item[0], item[1]))
        removed[selected_index] += multiple
        path.append(widths)

    best = min(
        path,
        key=lambda item: (
            abs(count_resnet20_parameters(item) - target_parameters),
            count_resnet20_parameters(item) > target_parameters,
        ),
    )
    return Architecture(name, best, count_resnet20_parameters(best))


def extend_with_pruning_architectures(
    architecture_report: dict[str, object],
    scores_by_method: dict[str, Sequence[Sequence[float]]],
) -> dict[str, object]:
    """Append L1/Taylor architectures to a schema-v2 ERank report."""
    if architecture_report.get("schema_version") != 2:
        raise ValueError("Expected an architecture report with schema_version=2")
    expected_methods = {"l1_weight", "taylor"}
    if set(scores_by_method) != expected_methods:
        raise ValueError(f"Expected pruning methods {sorted(expected_methods)}")

    report = {
        **architecture_report,
        "schema_version": 3,
        "pruning_policy": {
            "score_normalization": "layer_l2",
            "selection": "lowest normalized score per saved parameter",
            "channel_group": 2,
            "minimum_width": 4,
        },
    }
    full_parameters = int(report["full_architecture"]["parameters"])
    updated_budgets: list[dict[str, object]] = []
    for original_budget in report["budgets"]:
        budget = dict(original_budget)
        architectures = [dict(item) for item in budget["architectures"]]
        reference_parameters = int(budget["reference_parameters"])
        for method in ("l1_weight", "taylor"):
            architecture = make_pruning_architecture(
                scores_by_method[method],
                reference_parameters,
                name=method,
            )
            entry = architecture.to_dict()
            entry["parameter_ratio_vs_full"] = architecture.parameters / full_parameters
            entry["difference_vs_reference_percent"] = (
                100.0
                * (architecture.parameters - reference_parameters)
                / reference_parameters
            )
            entry["rank_source"] = None
            entry["pruning_score_source"] = method
            architectures.append(entry)

        width_groups: dict[tuple[int, ...], list[str]] = {}
        for item in architectures:
            widths = tuple(int(value) for value in item["inner_widths"])
            width_groups.setdefault(widths, []).append(str(item["name"]))
        budget["architectures"] = architectures
        budget["equivalent_architectures"] = [
            names for names in width_groups.values() if len(names) > 1
        ]
        budget["max_absolute_difference_vs_reference_percent"] = max(
            abs(float(item["difference_vs_reference_percent"]))
            for item in architectures
        )
        updated_budgets.append(budget)
    report["budgets"] = updated_budgets
    return report

