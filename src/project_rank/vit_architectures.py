"""MLP-only ViT width allocation. Budgets always refer to the full ViT."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence

from project_rank.architectures import Architecture
from project_rank.vit_spec import ViTSpec, count_vit_parameters


def _bounds(spec: ViTSpec, base_widths: Sequence[int] | None, minimum: int, multiple: int) -> tuple[int, ...]:
    widths = spec.widths(base_widths)
    if minimum < 1 or multiple < 1 or minimum % multiple:
        raise ValueError("minimum must be positive and divisible by multiple")
    if any(w < minimum or w > full or w % multiple for w, full in zip(widths, spec.default_widths, strict=True)):
        raise ValueError("Source widths must be on the grid and within the full model")
    return widths


def _check_budget(target: int, base: tuple[int, ...], spec: ViTSpec, minimum: int) -> None:
    low, high = count_vit_parameters([minimum] * spec.depth, spec), count_vit_parameters(base, spec)
    if not low <= target <= high:
        raise ValueError(f"Unreachable MLP-only budget {target}; feasible parameters [{low}, {high}]")


def scaled_vit_architecture(
    scores: Sequence[float], target_parameters: int, spec: ViTSpec = ViTSpec(),
    base_widths: Sequence[int] | None = None, minimum: int = 4, multiple: int = 2,
    name: str = "erank",
) -> Architecture:
    """Closest rounded alpha*score profile, including all rounding breakpoints.

    Unlike the ResNet implementation's fixed alpha grid, this covers the entire
    admissible scaling path even when ranks are bounded by the token count.
    It does NOT independently adjust widths to improve the budget afterwards.
    """
    base = _bounds(spec, base_widths, minimum, multiple)
    _check_budget(target_parameters, base, spec, minimum)
    if len(scores) != spec.depth or any(not math.isfinite(s) or s <= 0 for s in scores):
        raise ValueError("Expected one positive finite score per block")
    breaks = sorted({
        (width + multiple / 2) / score
        for score, upper in zip(scores, base, strict=True)
        for width in range(minimum, upper, multiple)
    })
    scales = [0.0, *breaks]
    scales.extend((left + right) / 2 for left, right in zip([0.0, *breaks], breaks))
    if breaks:
        scales.append(breaks[-1] * 2)
    candidates = {tuple(
        min(upper, max(minimum, round(scale * score / multiple) * multiple))
        for score, upper in zip(scores, base, strict=True)
    ) for scale in scales}
    best = min(candidates, key=lambda w: (
        abs(count_vit_parameters(w, spec) - target_parameters),
        count_vit_parameters(w, spec) > target_parameters, w,
    ))
    return Architecture(name, best, count_vit_parameters(best, spec))


def random_vit_architecture(
    target_parameters: int, seed: int, spec: ViTSpec = ViTSpec(),
    base_widths: Sequence[int] | None = None, minimum: int = 4, multiple: int = 2,
) -> Architecture:
    """Uniform bounded composition conditional on the nearest feasible width sum.

    Dynamic-programming completion counts give each admissible ordered width
    profile equal probability. This is NOT independent sampling + budget repair.
    No activations, scores or validation accuracy participate in sampling.
    """
    base = _bounds(spec, base_widths, minimum, multiple)
    _check_budget(target_parameters, base, spec, minimum)
    low = count_vit_parameters([minimum] * spec.depth, spec)
    unit_cost = (2 * spec.embed_dim + 1) * multiple
    capacity = [(w - minimum) // multiple for w in base]
    value = (target_parameters - low) / unit_cost
    units = min((math.floor(value), math.ceil(value)), key=lambda u: (
        abs(low + unit_cost * u - target_parameters), low + unit_cost * u > target_parameters,
    ))
    # ways[i][s]: number of profiles in blocks i..depth-1 with s remaining units.
    ways = [[1] + [0] * units]
    for cap in reversed(capacity):
        suffix = ways[-1]
        current, window = [], 0
        for s in range(units + 1):
            window += suffix[s]
            if s > cap:
                window -= suffix[s - cap - 1]
            current.append(window)
        ways.append(current)
    ways.reverse()
    rng, remaining, widths = random.Random(seed), units, []
    for i, cap in enumerate(capacity):
        draw = rng.randrange(ways[i][remaining])
        for chosen in range(min(cap, remaining) + 1):
            completions = ways[i + 1][remaining - chosen]
            if draw < completions:
                widths.append(minimum + multiple * chosen)
                remaining -= chosen
                break
            draw -= completions
    result = tuple(widths)
    if remaining or len(result) != spec.depth:
        raise RuntimeError("Failed to sample a complete budget-conditioned profile")
    return Architecture("random", result, count_vit_parameters(result, spec))
