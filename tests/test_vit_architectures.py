import itertools
import random

import pytest

pytest.importorskip("torch")

from project_rank.vit_architectures import random_vit_architecture, scaled_vit_architecture
from project_rank.vit_spec import ViTSpec, count_vit_parameters


def test_full_p060_budget_and_random_exact_match():
    full = count_vit_parameters()
    erank = scaled_vit_architecture([39 + i / 2 for i in range(12)], round(.6 * full))
    assert abs(erank.parameters / full - .6) < .001
    profiles = [random_vit_architecture(erank.parameters, seed) for seed in range(1001, 1011)]
    assert all(a.parameters == erank.parameters for a in profiles)
    assert len({a.inner_widths for a in profiles}) == 10
    assert profiles[0] == random_vit_architecture(erank.parameters, 1001)
    assert all(4 <= w <= 768 and w % 2 == 0 for a in profiles for w in a.inner_widths)


def test_scale_not_limited_to_sixteen_and_p030_rejected():
    full = count_vit_parameters()
    result = scaled_vit_architecture([1.] * 12, round(.9 * full))
    assert min(result.inner_widths) > 600
    with pytest.raises(ValueError, match="Unreachable"):
        scaled_vit_architecture([40.] * 12, round(.3 * full))
    with pytest.raises(ValueError, match="Unreachable"):
        random_vit_architecture(round(.3 * full), 1001)


def test_scaled_budget_ties_and_iterative_caps():
    spec = ViTSpec(embed_dim=6, depth=2, num_heads=3)
    base = (10, 12)
    result = scaled_vit_architecture([1., 2.], count_vit_parameters([6, 8], spec), spec, base)
    assert all(w <= b for w, b in zip(result.inner_widths, base))
    assert result.parameters <= count_vit_parameters(base, spec)
    with pytest.raises(ValueError):
        scaled_vit_architecture([0., 2.], result.parameters, spec, base)


def test_conditional_sampler_support_and_no_global_rng_change():
    spec = ViTSpec(embed_dim=6, depth=3, num_heads=3)
    target = count_vit_parameters([6, 6, 6], spec)
    base = (8, 8, 8)
    expected = {w for w in itertools.product((4, 6, 8), repeat=3) if sum(w) == 18}
    state = random.getstate()
    draws = [random_vit_architecture(target, seed, spec, base).inner_widths for seed in range(200)]
    assert set(draws) == expected
    assert random.getstate() == state
    assert all(count_vit_parameters(w, spec) == target for w in draws)


def test_extreme_budgets():
    spec = ViTSpec(embed_dim=6, depth=2, num_heads=3)
    for widths in ([4, 4], [24, 24]):
        target = count_vit_parameters(widths, spec)
        assert random_vit_architecture(target, 1, spec).inner_widths == tuple(widths)
