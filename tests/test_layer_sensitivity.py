import pytest

from project_rank.layer_sensitivity import spearman_correlation


def test_spearman_correlation_with_ties_and_reverse_order():
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert spearman_correlation([1, 1, 1], [1, 2, 3]) is None
