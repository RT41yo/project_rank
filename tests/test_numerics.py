import numpy as np
import pytest

from project_rank.numerics import effective_rank_numpy


def test_equal_singular_values_give_full_effective_rank():
    matrix = np.eye(5)
    assert effective_rank_numpy(matrix, center=False) == pytest.approx(5.0)


def test_rank_one_matrix_has_effective_rank_one():
    column = np.array([-2.0, -1.0, 1.0, 2.0])[:, None]
    matrix = column @ np.array([[1.0, 2.0, -3.0]])
    assert effective_rank_numpy(matrix, center=True) == pytest.approx(1.0)


def test_zero_matrix_has_zero_effective_rank():
    assert effective_rank_numpy(np.zeros((10, 4))) == 0.0

