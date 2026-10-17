from __future__ import annotations

import numpy as np


def effective_rank_numpy(matrix: np.ndarray, center: bool = True) -> float:
    """Roy--Vetterli effective rank from normalized singular values."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if center:
        values = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(values, compute_uv=False)
    total = float(singular_values.sum())
    if total <= np.finfo(np.float64).eps:
        return 0.0
    probabilities = singular_values / total
    probabilities = probabilities[probabilities > 0]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.exp(entropy))

