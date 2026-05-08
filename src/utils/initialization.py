"""Weight initialization strategies for the SOM."""

from __future__ import annotations

import numpy as np


def random_weights(
    rows: int,
    cols: int,
    n_features: int,
    rng: np.random.Generator,
    low: float = -1.0,
    high: float = 1.0,
) -> np.ndarray:
    return rng.uniform(low, high, size=(rows, cols, n_features))


def sample_weights(
    rows: int,
    cols: int,
    data: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(data) == 0:
        raise ValueError("sample initialization requires non-empty data")
    n = rows * cols
    indices = rng.integers(0, len(data), size=n)
    return data[indices].reshape(rows, cols, data.shape[1]).astype(float).copy()


def initialize(
    method: str,
    rows: int,
    cols: int,
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    random_low: float = -1.0,
    random_high: float = 1.0,
) -> np.ndarray:
    if method == "random":
        return random_weights(
            rows, cols, data.shape[1], rng, low=random_low, high=random_high
        )
    if method == "sample":
        return sample_weights(rows, cols, data, rng)
    raise ValueError(
        f"Unknown init method '{method}'. Available: 'random', 'sample'"
    )
