from __future__ import annotations

from typing import Callable

import numpy as np


def euclidean(weights: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Standard Euclidean distance ``||w - x||``."""
    return np.linalg.norm(weights - sample, axis=-1)


def exponential(weights: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Bounded "exponential" dissimilarity ``1 - exp(-||w - x||^2)``."""
    sq = np.sum((weights - sample) ** 2, axis=-1)
    return 1.0 - np.exp(-sq)


DistanceFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

DISTANCES: dict[str, DistanceFn] = {
    "euclidean": euclidean,
    "exponential": exponential,
}


def get(name: str) -> DistanceFn:
    if name not in DISTANCES:
        raise ValueError(
            f"Unknown distance '{name}'. Available: {list(DISTANCES)}"
        )
    return DISTANCES[name]
