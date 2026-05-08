from __future__ import annotations

from typing import Sequence

import numpy as np

from utils import distances as dist_mod
from utils import grid as grid_mod
from utils import initialization as init_mod
from utils import schedules


class SOM:
    def __init__(
        self,
        n_features: int,
        grid_rows: int,
        grid_cols: int,
        *,
        topology: str = "rectangular",
        init_method: str = "sample",
        distance: str = "euclidean",
        neighborhood: str = "gaussian",
        learning_rate: schedules.ScheduleSpec = 0.5,
        radius: schedules.ScheduleSpec | None = None,
        epochs: int = 500,
        seed: int | None = None,
    ) -> None:
        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError("grid_rows and grid_cols must be positive")

        self.n_features = n_features
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.topology = topology
        self.init_method = init_method
        self.distance_name = distance
        self.neighborhood_kind = neighborhood
        self.epochs = epochs
        self.seed = seed

        if radius is None:
            r0 = max(grid_rows, grid_cols) / 2.0
            radius = {"kind": "exponential", "initial": r0, "final": 1.0}

        self._lr_schedule = schedules.parse(learning_rate)
        self._radius_schedule = schedules.parse(radius)
        self._distance_fn = dist_mod.get(distance)
        self._rng = np.random.default_rng(seed)

        self._grid_dists = grid_mod.pairwise_grid_distances(
            grid_rows, grid_cols, topology
        )

        self.weights: np.ndarray | None = None
        self.history: list[dict] = []

    def fit(self, data: np.ndarray) -> "SOM":
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != self.n_features:
            raise ValueError(
                f"data must have shape (n_samples, {self.n_features}); got {data.shape}"
            )

        self.weights = init_mod.initialize(
            self.init_method,
            self.grid_rows,
            self.grid_cols,
            data,
            self._rng,
        )
        self.history = []

        n_samples = len(data)
        total_steps = max(self.epochs, 1)

        for epoch in range(self.epochs):
            lr = self._lr_schedule(epoch, total_steps)
            sigma = self._radius_schedule(epoch, total_steps)

            order = self._rng.permutation(n_samples)
            for sample_idx in order:
                x = data[sample_idx]
                bmu_flat = self._bmu_flat_index(x)
                h = grid_mod.neighborhood(
                    self._grid_dists[bmu_flat], sigma, self.neighborhood_kind
                ).reshape(self.grid_rows, self.grid_cols, 1)
                self.weights += lr * h * (x - self.weights)

            self.history.append(
                {
                    "epoch": epoch,
                    "learning_rate": lr,
                    "radius": sigma,
                    "quantization_error": self.quantization_error(data),
                }
            )

        return self

    def _bmu_flat_index(self, sample: np.ndarray) -> int:
        scores = self._distance_fn(self.weights, sample)
        return int(np.argmin(scores))

    def bmu(self, sample: np.ndarray) -> tuple[int, int]:
        flat = self._bmu_flat_index(np.asarray(sample, dtype=float))
        return divmod(flat, self.grid_cols)

    def predict(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        out = np.empty((len(data), 2), dtype=int)
        for i, x in enumerate(data):
            out[i] = self.bmu(x)
        return out

    def quantization_error(self, data: np.ndarray) -> float:
        data = np.asarray(data, dtype=float)
        total = 0.0
        for x in data:
            scores = self._distance_fn(self.weights, x)
            total += float(scores.min())
        return total / max(len(data), 1)

    def hit_map(self, data: np.ndarray) -> np.ndarray:
        counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        for r, c in self.predict(data):
            counts[r, c] += 1
        return counts

    def assignments(
        self, data: np.ndarray, labels: Sequence[str] | None = None
    ) -> dict[tuple[int, int], list]:
        """Map each (row, col) to the list of samples (or labels) assigned there."""
        out: dict[tuple[int, int], list] = {}
        coords = self.predict(data)
        for i, (r, c) in enumerate(coords):
            key = (int(r), int(c))
            value = labels[i] if labels is not None else int(i)
            out.setdefault(key, []).append(value)
        return out

    def u_matrix(self) -> np.ndarray:
        """Average weight-distance to immediate grid neighbors per neuron."""
        if self.weights is None:
            raise RuntimeError("SOM must be fitted before computing U-matrix")
        flat_w = self.weights.reshape(-1, self.n_features)
        neigh = grid_mod.neighbor_indices(self.grid_rows, self.grid_cols, self.topology)
        u = np.zeros(self.grid_rows * self.grid_cols, dtype=float)
        for i, nbrs in enumerate(neigh):
            if not nbrs:
                u[i] = 0.0
                continue
            diffs = flat_w[nbrs] - flat_w[i]
            u[i] = float(np.mean(np.linalg.norm(diffs, axis=1)))
        return u.reshape(self.grid_rows, self.grid_cols)
