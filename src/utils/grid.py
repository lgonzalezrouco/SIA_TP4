from __future__ import annotations

import numpy as np


def grid_coords(rows: int, cols: int, topology: str) -> np.ndarray:
    """Cartesian (y, x) coordinate of each neuron, shape ``(rows*cols, 2)``,
    in row-major order matching ``np.ndindex(rows, cols)``."""
    rs, cs = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    if topology == "rectangular":
        y = rs.astype(float)
        x = cs.astype(float)
    elif topology == "hexagonal":
        y = rs.astype(float) * (np.sqrt(3.0) / 2.0)
        x = cs.astype(float) + 0.5 * (rs % 2)
    else:
        raise ValueError(
            f"Unknown topology '{topology}'. Available: 'rectangular', 'hexagonal'"
        )
    return np.stack([y.ravel(), x.ravel()], axis=1)


def pairwise_grid_distances(rows: int, cols: int, topology: str) -> np.ndarray:
    """``(rows*cols, rows*cols)`` matrix of grid distances between neurons."""
    coords = grid_coords(rows, cols, topology)
    diffs = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def neighborhood(
    grid_distances: np.ndarray, sigma: float, kind: str = "gaussian"
) -> np.ndarray:
    """Map grid distance to neighborhood weight in [0, 1]."""
    if sigma <= 0:
        # Degenerate: only the BMU itself updates.
        return (grid_distances == 0.0).astype(float)
    if kind == "gaussian":
        return np.exp(-(grid_distances**2) / (2.0 * sigma**2))
    if kind == "bubble":
        return (grid_distances <= sigma).astype(float)
    raise ValueError(
        f"Unknown neighborhood kind '{kind}'. Available: 'gaussian', 'bubble'"
    )


def neighbor_indices(rows: int, cols: int, topology: str) -> list[list[int]]:
    """For each neuron (row-major flat index) the list of immediate-neighbor
    flat indices. Useful for U-matrix calculations.

    Rectangular: 4-connected (N, S, E, W).
    Hexagonal (offset rows): 6 neighbors, shifted depending on row parity.
    """
    out: list[list[int]] = []
    if topology == "rectangular":
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif topology == "hexagonal":
        even = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
        odd = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
    else:
        raise ValueError(f"Unknown topology '{topology}'")

    for r in range(rows):
        for c in range(cols):
            if topology == "rectangular":
                deltas = offsets
            else:
                deltas = even if (r % 2 == 0) else odd
            neigh = []
            for dr, dc in deltas:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    neigh.append(rr * cols + cc)
            out.append(neigh)
    return out
