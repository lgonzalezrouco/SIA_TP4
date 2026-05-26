"""Inner-product / orthogonality analysis for groups of stored patterns.

Replicates the notebook the professor showed: for every combination of
``k`` patterns out of the alphabet, compute the off-diagonal absolute
inner-product matrix, its mean, its max and how many off-diagonal entries
reach that max (counting each unordered pair once).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def pairwise_dot_matrix(group: dict[str, np.ndarray] | list[np.ndarray]) -> np.ndarray:
    """Inner-product matrix with zeroed diagonal.

    Accepts either a dict ``{name: 2D pattern}`` or a list of flat vectors.
    Output is ``k x k`` with ``M[i, j] = <p_i, p_j>`` and ``diag = 0``.
    """
    if isinstance(group, dict):
        vectors = [p.reshape(-1) for p in group.values()]
    else:
        vectors = [np.asarray(p).reshape(-1) for p in group]
    G = np.stack(vectors).astype(float)
    M = G @ G.T
    np.fill_diagonal(M, 0.0)
    return M


def combinations_summary(
    letters: dict[str, np.ndarray], k: int = 4
) -> pd.DataFrame:
    """Summary table over every k-combination of letters.

    Columns
    -------
    grupo: tuple of letter names
    pi_medio: mean of |M_ij| over off-diagonal entries
    pi_max: max of |M_ij| over off-diagonal entries
    count_max: how many distinct unordered pairs reach pi_max
    """
    flat = {name: p.reshape(-1).astype(float) for name, p in letters.items()}
    rows = []
    for combo in itertools.combinations(flat.keys(), k):
        G = np.stack([flat[name] for name in combo])
        M = np.abs(G @ G.T)
        np.fill_diagonal(M, 0.0)
        size = M.size - M.shape[0]  # off-diagonal cell count
        pi_mean = float(M.sum() / size)
        pi_max = float(M.max())
        count_max = int(np.count_nonzero(M == pi_max) // 2)
        rows.append(
            {
                "grupo": combo,
                "pi_medio": pi_mean,
                "pi_max": pi_max,
                "count_max": count_max,
            }
        )
    df = pd.DataFrame(rows)
    return df


def select_best_group(
    df: pd.DataFrame,
    *,
    by: str = "pi_medio",
) -> tuple[str, ...]:
    """Devuelve el grupo con menor PI medio (desempate por pi_max y count_max)."""
    sort_cols = ["pi_medio", "pi_max", "count_max"]
    if by == "pi_max":
        sort_cols = ["pi_max", "pi_medio", "count_max"]
    row = df.sort_values(sort_cols).iloc[0]
    grupo = row["grupo"]
    if isinstance(grupo, str):
        return tuple(grupo)
    return tuple(grupo)


def full_dot_matrix(
    letters: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    """Dot-product matrix across the entire alphabet (with zeroed diagonal)."""
    names = list(letters.keys())
    G = np.stack([letters[n].reshape(-1) for n in names]).astype(float)
    M = G @ G.T
    np.fill_diagonal(M, 0.0)
    return M, names
