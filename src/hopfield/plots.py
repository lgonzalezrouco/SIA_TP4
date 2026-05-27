"""Hopfield-specific visualizations: letter grids, recall steps, energy and
ortogonality heatmaps/scatters."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CMAP_BIN = "Blues"


def _draw_pattern(ax: plt.Axes, pattern: np.ndarray, title: str | None = None) -> None:
    """Render a single ±1 (or 0/1) 2D pattern with a clear grid."""
    arr = np.asarray(pattern)
    if arr.ndim == 1:
        side = int(np.sqrt(arr.size))
        arr = arr.reshape(side, side)
    binary = (arr > 0).astype(int)
    rows, cols = binary.shape
    ax.imshow(binary, cmap=CMAP_BIN, vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="grey", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    if title:
        ax.set_title(title, fontsize=10)


def plot_letters_grid(
    letters: dict[str, np.ndarray], ncols: int = 6, figsize_per_cell: float = 1.6
) -> plt.Figure:
    """Show every stored letter (replicates the notebook's alphabet grid)."""
    items = list(letters.items())
    n = len(items)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * figsize_per_cell, nrows * figsize_per_cell)
    )
    axes = np.atleast_2d(axes)
    for idx in range(nrows * ncols):
        ax = axes.ravel()[idx]
        if idx < n:
            name, pat = items[idx]
            _draw_pattern(ax, pat, title=name)
        else:
            ax.axis("off")
    fig.tight_layout()
    return fig


def plot_recall_steps(
    history: Sequence[np.ndarray],
    shape: tuple[int, int],
    title: str = "",
    target: np.ndarray | None = None,
    max_cols: int = 8,
) -> plt.Figure:
    """Show every state in the recall trajectory (query -> ... -> final)."""
    n = len(history)
    if target is not None:
        n += 1
    ncols = min(n, max_cols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.7, nrows * 1.9))
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        ax = axes.ravel()[idx]
        if idx >= n:
            ax.axis("off")
            continue
        if target is not None and idx == n - 1:
            _draw_pattern(ax, target.reshape(shape), title="target")
            continue
        label = "query" if idx == 0 else (
            "final" if idx == len(history) - 1 else f"t={idx}"
        )
        _draw_pattern(ax, history[idx].reshape(shape), title=label)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_energy(energies: Sequence[float], title: str = "Energy vs step") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(range(len(energies)), energies, marker="o", linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("E(S)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_dot_matrix(
    M: np.ndarray, names: Sequence[str], title: str = "Inner-product matrix"
) -> plt.Figure:
    """Heatmap of ``|<xi_i, xi_j>|`` (annotated)."""
    abs_M = np.abs(M)
    n = abs_M.shape[0]
    figsize = (max(6, 0.45 * n + 2), max(5, 0.45 * n + 1))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(abs_M, cmap="magma_r")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    if n <= 12:
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{abs_M[i, j]:.0f}",
                    ha="center",
                    va="center",
                    color="white" if abs_M[i, j] > abs_M.max() / 2 else "black",
                    fontsize=8,
                )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|<xi_i, xi_j>|")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_combinations_scatter(
    df: pd.DataFrame, title: str = "Group ortogonality: mean vs max"
) -> plt.Figure:
    """Scatter of every k-combination: ``pi_medio`` vs ``pi_max`` (size = count_max)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sizes = 25 + 20 * (df["count_max"].clip(upper=8))
    sc = ax.scatter(
        df["pi_medio"],
        df["pi_max"],
        c=df["count_max"],
        s=sizes,
        cmap="viridis",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    plt.colorbar(sc, ax=ax, label="count_max (#pairs reaching pi_max)")
    ax.set_xlabel("PI medio (off-diagonal mean of |<xi_i, xi_j>|)")
    ax.set_ylabel("PI max")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Highlight and annotate only the two most relevant extremes by PI medio:
    # best (lowest) and worst (highest). This avoids heavy label overlap.
    df_sorted = df.sort_values("pi_medio")
    best = df_sorted.iloc[0]
    worst = df_sorted.iloc[-1]

    ax.scatter(
        [best["pi_medio"]],
        [best["pi_max"]],
        s=220,
        marker="*",
        color="limegreen",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )
    ax.scatter(
        [worst["pi_medio"]],
        [worst["pi_max"]],
        s=220,
        marker="X",
        color="crimson",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )

    ax.annotate(
        f"best: {''.join(best['grupo'])}",
        (best["pi_medio"], best["pi_max"]),
        xytext=(-70, -16),
        textcoords="offset points",
        fontsize=9,
        color="black",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="limegreen", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="limegreen", lw=1.2),
        zorder=6,
    )
    ax.annotate(
        f"worst: {''.join(worst['grupo'])}",
        (worst["pi_medio"], worst["pi_max"]),
        xytext=(-90, 14),
        textcoords="offset points",
        fontsize=9,
        color="black",
        fontweight="bold",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="crimson", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2),
        zorder=6,
    )
    fig.tight_layout()
    return fig


def plot_capacity_heatmap(
    success_rate: np.ndarray,
    p_values: Sequence[int],
    noise_values: Sequence[int],
    title: str = "Recall success rate",
) -> plt.Figure:
    """Heatmap of recall accuracy across (p, noise_flips)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(success_rate, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(noise_values)))
    ax.set_xticklabels(noise_values)
    ax.set_yticks(range(len(p_values)))
    ax.set_yticklabels(p_values)
    ax.set_xlabel("# bits flipped (noise)")
    ax.set_ylabel("# patterns stored (p)")
    ax.set_title(title)
    for i in range(success_rate.shape[0]):
        for j in range(success_rate.shape[1]):
            ax.text(
                j,
                i,
                f"{success_rate[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="success rate")
    fig.tight_layout()
    return fig


def plot_spurious_examples(
    examples: list[dict],
    shape: tuple[int, int],
    title: str = "Spurious attractors",
    max_examples: int = 8,
) -> plt.Figure:
    """Horizontal layout: top row = queries, bottom row = spurious finals."""
    examples = examples[:max_examples]
    n = len(examples)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 2.2))
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "No spurious examples found", ha="center", va="center")
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.2))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, ex in enumerate(examples):
        _draw_pattern(
            axes[0, i],
            ex["query"].reshape(shape),
            title=f"query ({ex['source']})",
        )
        _draw_pattern(
            axes[1, i],
            ex["final"].reshape(shape),
            title="spurious final",
        )
    fig.suptitle(title)
    fig.tight_layout()
    return fig
