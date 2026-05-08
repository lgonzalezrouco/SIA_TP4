"""SOM-specific visualizations: country map, U-matrix, hit map, component
planes, quantization-error and schedule curves."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon

from utils import grid as grid_mod


def _draw_grid_values(
    values: np.ndarray,
    topology: str,
    ax: plt.Axes,
    *,
    cmap: str = "viridis",
    annotate: bool = True,
    annot_fmt: str = "{:.2f}",
    annot_color: str = "white",
    cbar_label: str | None = None,
):
    """Draw a per-neuron scalar field on a rectangular or hexagonal grid."""
    rows, cols = values.shape
    coords = grid_mod.grid_coords(rows, cols, topology)
    flat = values.ravel()

    if topology == "rectangular":
        im = ax.imshow(values, cmap=cmap, origin="upper")
        if annotate:
            for r in range(rows):
                for c in range(cols):
                    ax.text(
                        c,
                        r,
                        annot_fmt.format(values[r, c]),
                        ha="center",
                        va="center",
                        color=annot_color,
                        fontsize=8,
                    )
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:  # hexagonal
        radius = 1.0 / np.sqrt(3)
        cmap_obj = plt.get_cmap(cmap)
        patches = []
        for y, x in coords:
            patches.append(
                RegularPolygon(
                    (x, -y), numVertices=6, radius=radius, orientation=np.radians(30)
                )
            )
        coll = PatchCollection(patches, cmap=cmap_obj, edgecolor="white")
        coll.set_array(flat)
        ax.add_collection(coll)
        if annotate:
            for (y, x), v in zip(coords, flat):
                ax.text(
                    x,
                    -y,
                    annot_fmt.format(v),
                    ha="center",
                    va="center",
                    color=annot_color,
                    fontsize=8,
                )
        ax.set_xlim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)
        ax.set_ylim(-coords[:, 0].max() - 1, -coords[:, 0].min() + 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(coll, ax=ax, fraction=0.046, pad=0.04)

    if cbar_label:
        cbar.set_label(cbar_label)
    return ax


def plot_u_matrix(
    som, ax: plt.Axes | None = None, title: str | None = None
) -> plt.Axes:
    """Average distance-to-neighbors heatmap. Dark cells = cluster boundaries."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    u = som.u_matrix()
    _draw_grid_values(
        u,
        som.topology,
        ax,
        cmap="bone_r",
        annotate=True,
        annot_fmt="{:.2f}",
        annot_color="red",
        cbar_label="avg neighbor distance",
    )
    ax.set_title(title or f"U-matrix ({som.topology} {som.grid_rows}x{som.grid_cols})")
    return ax


def plot_hit_map(
    som,
    data: np.ndarray,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """How many samples landed on each neuron."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    hits = som.hit_map(data)
    _draw_grid_values(
        hits.astype(float),
        som.topology,
        ax,
        cmap="YlOrRd",
        annotate=True,
        annot_fmt="{:.0f}",
        annot_color="black",
        cbar_label="samples per neuron",
    )
    ax.set_title(title or "Hit map")
    return ax


def plot_country_labels(
    som,
    data: np.ndarray,
    labels: Sequence[str],
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Place the label of each sample on its BMU."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    hits = som.hit_map(data).astype(float)
    _draw_grid_values(
        hits,
        som.topology,
        ax,
        cmap="Blues",
        annotate=False,
        cbar_label="samples per neuron",
    )

    coords = grid_mod.grid_coords(som.grid_rows, som.grid_cols, som.topology)
    bmus = som.predict(data)
    grouped: dict[tuple[int, int], list[str]] = {}
    for label, (r, c) in zip(labels, bmus):
        grouped.setdefault((int(r), int(c)), []).append(str(label))

    for (r, c), names in grouped.items():
        flat = r * som.grid_cols + c
        y, x = coords[flat]
        if som.topology == "rectangular":
            px, py = c, r
        else:
            px, py = x, -y
        ax.text(
            px,
            py,
            "\n".join(names),
            ha="center",
            va="center",
            fontsize=7,
            color="black",
            fontweight="bold",
        )

    ax.set_title(title or "Country assignment per neuron")
    return ax


def plot_component_planes(
    som,
    feature_names: Sequence[str],
    fig: plt.Figure | None = None,
    ncols: int = 4,
) -> plt.Figure:
    """One heatmap per feature showing the learned weight surface."""
    n_features = len(feature_names)
    nrows = (n_features + ncols - 1) // ncols
    if fig is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    else:
        axes = fig.subplots(nrows, ncols)
    axes = np.atleast_2d(axes)

    for idx, name in enumerate(feature_names):
        ax = axes.ravel()[idx]
        _draw_grid_values(
            som.weights[:, :, idx],
            som.topology,
            ax,
            cmap="coolwarm",
            annotate=False,
            cbar_label=None,
        )
        ax.set_title(name)
    for idx in range(n_features, nrows * ncols):
        axes.ravel()[idx].axis("off")
    fig.suptitle("Component planes (weights per feature, z-score)")
    fig.tight_layout()
    return fig


def plot_quantization_error(
    histories: dict[str, list[dict]],
    ax: plt.Axes | None = None,
    title: str = "Quantization error vs epoch",
) -> plt.Axes:
    """Convergence curves. ``histories`` maps a label to a SOM ``history`` list."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for label, history in histories.items():
        epochs = [h["epoch"] for h in history]
        qe = [h["quantization_error"] for h in history]
        ax.plot(epochs, qe, label=label, linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("quantization error")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(histories) > 1:
        ax.legend(fontsize=8, loc="best")
    return ax


def plot_schedule_curves(
    schedules_dict: dict[str, list[float]],
    ax: plt.Axes | None = None,
    ylabel: str = "value",
    title: str = "Schedule",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for label, values in schedules_dict.items():
        ax.plot(values, label=label, linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax
