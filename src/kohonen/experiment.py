from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kohonen.plots import (
    plot_component_planes,
    plot_country_labels,
    plot_hit_map,
    plot_quantization_error,
    plot_schedule_curves,
    plot_u_matrix,
)
from kohonen.som import SOM
from utils import schedules

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "kohonen"
FEATURES = [
    "Area",
    "GDP",
    "Inflation",
    "Life.expect",
    "Military",
    "Pop.growth",
    "Unemployment",
]


def _resolve_grid_size(config: dict) -> tuple[int, int]:
    if "grid_rows" in config or "grid_cols" in config:
        rows = config.get("grid_rows", config.get("grid_cols", 5))
        cols = config.get("grid_cols", config.get("grid_rows", 5))
        return int(rows or 5), int(cols or 5)
    size = config.get("grid_size", 5)
    return int(size or 5), int(size or 5)


class KohonenExperiment:
    def __init__(self, config, data, labels):
        self.config = config or {}
        self.data = np.asarray(data, dtype=float)
        self.labels = labels
        self.model: SOM | None = None
        self.output_dir = Path(self.config.get("output_dir", DEFAULT_OUTPUT_DIR))

    # --------------------------------------------------------------- helpers
    def _make_som(self, **overrides) -> SOM:
        rows, cols = _resolve_grid_size(self.config)
        params = dict(
            n_features=self.data.shape[1],
            grid_rows=rows,
            grid_cols=cols,
            topology=self.config.get("topology", "rectangular"),
            init_method=self.config.get("init_method", "sample"),
            distance=self.config.get("distance", "euclidean"),
            neighborhood=self.config.get("neighborhood", "gaussian"),
            learning_rate=self.config.get("learning_rate", 0.5),
            radius=self.config.get("radius"),
            epochs=int(self.config.get("epochs", 500)),
            seed=self.config.get("seed"),
        )
        params.update(overrides)
        return SOM(**params)

    def _save_fig(self, fig: plt.Figure, name: str) -> None:
        path = self.output_dir / name
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path.relative_to(PROJECT_ROOT)}")

    # --------------------------------------------------------------- entry
    def run(self):
        print("Iniciando Experimento de Kohonen...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = self._make_som().fit(self.data)
        final = self.model.history[-1]
        print(
            f"Trained {self.model.grid_rows}x{self.model.grid_cols} "
            f"{self.model.topology} grid ({self.model.epochs} epochs, "
            f"distance={self.model.distance_name}, init={self.model.init_method}). "
            f"Final QE: {final['quantization_error']:.4f}"
        )
        if self.labels is not None:
            groups = self.model.assignments(self.data, self.labels)
            print("\nCountry assignments per neuron:")
            for (r, c), members in sorted(groups.items()):
                print(f"  ({r},{c}): {', '.join(map(str, members))}")

        self.analyze()

    def analyze(self):
        print("\nAnalizando resultados...")
        self._baseline_plots()
        summary: dict[str, list[dict]] = {}
        summary["grid_size"] = self._sweep_grid_size()
        summary["topology_init"] = self._sweep_topology_init()
        summary["distance"] = self._sweep_distance()
        self._sweep_lr_schedule()
        self._sweep_radius_schedule()
        self._sweep_epochs()
        self._sweep_seeds()
        self._write_summary(summary)
        print(
            "\nDone. Plots and summary saved to "
            f"{self.output_dir.relative_to(PROJECT_ROOT)}/"
        )

    # --------------------------------------------------------------- baseline
    def _baseline_plots(self):
        print("== Baseline plots ==")
        som = self.model
        assert som is not None, "Model must be trained before analyzing"

        fig, ax = plt.subplots(figsize=(9, 8))
        plot_country_labels(som, self.data, self.labels, ax=ax)
        self._save_fig(fig, "01_country_map.png")

        fig, ax = plt.subplots(figsize=(7, 6))
        plot_u_matrix(som, ax=ax)
        self._save_fig(fig, "02_u_matrix.png")

        fig, ax = plt.subplots(figsize=(7, 6))
        plot_hit_map(som, self.data, ax=ax)
        self._save_fig(fig, "03_hit_map.png")

        fig = plot_component_planes(som, FEATURES, ncols=4)
        self._save_fig(fig, "04_component_planes.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_quantization_error({"baseline": som.history}, ax=ax)
        self._save_fig(fig, "05_quantization_error.png")

    # --------------------------------------------------------------- sweeps
    def _sweep_grid_size(self) -> list[dict]:
        print("== Sweep: grid size ==")
        sizes = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
        seeds = list(range(5))
        rows_data = []
        for r, c in sizes:
            qes = []
            for seed in seeds:
                som = self._make_som(grid_rows=r, grid_cols=c, seed=seed).fit(self.data)
                qes.append(som.history[-1]["quantization_error"])
            rows_data.append(
                {
                    "grid": f"{r}x{c}",
                    "mean_qe": float(np.mean(qes)),
                    "std_qe": float(np.std(qes)),
                }
            )
            print(f"  {r}x{c}: QE = {np.mean(qes):.4f} +/- {np.std(qes):.4f}")

        fig, ax = plt.subplots(figsize=(7, 5))
        xs = [row["grid"] for row in rows_data]
        means = [row["mean_qe"] for row in rows_data]
        stds = [row["std_qe"] for row in rows_data]
        ax.bar(xs, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
        ax.set_xlabel("grid size")
        ax.set_ylabel("quantization error (final, mean over 5 seeds)")
        ax.set_title("Effect of grid size on quantization error")
        ax.grid(True, axis="y", alpha=0.3)
        self._save_fig(fig, "10_sweep_grid_size.png")
        return rows_data

    def _sweep_topology_init(self) -> list[dict]:
        print("== Sweep: topology x init_method ==")
        rows_data = []
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        combos = [
            ("rectangular", "sample"),
            ("rectangular", "random"),
            ("hexagonal", "sample"),
            ("hexagonal", "random"),
        ]
        for ax, (topo, init) in zip(axes.ravel(), combos):
            som = self._make_som(topology=topo, init_method=init, seed=42).fit(
                self.data
            )
            qe = som.history[-1]["quantization_error"]
            plot_country_labels(
                som,
                self.data,
                self.labels,
                ax=ax,
                title=f"{topo} / init={init}  QE={qe:.3f}",
            )
            rows_data.append({"topology": topo, "init": init, "qe": float(qe)})
        fig.suptitle("Topology x initialization", fontsize=14)
        fig.tight_layout()
        self._save_fig(fig, "11_sweep_topology_init.png")

        print("== Sweep: init convergence (random vs sample) ==")
        histories = {}
        for init in ["random", "sample"]:
            per_seed = []
            for seed in range(10):
                som = self._make_som(init_method=init, seed=seed, epochs=300).fit(
                    self.data
                )
                per_seed.append([h["quantization_error"] for h in som.history])
            avg = np.mean(per_seed, axis=0)
            histories[init] = [
                {"epoch": i, "quantization_error": v} for i, v in enumerate(avg)
            ]
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_quantization_error(
            histories,
            ax=ax,
            title="Convergence: random vs sample init (mean over 10 seeds)",
        )
        self._save_fig(fig, "12_sweep_init_convergence.png")
        return rows_data

    def _sweep_distance(self) -> list[dict]:
        print("== Sweep: distance metric ==")
        rows_data = []
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, dist_name in zip(axes, ["euclidean", "exponential"]):
            som = self._make_som(distance=dist_name, seed=42).fit(self.data)
            qe = som.history[-1]["quantization_error"]
            plot_country_labels(
                som,
                self.data,
                self.labels,
                ax=ax,
                title=f"distance={dist_name}  QE={qe:.3f}",
            )
            rows_data.append({"distance": dist_name, "qe": float(qe)})
        fig.suptitle("Distance metric for BMU selection", fontsize=14)
        fig.tight_layout()
        self._save_fig(fig, "20_sweep_distance.png")
        return rows_data

    def _sweep_lr_schedule(self) -> None:
        print("== Sweep: learning-rate schedule ==")
        specs = {
            "constant 0.1": 0.1,
            "constant 0.5": 0.5,
            "linear 0.5->0.01": {"kind": "linear", "initial": 0.5, "final": 0.01},
            "exponential 0.5->0.01": {
                "kind": "exponential",
                "initial": 0.5,
                "final": 0.01,
            },
            "inverse 0.5->0.01": {"kind": "inverse", "initial": 0.5, "final": 0.01},
        }
        histories = {}
        sched_curves = {}
        for label, spec in specs.items():
            som = self._make_som(learning_rate=spec, epochs=400, seed=42).fit(self.data)
            histories[label] = som.history
            sched = schedules.parse(spec)
            sched_curves[label] = [sched(t, 400) for t in range(400)]
            print(f"  {label}: QE = {som.history[-1]['quantization_error']:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_schedule_curves(
            sched_curves,
            ax=axes[0],
            ylabel="learning rate",
            title="Learning-rate schedules",
        )
        plot_quantization_error(
            histories, ax=axes[1], title="QE vs epoch by LR schedule"
        )
        fig.tight_layout()
        self._save_fig(fig, "30_sweep_lr_schedule.png")

    def _sweep_radius_schedule(self) -> None:
        print("== Sweep: radius schedule ==")
        specs = {
            "constant 1": 1.0,
            "constant 3": 3.0,
            "linear 3->1": {"kind": "linear", "initial": 3.0, "final": 1.0},
            "exponential 3->1": {"kind": "exponential", "initial": 3.0, "final": 1.0},
            "exponential 3->0.5": {"kind": "exponential", "initial": 3.0, "final": 0.5},
        }
        histories = {}
        sched_curves = {}
        for label, spec in specs.items():
            som = self._make_som(radius=spec, epochs=400, seed=42).fit(self.data)
            histories[label] = som.history
            sched = schedules.parse(spec)
            sched_curves[label] = [sched(t, 400) for t in range(400)]
            print(f"  {label}: QE = {som.history[-1]['quantization_error']:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_schedule_curves(
            sched_curves,
            ax=axes[0],
            ylabel="neighborhood radius",
            title="Radius schedules",
        )
        plot_quantization_error(
            histories, ax=axes[1], title="QE vs epoch by radius schedule"
        )
        fig.tight_layout()
        self._save_fig(fig, "31_sweep_radius_schedule.png")

    def _sweep_epochs(self) -> None:
        print("== Sweep: epochs ==")
        epochs_options = [50, 100, 250, 500, 1000, 2000]
        results = []
        for ep in epochs_options:
            per_seed = []
            for seed in range(5):
                som = self._make_som(epochs=ep, seed=seed).fit(self.data)
                per_seed.append(som.history[-1]["quantization_error"])
            results.append((np.mean(per_seed), np.std(per_seed)))
            print(f"  epochs={ep}: QE = {results[-1][0]:.4f} +/- {results[-1][1]:.4f}")
        means = [m for m, _ in results]
        stds = [s for _, s in results]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(
            epochs_options, means, yerr=stds, marker="o", capsize=4, linewidth=1.5
        )
        ax.set_xscale("log")
        ax.set_xlabel("epochs")
        ax.set_ylabel("quantization error (final)")
        ax.set_title("Effect of training length")
        ax.grid(True, alpha=0.3)
        self._save_fig(fig, "40_sweep_epochs.png")

    def _sweep_seeds(self) -> None:
        print("== Sweep: seed stability ==")
        qes = []
        for seed in range(20):
            som = self._make_som(seed=seed).fit(self.data)
            qes.append(som.history[-1]["quantization_error"])
        print(
            f"  mean={np.mean(qes):.4f}  std={np.std(qes):.4f}  "
            f"min={min(qes):.4f}  max={max(qes):.4f}"
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(qes, bins=10, color="darkorange", edgecolor="black")
        ax.axvline(
            np.mean(qes),
            color="black",
            linestyle="--",
            label=f"mean={np.mean(qes):.3f}",
        )
        ax.set_xlabel("final quantization error")
        ax.set_ylabel("count")
        ax.set_title("Seed stability (20 random seeds, baseline config)")
        ax.legend()
        self._save_fig(fig, "50_seed_stability.png")

    # --------------------------------------------------------------- summary
    def _write_summary(self, rows: dict[str, list[dict]]) -> None:
        metric_keys = ("qe", "mean_qe")
        path = self.output_dir / "summary.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sweep", "params", "metric", "value"])
            for sweep_name, entries in rows.items():
                for entry in entries:
                    value = next((entry[k] for k in metric_keys if k in entry), None)
                    params = ";".join(
                        f"{k}={v}" for k, v in entry.items() if k not in metric_keys
                    )
                    writer.writerow([sweep_name, params, "qe", value])
        print(f"  wrote {path.relative_to(PROJECT_ROOT)}")
