"""Barrido completo: dimensión × p patrones → ortogonalidad → mejor grupo → recall."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hopfield.hopfield_net import HopfieldNet
from hopfield.orthogonality import (
    combinations_summary,
    full_dot_matrix,
    pairwise_dot_matrix,
    select_best_group,
)
from hopfield.patterns import add_noise, flatten, match_stored, resolve_letters_for_shape
from hopfield.plots import (
    plot_combinations_scatter,
    plot_dot_matrix,
    plot_energy,
    plot_letters_grid,
    plot_recall_steps,
)


def _dim_label(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def run_recall_for_group(
    letters: dict[str, np.ndarray],
    group: tuple[str, ...],
    shape: tuple[int, int],
    *,
    n_flips: int,
    trials: int,
    max_steps: int,
    async_update: bool,
    seed: int,
    out_dir: Path,
    tag: str,
) -> pd.DataFrame:
    """Entrena con ``group``, recall ruidoso y guarda figuras + CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    flats = {n: flatten(letters[n]) for n in group}
    net = HopfieldNet(n_neurons=shape[0] * shape[1], seed=seed)
    net.train(np.stack([flats[n] for n in group]))

    rows = []
    for name in group:
        target = flats[name]
        success = 0
        history = []
        energies = []
        for trial in range(trials):
            rng = np.random.default_rng(seed + hash((tag, name, trial)) % 10_000_000)
            query = add_noise(target, n_flips=n_flips, rng=rng)
            result = net.recall(
                query,
                max_steps=max_steps,
                async_update=async_update,
                record_history=(trial == 0),
                rng=rng,
            )
            rec, kind = match_stored(result.final_state, flats)
            if rec == name and kind == "pattern":
                success += 1
            if trial == 0:
                history = result.history
                energies = result.energies

        rows.append(
            {
                "letter": name,
                "n_flips": n_flips,
                "trials": trials,
                "success_rate": success / trials,
            }
        )
        fig = plot_recall_steps(
            history,
            shape,
            title=f"{tag} {name}: {n_flips} flips",
            target=target,
        )
        fig.savefig(out_dir / f"{name}_steps.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        fig = plot_energy(energies, title=f"{tag} {name}")
        fig.savefig(out_dir / f"{name}_energy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "recall_summary.csv", index=False)
    return df


def run_complete_sweep(
    *,
    base_letters_path: Path,
    output_dir: Path,
    dimensions: list[dict],
    p_values: list[int],
    noise_cfg: dict,
    max_steps: int,
    async_update: bool,
    seed: int,
    compare_fixed: list[str] | None = None,
) -> pd.DataFrame:
    """Ejecuta el barrido dimensión × p y devuelve tabla resumen."""
    n_flips = int(noise_cfg.get("n_flips", 3))
    trials = max(1, int(noise_cfg.get("trials", 20)))
    summary_rows: list[dict] = []

    for dim_spec in dimensions:
        shape = tuple(int(x) for x in dim_spec["shape"])
        dim_path = dim_spec.get("letters_path")
        letters = resolve_letters_for_shape(
            base_letters_path,
            shape,
            letters_path=dim_path,
        )
        label = _dim_label(shape)
        dim_out = output_dir / label
        dim_out.mkdir(parents=True, exist_ok=True)

        fig = plot_letters_grid(letters)
        fig.suptitle(f"Alfabeto {label} ({len(letters)} letras)", y=1.02)
        fig.savefig(dim_out / "letters_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        M, names = full_dot_matrix(letters)
        fig = plot_dot_matrix(M, names, title=f"PI global {label}")
        fig.savefig(dim_out / "dot_product_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        for p in p_values:
            if p > len(letters):
                print(f"  [{label}] p={p}: omitido (alfabeto tiene {len(letters)} letras)")
                continue

            p_out = dim_out / f"p{p}"
            p_out.mkdir(parents=True, exist_ok=True)
            print(f"  [{label}] p={p}: C({len(letters)},{p})...")

            df = combinations_summary(letters, k=p)
            df_export = df.copy()
            df_export["grupo"] = df_export["grupo"].apply(
                lambda t: "".join(t) if isinstance(t, tuple) else t
            )
            df_export.sort_values("pi_medio").to_csv(
                p_out / "combinations_all.csv", index=False
            )
            df_export.sort_values("pi_medio").head(20).to_csv(
                p_out / "combinations_top_avg_low.csv", index=False
            )

            fig = plot_combinations_scatter(
                df_export, title=f"PI medio vs max ({label}, p={p})"
            )
            fig.savefig(p_out / "combinations_scatter.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            best = select_best_group(df)
            worst = tuple(
                df.sort_values(["pi_medio", "pi_max", "count_max"], ascending=False)
                .iloc[0]["grupo"]
            )
            best_row = df[df["grupo"].apply(lambda g: tuple(g) == best)].iloc[0]
            worst_row = df[df["grupo"].apply(lambda g: tuple(g) == worst)].iloc[0]

            recall_rates: dict[str, float] = {}
            for tag, group in [("best", best), ("worst", worst)]:
                gdir = p_out / tag / "".join(group)
                gdir.mkdir(parents=True, exist_ok=True)
                M_g = pairwise_dot_matrix({n: letters[n] for n in group})
                fig = plot_dot_matrix(
                    M_g, list(group), title=f"{tag}: {''.join(group)}"
                )
                fig.savefig(gdir / "dot_matrix.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                recall_df = run_recall_for_group(
                    letters,
                    group,
                    shape,
                    n_flips=n_flips,
                    trials=trials,
                    max_steps=max_steps,
                    async_update=async_update,
                    seed=seed,
                    out_dir=gdir / "recall",
                    tag=tag,
                )
                recall_rates[tag] = float(recall_df["success_rate"].mean())

            fixed_success = None
            fixed_group: tuple[str, ...] | None = None
            if compare_fixed is not None and len(compare_fixed) == p:
                fixed_group = tuple(compare_fixed)
                if all(n in letters for n in fixed_group):
                    gdir = p_out / "fixed" / "".join(fixed_group)
                    gdir.mkdir(parents=True, exist_ok=True)
                    recall_df = run_recall_for_group(
                        letters,
                        fixed_group,
                        shape,
                        n_flips=n_flips,
                        trials=trials,
                        max_steps=max_steps,
                        async_update=async_update,
                        seed=seed,
                        out_dir=gdir / "recall",
                        tag="fixed",
                    )
                    fixed_success = float(recall_df["success_rate"].mean())

            summary_rows.append(
                {
                    "dimension": label,
                    "p": p,
                    "n_letters": len(letters),
                    "n_combinations": len(df),
                    "best_group": "".join(best),
                    "best_pi_medio": float(best_row["pi_medio"]),
                    "best_pi_max": float(best_row["pi_max"]),
                    "best_count_max": int(best_row["count_max"]),
                    "best_recall_mean": recall_rates["best"],
                    "worst_group": "".join(worst),
                    "worst_pi_medio": float(worst_row["pi_medio"]),
                    "worst_recall_mean": recall_rates["worst"],
                    "fixed_group": "".join(fixed_group) if fixed_group else "",
                    "fixed_recall_mean": fixed_success,
                    "n_flips": n_flips,
                    "trials": trials,
                }
            )
            print(
                f"    mejor={''.join(best)} PI={best_row['pi_medio']:.2f} "
                f"recall={recall_rates['best']:.0%} | peor={''.join(worst)} "
                f"PI={worst_row['pi_medio']:.2f} recall={recall_rates['worst']:.0%}"
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "sweep_summary.csv", index=False)
    return summary
