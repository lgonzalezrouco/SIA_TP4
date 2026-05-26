"""Orchestrator for the Hopfield exercise.

Runs (depending on the ``analyses`` flags):
 1. Ortogonality analysis over every C(n, k) combination of letters.
 2. Step-by-step recall on a chosen group of stored patterns.
 3. Spurious-state hunting from very noisy queries.
 4. Single-pixel robustness check.
 5. Capacity heatmap (recall success vs ``p`` and noise level).
"""

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
from hopfield.patterns import (
    add_noise,
    closest_stored,
    flatten,
    load_letters,
    match_stored,
)
from hopfield.plots import (
    plot_capacity_heatmap,
    plot_combinations_scatter,
    plot_dot_matrix,
    plot_energy,
    plot_letters_grid,
    plot_recall_steps,
    plot_spurious_examples,
)
from hopfield.sweep import run_complete_sweep

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "hopfield"
DEFAULT_LETTERS_PATH = PROJECT_ROOT / "data" / "letters.json"


class HopfieldExperiment:
    def __init__(self, config):
        self.config = config or {}
        self.seed = int(self.config.get("seed", 42))
        self.rng = np.random.default_rng(self.seed)

        shape = self.config.get("shape", [5, 5])
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_neurons = self.shape[0] * self.shape[1]

        letters_path = Path(self.config.get("letters_path", DEFAULT_LETTERS_PATH))
        if not letters_path.is_absolute():
            letters_path = PROJECT_ROOT / letters_path
        self.letters, file_shape = load_letters(letters_path, expected_shape=self.shape)

        self.group_names = list(self.config.get("patterns", ["A", "B", "C", "D"]))
        missing = [n for n in self.group_names if n not in self.letters]
        if missing:
            raise ValueError(f"patterns not found in letters file: {missing}")

        self.k = int(self.config.get("combo_k", len(self.group_names)))

        self.max_steps = int(self.config.get("max_steps", 50))
        self.async_update = bool(self.config.get("async_update", True))

        self.noise_cfg = self.config.get("noise", {"n_flips": 3, "trials": 1})
        self.spurious_cfg = self.config.get(
            "spurious", {"n_flips": 13, "trials": 100}
        )
        self.capacity_cfg = self.config.get(
            "capacity",
            {
                "p_values": [2, 3, 4, 5, 6, 7, 8],
                "noise_flips": [1, 2, 3, 5, 8, 12],
                "trials": 30,
            },
        )

        analyses = self.config.get("analyses", {})
        self.do_ppt = bool(analyses.get("ppt_bundle", False))
        self.do_sweep = bool(analyses.get("complete_sweep", False))
        self.do_ortho = bool(analyses.get("orthogonality", True))
        self.do_recall = bool(analyses.get("recall", True))
        self.do_spurious = bool(analyses.get("spurious", True))
        self.do_single = bool(analyses.get("single_pixel", True))
        self.do_capacity = bool(analyses.get("capacity", False))

        sweep_cfg = self.config.get("sweep", {})
        self.sweep_dimensions = sweep_cfg.get(
            "dimensions",
            [{"shape": list(self.shape)}],
        )
        self.sweep_p_values = list(sweep_cfg.get("p_values", [2, 3, 4, 5, 6]))
        self.sweep_compare_fixed = sweep_cfg.get("compare_fixed")
        if self.sweep_compare_fixed is None and self.group_names:
            self.sweep_compare_fixed = self.group_names

        self.output_dir = Path(
            self.config.get("output_dir", DEFAULT_OUTPUT_DIR)
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- run
    def run(self) -> None:
        print("Iniciando Experimento de Hopfield...")
        print(
            f"  shape={self.shape}, n_neurons={self.n_neurons}, "
            f"alphabet={len(self.letters)} letters, group={self.group_names}"
        )
        if self.do_sweep:
            self._run_complete_sweep()
            print(f"Resultados guardados en {self.output_dir}")
            return

        if self.do_ppt:
            self._run_ppt_bundle()
            print(f"Resultados PPT guardados en {self.output_dir}")
            return

        self._save_alphabet_overview()

        if self.do_ortho:
            self._run_orthogonality()
        if self.do_recall:
            self._run_recall()
        if self.do_spurious:
            self._run_spurious()
        if self.do_single:
            self._run_single_pixel()
        if self.do_capacity:
            self._run_capacity()
        print(f"Resultados guardados en {self.output_dir}")

    def _run_complete_sweep(self) -> None:
        print("Barrido completo (dimension x p -> ortogonalidad -> mejor/peor grupo -> recall)...")
        letters_path = Path(self.config.get("letters_path", DEFAULT_LETTERS_PATH))
        if not letters_path.is_absolute():
            letters_path = PROJECT_ROOT / letters_path

        summary = run_complete_sweep(
            base_letters_path=letters_path,
            output_dir=self.output_dir,
            dimensions=self.sweep_dimensions,
            p_values=self.sweep_p_values,
            noise_cfg=self.noise_cfg,
            max_steps=self.max_steps,
            async_update=self.async_update,
            seed=self.seed,
            compare_fixed=self.sweep_compare_fixed,
        )
        print("\nResumen del barrido:")
        print(summary.to_string(index=False))

    def _run_ppt_bundle(self) -> None:
        """Solo outputs utiles para la presentacion (5x5, grupo ALTV por defecto)."""
        print("Modo PPT: generando figuras y tablas para la presentacion...")

        # --- 1) Alfabeto
        self._save_alphabet_overview()

        # --- 2) Ortogonalidad C(n,4)
        ortho_dir = self.output_dir / "orthogonality"
        ortho_dir.mkdir(exist_ok=True)
        k = len(self.group_names)
        print(f"  [1/5] Ortogonalidad C({len(self.letters)},{k})...")
        df = combinations_summary(self.letters, k=k)
        df_str = df.copy()
        df_str["grupo"] = df_str["grupo"].apply(
            lambda t: "".join(t) if isinstance(t, tuple) else t
        )
        by_mean = df_str.sort_values("pi_medio").reset_index(drop=True)
        by_mean.head(15).to_csv(ortho_dir / "combinations_top_avg_low.csv", index=False)
        by_mean.tail(5).iloc[::-1].to_csv(
            ortho_dir / "combinations_top_avg_high.csv", index=False
        )
        fig = plot_combinations_scatter(
            df_str, title=f"PI medio vs max (p={k})"
        )
        fig.savefig(ortho_dir / "combinations_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        best = select_best_group(df)
        worst = tuple(
            df.sort_values(["pi_medio", "pi_max", "count_max"], ascending=False)
            .iloc[0]["grupo"]
        )
        best_label = "".join(best)
        worst_label = "".join(worst)
        best_row = df[df["grupo"].apply(lambda g: tuple(g) == best)].iloc[0]
        print(
            f"    mejor={best_label} (PI={best_row['pi_medio']:.2f})  "
            f"peor={worst_label}"
        )

        M = pairwise_dot_matrix({n: self.letters[n] for n in self.group_names})
        fig = plot_dot_matrix(
            M,
            self.group_names,
            title=f"Grupo almacenado: {''.join(self.group_names)}",
        )
        fig.savefig(ortho_dir / "group_stored_dot_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- 3) Recall grupo elegido (config patterns)
        print(f"  [2/5] Recall ruidoso ({''.join(self.group_names)})...")
        self._run_recall_to_dir(self.output_dir / "recall", self.group_names)

        # --- 4) Contraste peor grupo
        print(f"  [3/5] Contraste peor grupo ({worst_label})...")
        contrast_dir = self.output_dir / "contrast_worst"
        contrast_dir.mkdir(exist_ok=True)
        M_w = pairwise_dot_matrix({n: self.letters[n] for n in worst})
        fig = plot_dot_matrix(M_w, list(worst), title=f"Peor grupo: {worst_label}")
        fig.savefig(contrast_dir / "dot_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        self._run_recall_to_dir(contrast_dir / "recall", list(worst))

        # --- 5) Espurios + 1 pixel
        print("  [4/5] Estados espurios...")
        self._run_spurious()
        print("  [5/5] Robustez 1 pixel...")
        self._run_single_pixel()

        # --- Tabla ortogonalidad vs p (solo metricas, sin recall)
        p_values = self.config.get("ppt", {}).get(
            "p_values", [2, 3, 4, 5, 6]
        )
        rows = []
        for p in p_values:
            if p > len(self.letters):
                continue
            dfp = combinations_summary(self.letters, k=p)
            bg = select_best_group(dfp)
            wg = tuple(
                dfp.sort_values(["pi_medio", "pi_max", "count_max"], ascending=False)
                .iloc[0]["grupo"]
            )
            br = dfp[dfp["grupo"].apply(lambda g: tuple(g) == bg)].iloc[0]
            rows.append(
                {
                    "p": p,
                    "best_group": "".join(bg),
                    "best_pi_medio": br["pi_medio"],
                    "worst_group": "".join(wg),
                }
            )
        pd.DataFrame(rows).to_csv(self.output_dir / "orthogonality_vs_p.csv", index=False)

        # Indice para la presentacion
        lines = [
            "# Archivos para PowerPoint (Hopfield)",
            "",
            "## Patrones y modelo",
            "- letters_grid.png — alfabeto 5x5",
            "- dot_product_matrix.png — PI entre todas las letras",
            "",
            "## Ortogonalidad (p=4)",
            "- orthogonality/combinations_scatter.png",
            "- orthogonality/combinations_top_avg_low.csv — mejores grupos",
            "- orthogonality/combinations_top_avg_high.csv — peores grupos",
            "- orthogonality/group_stored_dot_matrix.png — matriz del grupo ALTV",
            "- orthogonality_vs_p.csv — mejor grupo segun p",
            "",
            "## Recall con ruido (3 flips)",
            f"- recall/{{letra}}_steps.png, recall/{{letra}}_energy.png",
            "- recall/summary.csv",
            "",
            "## Contraste (peor grupo)",
            "- contrast_worst/dot_matrix.png",
            "- contrast_worst/recall/ — recall del peor grupo",
            "",
            "## Extras clase",
            "- spurious/examples.png, spurious/summary.csv",
            "- single_pixel/results.csv",
            "",
            f"Grupo almacenado: {''.join(self.group_names)}",
            f"Mejor ortogonalidad p=4: {best_label} (PI medio={best_row['pi_medio']:.2f})",
            f"Peor ortogonalidad p=4: {worst_label}",
        ]
        (self.output_dir / "README_PPT.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def _run_recall_to_dir(
        self, recall_dir: Path, group: list[str]
    ) -> None:
        """Recall ruidoso paso a paso para un grupo; guarda en ``recall_dir``."""
        recall_dir.mkdir(parents=True, exist_ok=True)
        net, flats = self._train_group(group)
        n_flips = int(self.noise_cfg.get("n_flips", 3))
        trials = max(1, int(self.noise_cfg.get("trials", 1)))
        rows = []
        for name in group:
            target = flats[name]
            success = 0
            best_history: list[np.ndarray] = []
            best_energies: list[float] = []
            for trial in range(trials):
                rng = np.random.default_rng(
                    self.seed + hash((name, trial, str(recall_dir))) % 10_000_000
                )
                query = add_noise(target, n_flips=n_flips, rng=rng)
                result = net.recall(
                    query,
                    max_steps=self.max_steps,
                    async_update=self.async_update,
                    record_history=(trial == 0),
                    rng=rng,
                )
                rec_name, kind = match_stored(result.final_state, flats)
                if rec_name == name and kind == "pattern":
                    success += 1
                if trial == 0:
                    best_history = result.history
                    best_energies = result.energies
            rows.append(
                {
                    "letter": name,
                    "n_flips": n_flips,
                    "trials": trials,
                    "success_rate": success / trials,
                }
            )
            fig = plot_recall_steps(
                best_history,
                self.shape,
                title=f"{name}: {n_flips} bits invertidos",
                target=target,
            )
            fig.savefig(recall_dir / f"{name}_steps.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            fig = plot_energy(best_energies, title=f"Energia — {name}")
            fig.savefig(recall_dir / f"{name}_energy.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        pd.DataFrame(rows).to_csv(recall_dir / "summary.csv", index=False)

    # ---------------------------------------------------------------- helpers
    def _train_group(self, names: list[str]) -> tuple[HopfieldNet, dict[str, np.ndarray]]:
        flats = {n: flatten(self.letters[n]) for n in names}
        net = HopfieldNet(n_neurons=self.n_neurons, seed=self.seed)
        net.train(np.stack([flats[n] for n in names]))
        return net, flats

    def _save_alphabet_overview(self) -> None:
        fig = plot_letters_grid(self.letters)
        fig.suptitle(f"Alfabeto almacenado ({len(self.letters)} letras {self.shape[0]}x{self.shape[1]})", y=1.02)
        fig.savefig(self.output_dir / "letters_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        M, names = full_dot_matrix(self.letters)
        fig = plot_dot_matrix(M, names, title="Producto interno entre todas las letras")
        fig.savefig(self.output_dir / "dot_product_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------------------------------------------------------------- (1) ortho
    def _run_orthogonality(self) -> None:
        print(f"[1/5] Ortogonalidad sobre C({len(self.letters)},{self.k}) combinaciones...")
        df = combinations_summary(self.letters, k=self.k)
        df_str = df.copy()
        df_str["grupo"] = df_str["grupo"].apply(lambda t: "".join(t))

        by_mean = df_str.sort_values("pi_medio").reset_index(drop=True)
        by_max = df_str.sort_values(["pi_max", "count_max"]).reset_index(drop=True)
        by_mean.head(20).to_csv(self.output_dir / "combinations_top_avg_low.csv", index=False)
        by_mean.tail(20).iloc[::-1].to_csv(
            self.output_dir / "combinations_top_avg_high.csv", index=False
        )
        by_max.head(20).to_csv(self.output_dir / "combinations_top_max_low.csv", index=False)
        df_str.to_csv(self.output_dir / "combinations_all.csv", index=False)

        print("    mejor grupo (menor PI medio):", by_mean.iloc[0].to_dict())
        print("    peor grupo  (mayor PI medio):", by_mean.iloc[-1].to_dict())

        fig = plot_combinations_scatter(df_str)
        fig.savefig(
            self.output_dir / "combinations_scatter.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        # also dump the orthogonality matrix of the CHOSEN group
        M = pairwise_dot_matrix({n: self.letters[n] for n in self.group_names})
        fig = plot_dot_matrix(M, self.group_names, title=f"Grupo elegido: {''.join(self.group_names)}")
        fig.savefig(self.output_dir / "group_dot_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------------------------------------------------------------- (2) recall
    def _run_recall(self) -> None:
        print("[2/5] Recall paso a paso del grupo elegido...")
        self._run_recall_to_dir(self.output_dir / "recall", self.group_names)
        df = pd.read_csv(self.output_dir / "recall" / "summary.csv")
        print(df.to_string(index=False))

    # ---------------------------------------------------------------- (3) spurious
    def _run_spurious(self) -> None:
        print("[3/5] Caza de estados espurios...")
        sp_dir = self.output_dir / "spurious"
        sp_dir.mkdir(exist_ok=True)

        net, flats = self._train_group(self.group_names)
        n_flips = int(self.spurious_cfg.get("n_flips", self.n_neurons // 2))
        trials = int(self.spurious_cfg.get("trials", 100))

        counts = {"pattern": 0, "negative": 0, "spurious": 0}
        per_source: dict[str, dict[str, int]] = {}
        examples: list[dict] = []

        for trial in range(trials):
            source = self.group_names[trial % len(self.group_names)]
            rng = np.random.default_rng(self.seed + 1000 + trial)
            target = flats[source]
            query = add_noise(target, n_flips=n_flips, rng=rng)
            result = net.recall(
                query,
                max_steps=self.max_steps,
                async_update=self.async_update,
                record_history=False,
                rng=rng,
            )
            name, kind = match_stored(result.final_state, flats)
            counts[kind] += 1
            per_source.setdefault(source, {"pattern": 0, "negative": 0, "spurious": 0})
            per_source[source][kind] += 1
            if kind == "spurious" and len(examples) < 8:
                examples.append(
                    {
                        "query": query.copy(),
                        "final": result.final_state.copy(),
                        "source": source,
                    }
                )

        summary = {
            "trials": trials,
            "n_flips": n_flips,
            "pattern_rate": counts["pattern"] / trials,
            "negative_rate": counts["negative"] / trials,
            "spurious_rate": counts["spurious"] / trials,
        }
        pd.DataFrame([summary]).to_csv(sp_dir / "summary.csv", index=False)
        rows = [
            {"source": s, **c, "spurious_rate": c["spurious"] / sum(c.values())}
            for s, c in per_source.items()
        ]
        pd.DataFrame(rows).to_csv(sp_dir / "per_source.csv", index=False)
        print("    counts:", counts)

        if examples:
            fig = plot_spurious_examples(examples, self.shape)
            fig.savefig(sp_dir / "examples.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            print("    no se encontraron estados espurios con la configuración actual.")

    # ---------------------------------------------------------------- (4) single pixel
    def _run_single_pixel(self) -> None:
        print("[4/5] Robustez ante 1 píxel invertido...")
        sp_dir = self.output_dir / "single_pixel"
        sp_dir.mkdir(exist_ok=True)

        net, flats = self._train_group(self.group_names)

        rows = []
        for name in self.group_names:
            target = flats[name]
            outcomes = {"pattern": 0, "negative": 0, "spurious": 0, "other_pattern": 0}
            mismatches = []
            for i in range(self.n_neurons):
                q = target.copy()
                q[i] *= -1
                rng = np.random.default_rng(self.seed + 50000 + i)
                result = net.recall(
                    q,
                    max_steps=self.max_steps,
                    async_update=self.async_update,
                    record_history=False,
                    rng=rng,
                )
                rec, kind = match_stored(result.final_state, flats)
                if kind == "pattern":
                    if rec == name:
                        outcomes["pattern"] += 1
                    else:
                        outcomes["other_pattern"] += 1
                        mismatches.append((i, rec))
                elif kind == "negative":
                    outcomes["negative"] += 1
                else:
                    outcomes["spurious"] += 1
            rows.append(
                {
                    "letter": name,
                    "correct": outcomes["pattern"],
                    "other_pattern": outcomes["other_pattern"],
                    "negative": outcomes["negative"],
                    "spurious": outcomes["spurious"],
                    "n_pixels": self.n_neurons,
                    "accuracy": outcomes["pattern"] / self.n_neurons,
                    "collisions": ";".join(
                        f"p{i}->{r}" for i, r in mismatches
                    )
                    if mismatches
                    else "",
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(sp_dir / "results.csv", index=False)
        print(df.drop(columns=["collisions"]).to_string(index=False))

    # ---------------------------------------------------------------- (5) capacity
    def _run_capacity(self) -> None:
        print("[5/5] Análisis de capacidad...")
        p_values = list(self.capacity_cfg.get("p_values", [2, 3, 4, 5]))
        noise_values = list(self.capacity_cfg.get("noise_flips", [1, 2, 3]))
        trials = int(self.capacity_cfg.get("trials", 30))

        all_names = list(self.letters.keys())
        success = np.zeros((len(p_values), len(noise_values)))

        for i, p in enumerate(p_values):
            if p > len(all_names):
                continue
            chosen = all_names[:p]
            net, flats = self._train_group(chosen)
            for j, n_flips in enumerate(noise_values):
                ok = 0
                total = 0
                for trial in range(trials):
                    name = chosen[trial % p]
                    rng = np.random.default_rng(self.seed + 9000 + i * 1000 + j * 100 + trial)
                    q = add_noise(flats[name], n_flips=n_flips, rng=rng)
                    res = net.recall(
                        q,
                        max_steps=self.max_steps,
                        async_update=self.async_update,
                        record_history=False,
                        rng=rng,
                    )
                    rec, kind = match_stored(res.final_state, flats)
                    if rec == name and kind == "pattern":
                        ok += 1
                    total += 1
                success[i, j] = ok / max(1, total)

        fig = plot_capacity_heatmap(success, p_values, noise_values)
        fig.savefig(self.output_dir / "capacity_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(success, index=p_values, columns=noise_values).to_csv(
            self.output_dir / "capacity.csv"
        )
        print("    success rates:")
        print(success)
