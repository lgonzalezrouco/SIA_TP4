import os

import numpy as np

from oja.oja_neuron import OjaNeuron
from oja.pca_analysis import PCAAnalyzer
from oja.plots import (
    plot_convergence,
    plot_loadings_comparison,
    plot_projections_comparison,
    plot_sweep_results,
    plot_table,
)
from utils.data_loader import preprocess_data


class OjaExperiment:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.output_dir = "results/oja"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        print("Iniciando Experimento de Oja y PCA...")

        # 1. Ejecutar PCA con sklearn (Ground Truth)
        pca_analyzer = PCAAnalyzer(self.data, output_dir=self.output_dir)
        pca_analyzer.run()
        pca_loadings = pca_analyzer.pca.components_[0]
        feature_names = pca_analyzer.feature_names

        # 2. Ejecutar barrido de hiperparámetros
        best_config = self.run_sweep(pca_loadings)

        # 3. Entrenar modelo de Oja con la mejor configuración
        print(
            f"\nEntrenando Regla de Oja con la mejor configuración ({best_config['Method']}, LR={best_config['Learning_Rate']}, Decay={best_config['Decay_Type']})..."
        )
        X_std, _ = preprocess_data(self.data, method=best_config["Method"])

        epochs = self.config.get("oja", {}).get("epochs", 1000)
        tolerance = self.config.get("oja", {}).get("tolerance", 1e-6)
        seed = self.config.get("oja", {}).get("seed", 42)

        histories = []
        w_ojas = []

        for run_idx in range(5):
            run_seed = seed + run_idx
            oja_neuron = OjaNeuron(
                n_features=X_std.shape[1],
                learning_rate=best_config["Learning_Rate"],
                max_epochs=epochs,
                tol=tolerance,
                seed=run_seed,
                decay_type=best_config["Decay_Type"],
            )
            oja_neuron.fit(X_std)
            histories.append(oja_neuron.history)

            # Align weights of this run immediately with respect to pca_loadings
            w_run = oja_neuron.w.copy()
            alignment = np.sign(np.dot(w_run, pca_loadings))
            if alignment == 0:
                alignment = 1
            w_ojas.append(w_run * alignment)

        # Average aligned weights and normalize the resulting vector
        w_oja_aligned = np.mean(w_ojas, axis=0)
        w_oja_aligned /= np.linalg.norm(w_oja_aligned)

        # 4. Comparación Oja vs PCA
        norm_pca = np.linalg.norm(pca_loadings)
        norm_oja = np.linalg.norm(w_oja_aligned)

        cos_sim = np.dot(w_oja_aligned, pca_loadings) / (norm_pca * norm_oja)
        # Clip to avoid arccos NaN on float imprecision
        cos_sim_clipped = np.clip(cos_sim, -1.0, 1.0)
        angle = np.arccos(cos_sim_clipped) * 180 / np.pi

        print(f"\nSimilitud Coseno entre Oja y PCA (Promedio): {cos_sim:.6f}")
        print(f"Ángulo de diferencia: {angle:.4f}°")

        # Crear tabla comparativa
        print("\nComparación de Loadings:")
        print(
            f"{'Feature':<15} | {'sklearn PCA':<12} | {'Oja Rule':<12} | {'Diff Absoluta':<12}"
        )
        print("-" * 58)
        for i, feature in enumerate(feature_names):
            diff = abs(pca_loadings[i] - w_oja_aligned[i])
            print(
                f"{feature:<15} | {pca_loadings[i]:>12.6f} | {w_oja_aligned[i]:>12.6f} | {diff:>12.6f}"
            )

        # Graficar convergencia y comparación
        countries = self.data.iloc[:, 0].values
        plot_convergence(histories, output_dir=self.output_dir)
        plot_loadings_comparison(
            feature_names, pca_loadings, w_ojas, output_dir=self.output_dir
        )
        plot_table(
            feature_names, pca_loadings, w_ojas, output_dir=self.output_dir
        )
        plot_projections_comparison(
            countries, X_std, pca_loadings, w_ojas, output_dir=self.output_dir
        )

        print(
            f"\nExperimento de Oja finalizado. Resultados guardados en '{self.output_dir}'."
        )

    def run_sweep(self, pca_loadings):
        print("\n" + "=" * 50)
        print("Iniciando Barrido de Hiperparámetros (Sweep)...")
        print("=" * 50)

        methods = ["z-score", "min-max", "unit"]
        lrs = [0.01, 0.05, 0.1]
        decays = ["constant", "inverse", "exponential"]
        epochs = self.config.get("oja", {}).get("epochs", 1000)
        tolerance = self.config.get("oja", {}).get("tolerance", 1e-6)
        seed = self.config.get("oja", {}).get("seed", 42)

        results = []

        for method in methods:
            X_std, _ = preprocess_data(self.data, method=method)
            for lr in lrs:
                for decay in decays:
                    for run_idx in range(5):
                        run_seed = seed + run_idx
                        oja_neuron = OjaNeuron(
                            n_features=X_std.shape[1],
                            learning_rate=lr,
                            max_epochs=epochs,
                            tol=tolerance,
                            seed=run_seed,
                            decay_type=decay,
                        )
                        oja_neuron.fit(X_std)
                        w_oja = oja_neuron.w.copy()
                        if np.isnan(w_oja).any():
                            angle = 180.0
                            cos_sim = 0.0
                        else:
                            alignment = np.sign(np.dot(w_oja, pca_loadings))
                            if alignment == 0:
                                alignment = 1
                            w_oja_aligned = w_oja * alignment

                            norm_pca = np.linalg.norm(pca_loadings)
                            norm_oja = np.linalg.norm(w_oja_aligned)

                            cos_sim = np.dot(w_oja_aligned, pca_loadings) / (
                                norm_pca * norm_oja
                            )
                            cos_sim_clipped = np.clip(cos_sim, -1.0, 1.0)
                            angle = np.arccos(cos_sim_clipped) * 180 / np.pi

                        epochs_run = len(oja_neuron.history)

                        results.append(
                            {
                                "Method": method,
                                "Learning_Rate": lr,
                                "Decay_Type": decay,
                                "Run": run_idx,
                                "Seed": run_seed,
                                "Angle_Degrees": angle,
                                "Cos_Sim": cos_sim,
                                "Epochs": epochs_run,
                            }
                        )

        import pandas as pd

        df_results = pd.DataFrame(results)

        csv_path = os.path.join(self.output_dir, "oja_sweep_results.csv")
        df_results.to_csv(csv_path, index=False)

        # Generar gráficos del sweep
        plot_sweep_results(df_results, output_dir=self.output_dir)

        print(f"Sweep completado. Se probaron {len(results)} corridas totales (45 combinaciones x 5 corridas).")
        print(f"Resultados guardados en {csv_path}\n")

        # Agrupar para encontrar el mejor promedio por configuración de hiperparámetros
        df_grouped = df_results.groupby(["Method", "Learning_Rate", "Decay_Type"]).agg(
            mean_angle=("Angle_Degrees", "mean"),
            mean_cos_sim=("Cos_Sim", "mean"),
            mean_epochs=("Epochs", "mean")
        ).reset_index()

        df_grouped = df_grouped.sort_values(by="mean_angle", ascending=True)

        print("Top 5 Mejores Configuraciones (menor ángulo promedio):")
        print(df_grouped.head(5).to_string(index=False))

        best_row = df_grouped.iloc[0]
        return {
            "Method": best_row["Method"],
            "Learning_Rate": best_row["Learning_Rate"],
            "Decay_Type": best_row["Decay_Type"]
        }
