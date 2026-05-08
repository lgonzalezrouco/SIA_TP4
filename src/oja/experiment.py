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

        oja_neuron = OjaNeuron(
            n_features=X_std.shape[1],
            learning_rate=best_config["Learning_Rate"],
            max_epochs=epochs,
            tol=tolerance,
            seed=seed,
            decay_type=best_config["Decay_Type"],
        )
        oja_neuron.fit(X_std)
        w_oja = oja_neuron.w.copy()

        # 4. Comparación Oja vs PCA
        # Alinear signos
        alignment = np.sign(np.dot(w_oja, pca_loadings))
        if alignment == 0:
            alignment = 1
        w_oja_aligned = w_oja * alignment

        norm_pca = np.linalg.norm(pca_loadings)
        norm_oja = np.linalg.norm(w_oja_aligned)

        cos_sim = np.dot(w_oja_aligned, pca_loadings) / (norm_pca * norm_oja)
        # Clip to avoid arccos NaN on float imprecision
        cos_sim_clipped = np.clip(cos_sim, -1.0, 1.0)
        angle = np.arccos(cos_sim_clipped) * 180 / np.pi

        print(f"\nSimilitud Coseno entre Oja y PCA: {cos_sim:.6f}")
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
        plot_convergence(oja_neuron.history, output_dir=self.output_dir)
        plot_loadings_comparison(
            feature_names, pca_loadings, w_oja_aligned, output_dir=self.output_dir
        )
        plot_table(
            feature_names, pca_loadings, w_oja_aligned, output_dir=self.output_dir
        )
        plot_projections_comparison(
            countries, X_std, pca_loadings, w_oja_aligned, output_dir=self.output_dir
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
                    oja_neuron = OjaNeuron(
                        n_features=X_std.shape[1],
                        learning_rate=lr,
                        max_epochs=epochs,
                        tol=tolerance,
                        seed=seed,
                        decay_type=decay,
                    )
                    oja_neuron.fit(X_std)
                    w_oja = oja_neuron.w.copy()

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
                            "Angle_Degrees": angle,
                            "Cos_Sim": cos_sim,
                            "Epochs": epochs_run,
                        }
                    )

        import pandas as pd

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="Angle_Degrees", ascending=True)

        csv_path = os.path.join(self.output_dir, "oja_sweep_results.csv")
        df_results.to_csv(csv_path, index=False)

        # Generar gráficos del sweep
        plot_sweep_results(df_results, output_dir=self.output_dir)

        print(f"Sweep completado. Se probaron {len(results)} combinaciones.")
        print(f"Resultados guardados en {csv_path}\n")

        print("Top 5 Mejores Configuraciones (menor ángulo):")
        print(df_results.head(5).to_string(index=False))

        return df_results.iloc[0].to_dict()
