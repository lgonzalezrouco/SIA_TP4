import os

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(history, output_dir="results/oja"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(history) + 1), history, linestyle="-")
    plt.yscale("log")
    plt.xlabel("Época")
    plt.ylabel("||Δw|| (Escala Log)")
    plt.title("Convergencia de la Regla de Oja")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_oja_convergence.png"))
    plt.close()


def plot_loadings_comparison(
    feature_names, pca_loadings, oja_loadings, output_dir="results/oja"
):
    os.makedirs(output_dir, exist_ok=True)

    x = np.arange(len(feature_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, pca_loadings, width, label="sklearn PCA", color="skyblue")
    plt.bar(x + width / 2, oja_loadings, width, label="Oja Rule", color="salmon")

    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Features")
    plt.ylabel("Loadings (Pesos)")
    plt.title("Comparación de Loadings: PCA vs Oja")
    plt.xticks(x, feature_names, rotation=45)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_oja_vs_pca_loadings.png"))
    plt.close()


def plot_sweep_results(df_results, output_dir="results/oja"):
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # 1. Ángulo de error según Método de estandarización y Learning Rate (solo decay inverso)
    plt.figure(figsize=(10, 6))
    df_inverse = df_results[df_results["Decay_Type"] == "inverse"]
    sns.barplot(
        data=df_inverse,
        x="Method",
        y="Angle_Degrees",
        hue="Learning_Rate",
        palette="viridis",
        errorbar=None,
    )
    plt.title("Error Angular según Estandarización y LR (con decay inverso 1/(1+t))")
    plt.ylabel("Ángulo de Diferencia (°)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "09_sweep_methods_lr.png"))
    plt.close()

    # 2. Ángulo de error según Método de decaimiento y Learning Rate (solo z-score)
    plt.figure(figsize=(10, 6))
    df_zscore = df_results[df_results["Method"] == "z-score"]
    sns.barplot(
        data=df_zscore,
        x="Decay_Type",
        y="Angle_Degrees",
        hue="Learning_Rate",
        palette="mako",
        errorbar=None,
    )
    plt.title("Error Angular según Función de Decaimiento y LR (con z-score)")
    plt.ylabel("Ángulo de Diferencia (°)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "10_sweep_decays_lr.png"))
    plt.close()

    # 3. Ángulo de error según Método de Estandarización y Decay (solo LR = 0.05)
    plt.figure(figsize=(10, 6))
    df_lr_005 = df_results[np.isclose(df_results["Learning_Rate"], 0.05)]
    sns.barplot(
        data=df_lr_005,
        x="Method",
        y="Angle_Degrees",
        hue="Decay_Type",
        palette="flare",
        errorbar=None,
    )
    plt.title("Error Angular según Estandarización y Decay (con LR fijo en 0.05)")
    plt.ylabel("Ángulo de Diferencia (°)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "11_sweep_methods_decays.png"))
    plt.close()


def plot_table(feature_names, pca_loadings, oja_loadings, output_dir="results/oja"):
    os.makedirs(output_dir, exist_ok=True)

    cell_text = []
    for i, feature in enumerate(feature_names):
        pca_val = pca_loadings[i]
        oja_val = oja_loadings[i]
        diff = abs(pca_val - oja_val)
        cell_text.append([feature, f"{pca_val:.6f}", f"{oja_val:.6f}", f"{diff:.6f}"])

    columns = ("Feature", "sklearn PCA", "Oja Rule", "Diff Absoluta")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=cell_text, colLabels=columns, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Hacer los headers en negrita (opcional, pero queda mejor)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    plt.title("Comparación de Loadings: PCA vs Oja", pad=20, weight="bold", size=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "12_oja_vs_pca_table.png"), dpi=150)
    plt.close()


def plot_projections_comparison(
    countries, X_std, pca_loadings, oja_loadings, output_dir="results/oja"
):
    os.makedirs(output_dir, exist_ok=True)

    pca_proj = X_std.dot(pca_loadings)
    oja_proj = X_std.dot(oja_loadings)

    plt.figure(figsize=(10, 10))
    plt.scatter(pca_proj, oja_proj, color="indigo", alpha=0.7)

    # Linea de identidad y=x
    min_val = min(min(pca_proj), min(oja_proj))
    max_val = max(max(pca_proj), max(oja_proj))
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        label="Identidad (PCA = Oja)",
        alpha=0.7,
    )

    # Anotar los países
    for i, country in enumerate(countries):
        plt.annotate(
            country,
            (pca_proj[i], oja_proj[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    plt.xlabel("Proyección sobre Componente Principal (sklearn PCA)")
    plt.ylabel("Proyección sobre Peso Sináptico (Oja Rule)")
    plt.title("Comparación de Proyecciones por País: PCA vs Oja")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "13_oja_vs_pca_projections.png"))
    plt.close()
