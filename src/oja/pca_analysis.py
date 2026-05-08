import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from utils.data_loader import preprocess_data


class PCAAnalyzer:
    def __init__(self, df: pd.DataFrame, output_dir: str = "results/pca"):
        self.df_raw = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Get feature names (all columns except Country)
        self.feature_names = df.columns[1:].tolist()
        self.countries = df.iloc[:, 0].values

        # Preprocess data (z-score normalization)
        self.X_std, _ = preprocess_data(df)

        # Create a DataFrame for standardized data to easily plot
        self.df_std = pd.DataFrame(self.X_std, columns=self.feature_names)

        # Initialize PCA
        self.pca = PCA()

    def run(self):
        print("Running sklearn PCA Analysis...")

        # Fit PCA and transform data
        self.X_pca = self.pca.fit_transform(self.X_std)

        # Generate plots
        self.plot_boxplots()
        self.plot_correlation_heatmap()
        self.plot_scree()
        self.plot_loadings()
        self.plot_country_ranking()
        self.plot_biplot()

        # Save CSVs
        self.save_csv_results()

        print(f"PCA Analysis completed. Results saved in '{self.output_dir}'")

    def plot_boxplots(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Raw Data Boxplot
        sns.boxplot(data=self.df_raw.iloc[:, 1:], ax=axes[0])
        axes[0].set_title(
            "Distribución de Variables Originales (Escalas muy distintas)"
        )
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].set_yscale(
            "log"
        )  # Use log scale for better visualization due to Area/GDP differences

        # Standardized Data Boxplot
        sns.boxplot(data=self.df_std, ax=axes[1])
        axes[1].set_title("Distribución de Variables Estandarizadas (Z-Score)")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "01_boxplots.png"))
        plt.close()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(8, 6))
        # Compute correlation matrix
        corr_matrix = self.df_std.corr()

        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1
        )
        plt.title("Matriz de Correlación (Datos Estandarizados)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "02_correlation_heatmap.png"))
        plt.close()

    def plot_scree(self):
        plt.figure(figsize=(8, 5))

        variance_ratio = self.pca.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(variance_ratio)

        x = range(1, len(variance_ratio) + 1)

        plt.bar(x, variance_ratio, alpha=0.7, label="Varianza Explicada Individual")
        plt.plot(
            x,
            cumulative_variance,
            marker="o",
            color="red",
            label="Varianza Explicada Acumulada",
        )

        plt.xlabel("Componente Principal")
        plt.ylabel("Porcentaje de Varianza Explicada (%)")
        plt.title("Scree Plot (Varianza Explicada)")
        labels = [f"PC{i}" for i in x]
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Annotate percentages
        for i, v in enumerate(variance_ratio):
            plt.text(i + 1, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "03_scree_plot.png"))
        plt.close()

    def plot_loadings(self):
        plt.figure(figsize=(10, 6))

        loadings = self.pca.components_[0]

        # Sort loadings by absolute value (optional, but sorting helps interpretation)
        # We will keep the original order to easily compare with features

        sns.barplot(
            x=self.feature_names,
            y=loadings,
            hue=self.feature_names,
            legend=False,
            palette="viridis",
        )

        plt.axhline(0, color="black", linewidth=1)
        plt.ylabel("Valor de Carga (Loading)")
        plt.title("Cargas (Loadings) del Primer Componente Principal (PC1)")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "04_pc1_loadings.png"))
        plt.close()

    def plot_country_ranking(self):
        # PC1 values for each country
        pc1_scores = self.X_pca[:, 0]

        # Create DataFrame for sorting
        df_ranking = pd.DataFrame({"Country": self.countries, "PC1_Score": pc1_scores})

        df_ranking = df_ranking.sort_values(by="PC1_Score", ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(
            x="PC1_Score",
            y="Country",
            data=df_ranking,
            hue="Country",
            legend=False,
            palette="coolwarm",
        )

        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("Valor Proyectado en PC1")
        plt.title("Ranking de Países según Primer Componente Principal (PC1)")
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "05_pc1_country_ranking.png"))
        plt.close()

    def plot_biplot(self):
        plt.figure(figsize=(12, 10))

        pc1 = self.X_pca[:, 0]
        pc2 = self.X_pca[:, 1]

        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)

        # Plot points (countries)
        plt.scatter(pc1, pc2, alpha=0.7)

        # Add labels to points
        for i, country in enumerate(self.countries):
            plt.annotate(
                country,
                (pc1[i], pc2[i]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=9,
            )

        # Plot loading vectors
        for i, feature in enumerate(self.feature_names):
            plt.arrow(
                0,
                0,
                loadings[i, 0] * 4,
                loadings[i, 1] * 4,
                color="r",
                alpha=0.8,
                head_width=0.1,
                head_length=0.1,
            )
            plt.text(
                loadings[i, 0] * 4.5,
                loadings[i, 1] * 4.5,
                feature,
                color="r",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        plt.axvline(0, color="black", linestyle="--", alpha=0.5)

        plt.xlabel(f"PC1 ({self.pca.explained_variance_ratio_[0] * 100:.1f}%)")
        plt.ylabel(f"PC2 ({self.pca.explained_variance_ratio_[1] * 100:.1f}%)")
        plt.title("Biplot: PC1 vs PC2")
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "06_biplot.png"))
        plt.close()

    def save_csv_results(self):
        # 1. Save PC1 Loadings
        df_loadings = pd.DataFrame(
            {"Feature": self.feature_names, "PC1_Loading": self.pca.components_[0]}
        )
        df_loadings.to_csv(
            os.path.join(self.output_dir, "pc1_loadings.csv"), index=False
        )

        # 2. Save Country Rankings
        df_ranking = pd.DataFrame(
            {
                "Country": self.countries,
                "PC1_Score": self.X_pca[:, 0],
                "PC2_Score": self.X_pca[:, 1],
            }
        ).sort_values(by="PC1_Score", ascending=False)

        df_ranking.to_csv(
            os.path.join(self.output_dir, "country_pca_scores.csv"), index=False
        )
