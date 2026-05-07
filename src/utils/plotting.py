import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(data, title, labels=None, cmap="viridis"):
    """
    Generic heatmap plotter.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True if labels is not None else False, fmt="", cmap=cmap)
    plt.title(title)
    plt.show()

def plot_u_matrix(weights, grid_size):
    """
    Placeholder for U-matrix plotting.
    """
    # TODO: Implement U-matrix calculation and plotting
    pass
