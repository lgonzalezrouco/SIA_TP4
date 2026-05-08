import pandas as pd
import numpy as np


def load_europe_data(file_path="data/europe.csv"):
    """
    Loads the Europe countries dataset and returns a DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None


def preprocess_data(df, method="z-score"):
    """
    Normalizes the features of the dataset.
    Features: Area, GDP, Inflation, Life.expect, Military, Pop.growth, Unemployment.
    method can be "z-score", "min-max", or "unit".
    """
    features = df.iloc[:, 1:].values.astype(float)  # Skip Country column

    if method == "z-score":
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1.0
        normalized_features = (features - mean) / std
    elif method == "min-max":
        f_min = np.min(features, axis=0)
        f_max = np.max(features, axis=0)
        rng = f_max - f_min
        rng[rng == 0] = 1.0
        normalized_features = (features - f_min) / rng
    elif method == "unit":
        norm = np.linalg.norm(features, axis=0)
        norm[norm == 0] = 1.0
        normalized_features = features / norm
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized_features, df.iloc[:, 0].values
