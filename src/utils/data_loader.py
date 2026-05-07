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

def preprocess_data(df):
    """
    Z-score normalizes the features of the dataset.
    Features: Area, GDP, Inflation, Life.expect, Military, Pop.growth, Unemployment.
    """
    features = df.iloc[:, 1:].values  # Skip Country column
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    # Avoid division by zero
    std[std == 0] = 1.0
    normalized_features = (features - mean) / std
    return normalized_features, df.iloc[:, 0].values
