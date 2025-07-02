
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_diabetes

def load_sample_data(dataset_name: str) -> pd.DataFrame:
    """
    Loads a sample dataset suitable for multiple linear regression analysis based on the specified dataset name.

    Args:
        dataset_name (str): The name of the dataset to load ('Boston', 'Diabetes', 'Synthetic').

    Returns:
        pd.DataFrame: DataFrame containing features and target variable.

    Raises:
        ValueError: If dataset_name is not recognized.
        TypeError: If dataset_name is not a string.
    """
    if not isinstance(dataset_name, str):
        raise TypeError("dataset_name must be a string.")
    name = dataset_name.strip()
    if not name:
        raise ValueError("Dataset name cannot be empty.")
    name_lower = name.lower()

    if name_lower == 'boston':
        data = load_boston()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df

    elif name_lower == 'diabetes':
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df

    elif name_lower == 'synthetic':
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        coef = np.random.randn(n_features)
        y = X @ coef + np.random.randn(n_samples) * 0.5
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df['target'] = y
        return df

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
