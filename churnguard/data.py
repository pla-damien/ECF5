"""Fonctions de chargement et préprocessing des données."""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Charge le fichier CSV depuis le chemin donné et retourne un DataFrame."""
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Nettoie le DataFrame et retourne X (features) et y (target)."""
    df = df.copy()
    # TotalCharges contient des espaces vides dans le dataset source
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop(columns=["customerID"])
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    return X, y
