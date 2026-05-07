"""Fonctions de chargement et préprocessing des données."""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Charge le fichier CSV depuis le chemin donné et retourne un DataFrame."""
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Nettoie le DataFrame et retourne X (features) et y (target)."""
    # TotalCharges contient des espaces vides qu'on convertit en NaN
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # On supprime les lignes avec des valeurs manquantes
    df = df.dropna()

    # On supprime l'identifiant client (inutile pour le modèle)
    df = df.drop(columns=["customerID"])

    # y = 1 si le client a résilié, 0 sinon
    y = (df["Churn"] == "Yes").astype(int)

    # X = toutes les colonnes sauf Churn
    X = df.drop(columns=["Churn"])

    return X, y
