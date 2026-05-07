"""Tests pour les modules churnguard.train et churnguard.evaluate."""

# On importe pandas pour construire les données de test
import pandas as pd

# On importe Pipeline pour vérifier le type retourné par train_model
from sklearn.pipeline import Pipeline

# On importe les deux fonctions qu'on teste
from churnguard.train import train_model
from churnguard.evaluate import compute_metrics


# --- Fixture partagée ---
# Une fixture pytest est une fonction qui prépare des données réutilisables.
# Le décorateur @pytest.fixture indique à pytest que c'est une fixture.
# Chaque test qui déclare "sample_data" en paramètre reçoit automatiquement
# ce que cette fonction retourne — sans avoir à réécrire le même DataFrame.
import pytest


@pytest.fixture
def sample_data():
    # On crée un DataFrame avec toutes les colonnes attendues par train_model
    # 3 lignes suffisent pour entraîner un modèle sans erreur
    X = pd.DataFrame({
        # Colonnes numériques (traitées par StandardScaler dans le pipeline)
        "tenure": [1, 12, 24],
        "MonthlyCharges": [20.0, 50.0, 80.0],
        "TotalCharges": [20.0, 600.0, 1920.0],
        "SeniorCitizen": [0, 0, 1],
        # Colonnes catégorielles (traitées par OneHotEncoder dans le pipeline)
        "gender": ["Male", "Female", "Male"],
        "Partner": ["No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes"],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No"],
        "OnlineBackup": ["No", "No", "Yes"],
        "DeviceProtection": ["No", "Yes", "No"],
        "TechSupport": ["No", "No", "Yes"],
        "StreamingTV": ["No", "Yes", "No"],
        "StreamingMovies": ["No", "No", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Bank transfer", "Credit card"],
    })

    # y = la cible : 0 = pas de churn, 1 = churn
    # On met des valeurs différentes pour que le modèle ait quelque chose à apprendre
    y = pd.Series([0, 1, 0])

    # La fixture retourne un tuple (X, y) que les tests recevront
    return X, y


def test_train_model_returns_fitted_pipeline(sample_data):
    # sample_data est injecté automatiquement par pytest grâce à la fixture ci-dessus
    # On décompacte le tuple en X et y
    X, y = sample_data

    # On entraîne un modèle de régression logistique avec des params par défaut
    # "lr" = LogisticRegression, {} = aucun hyperparamètre personnalisé
    result = train_model(X, y, "lr", {})

    # On vérifie que le résultat est bien un Pipeline scikit-learn
    assert isinstance(result, Pipeline)

    # On vérifie que le pipeline est bien entraîné en lui demandant de prédire
    # .predict(X) doit retourner un tableau de 3 valeurs (une par ligne de X)
    # .shape == (3,) signifie : tableau 1D de longueur 3
    assert result.predict(X).shape == (3,)


def test_compute_metrics_returns_expected_keys(sample_data):
    # On récupère X et y depuis la fixture
    X, y = sample_data

    # On entraîne d'abord un modèle (compute_metrics a besoin d'un modèle entraîné)
    model = train_model(X, y, "lr", {})

    # On calcule les métriques en passant le modèle et les données de test
    # (ici on utilise les mêmes données d'entraînement — c'est acceptable pour un test unitaire)
    metrics = compute_metrics(model, X, y)

    # On définit l'ensemble exact des clés qu'on attend dans le dictionnaire retourné
    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}

    # set(metrics.keys()) convertit les clés du dict en ensemble
    # On compare les deux ensembles — l'ordre n'a pas d'importance avec set
    assert set(metrics.keys()) == expected_keys
