"""Fonctions d'entraînement du modèle."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Colonnes numériques et catégorielles (identiques au notebook)
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    params: dict,
) -> Pipeline:
    """Entraîne un pipeline scikit-learn et le retourne."""
    # Toutes les colonnes qui ne sont pas numériques sont catégorielles
    cat_cols = [col for col in X.columns if col not in NUM_COLS]

    # Préprocessing : normalisation des nombres, encodage des catégories
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    # Choix du classifier selon model_name
    if model_name == "lr":
        classifier = LogisticRegression(**params)
    elif model_name == "rf":
        classifier = RandomForestClassifier(**params)
    elif model_name == "gb":
        classifier = GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"model_name inconnu : {model_name}. Choisir parmi : lr, rf, gb")

    # Pipeline = préprocessing + classifier
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", classifier),
    ])

    pipeline.fit(X, y)
    return pipeline


if __name__ == "__main__":
    # Ces imports sont uniquement nécessaires pour le script d'entraînement,
    # pas pour la fonction train_model — on les met donc ici, pas en haut du fichier
    import os

    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    from sklearn.model_selection import train_test_split

    # On importe nos propres modules pour charger et préprocesser les données
    from churnguard.data import load_data, preprocess
    from churnguard.evaluate import compute_metrics

    # --- Chargement des données ---
    # load_data lit le CSV et retourne un DataFrame brut
    df = load_data("data/telco_churn.csv")

    # preprocess nettoie le DataFrame et retourne X (features) et y (cible)
    X, y = preprocess(df)

    # --- Découpage train / test ---
    # 80% des lignes pour entraîner, 20% pour évaluer
    # random_state=42 garantit qu'on obtient toujours le même découpage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Configuration MLflow ---
    # On indique à MLflow où est le serveur — lit la variable d'environnement si présente
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

    # On crée (ou réutilise) une expérience nommée "churnguard"
    # Toutes les runs de cet entraînement seront regroupées sous cette expérience
    mlflow.set_experiment("churnguard")

    # --- Définition des 3 modèles à entraîner ---
    # Chaque entrée : (nom_court, hyperparamètres)
    # Ces hyperparamètres seront loggés dans MLflow pour traçabilité
    models_to_train = [
        ("lr", {"max_iter": 1000, "random_state": 42}),        # Régression logistique
        ("rf", {"n_estimators": 100, "random_state": 42}),     # Random Forest
        ("gb", {"n_estimators": 100, "random_state": 42}),     # Gradient Boosting
    ]

    # --- Boucle d'entraînement ---
    for model_name, params in models_to_train:

        # mlflow.start_run() ouvre une "run" = une session de tracking
        # run_name donne un nom lisible dans l'interface MLflow
        # Le bloc "with" ferme automatiquement la run à la fin
        with mlflow.start_run(run_name=model_name):

            # Étape 1 : entraîner le modèle avec nos données d'entraînement
            model = train_model(X_train, y_train, model_name, params)

            # Étape 2 : évaluer le modèle sur les données de TEST (jamais vues)
            metrics = compute_metrics(model, X_test, y_test)

            # Étape 3 : logger le nom du modèle comme paramètre
            mlflow.log_param("model_name", model_name)

            # Étape 4 : logger tous les hyperparamètres (ex: n_estimators=100)
            mlflow.log_params(params)

            # Étape 5 : logger les 5 métriques d'évaluation
            mlflow.log_metrics(metrics)

            # Étape 6 : créer la signature du modèle
            # La signature décrit le schéma exact des données d'entrée et de sortie
            # MLflow l'utilise pour valider les données lors des prédictions futures
            signature = infer_signature(X_train, model.predict(X_train))

            # Étape 7 : prendre une ligne d'exemple pour la documentation
            # MLflow l'affiche dans l'interface pour montrer le format attendu
            input_example = X_train.iloc[:1]

            # Étape 8 : logger le modèle complet (fichier .pkl + métadonnées)
            mlflow.sklearn.log_model(
                sk_model=model,              # le pipeline scikit-learn entraîné
                artifact_path="model",       # dossier dans lequel MLflow le stocke
                signature=signature,         # schéma entrée/sortie
                input_example=input_example, # exemple de données d'entrée
            )

            # Affichage dans le terminal pour suivre la progression
            print(
                f"[{model_name}] "
                f"accuracy={metrics['accuracy']:.3f}  "
                f"f1={metrics['f1']:.3f}  "
                f"roc_auc={metrics['roc_auc']:.3f}"
            )
