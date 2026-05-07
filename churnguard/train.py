"""Fonctions d'entraînement du modèle."""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Colonnes numériques — les autres sont considérées catégorielles
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    params: dict[str, Any],
) -> Pipeline:
    """Entraîne un pipeline scikit-learn et le retourne."""
    cat_cols = [col for col in X.columns if col not in NUM_COLS]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    if model_name == "lr":
        classifier = LogisticRegression(**params)
    elif model_name == "rf":
        classifier = RandomForestClassifier(**params)
    elif model_name == "gb":
        classifier = GradientBoostingClassifier(**params)
    else:
        raise ValueError(
            f"model_name inconnu : {model_name}. Choisir parmi : lr, rf, gb"
        )

    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", classifier),
        ]
    )

    pipeline.fit(X, y)
    return pipeline


if __name__ == "__main__":
    import os

    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    from sklearn.model_selection import train_test_split

    from churnguard.data import load_data, preprocess
    from churnguard.evaluate import compute_metrics

    df = load_data("data/telco_churn.csv")
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("churnguard")

    models_to_train = [
        ("lr", {"max_iter": 1000, "random_state": 42}),
        ("rf", {"n_estimators": 100, "random_state": 42}),
        ("gb", {"n_estimators": 100, "random_state": 42}),
    ]

    for model_name, params in models_to_train:
        with mlflow.start_run(run_name=model_name):
            model = train_model(X_train, y_train, model_name, params)
            metrics = compute_metrics(model, X_test, y_test)

            mlflow.log_param("model_name", model_name)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            signature = infer_signature(X_train, model.predict(X_train))
            input_example = X_train.iloc[:1]

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            print(
                f"[{model_name}] "
                f"accuracy={metrics['accuracy']:.3f}  "
                f"f1={metrics['f1']:.3f}  "
                f"roc_auc={metrics['roc_auc']:.3f}"
            )
