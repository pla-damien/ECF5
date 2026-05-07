"""Enregistre le meilleur modèle dans le MLflow Model Registry."""

import os

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

client = MlflowClient()

experiment = mlflow.get_experiment_by_name("churnguard")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.roc_auc DESC"],
)

best_run = runs.iloc[0]
best_run_id = best_run["run_id"]
best_roc_auc = best_run["metrics.roc_auc"]

print(f"Meilleur run : {best_run_id}  (roc_auc={best_roc_auc:.3f})")

model_uri = f"runs:/{best_run_id}/model"
model_version = mlflow.register_model(model_uri=model_uri, name="churnguard")

print(f"Modèle enregistré — version {model_version.version}")

client.transition_model_version_stage(
    name="churnguard",
    version=model_version.version,
    stage="Staging",
)

print("Promu en Staging ✓")

client.transition_model_version_stage(
    name="churnguard",
    version=model_version.version,
    stage="Production",
)

print("Promu en Production ✓")

loaded_model = mlflow.pyfunc.load_model("models:/churnguard/Production")
print("Verification OK — modele charge depuis models:/churnguard/Production")
