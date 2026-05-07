"""Enregistre le meilleur modèle dans le MLflow Model Registry."""

import os

import mlflow
from mlflow.tracking import MlflowClient

# On indique à MLflow où est le serveur — lit la variable d'environnement si présente
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

# MlflowClient est l'objet qui permet d'interagir avec le Registry
# (enregistrer, changer de stage, lister les versions, etc.)
client = MlflowClient()

# --- Étape 1 : trouver le meilleur run automatiquement ---

# On récupère l'objet "expérience" par son nom
experiment = mlflow.get_experiment_by_name("churnguard")

# On cherche tous les runs de cette expérience
# order_by=["metrics.roc_auc DESC"] → le meilleur roc_auc en premier
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.roc_auc DESC"],
)

# runs est un DataFrame pandas — on prend la première ligne = meilleur run
best_run = runs.iloc[0]

# On extrait l'ID unique du run (ex: "469ab7e64c924572b3b6786586706524")
best_run_id = best_run["run_id"]

# On extrait la valeur du roc_auc pour l'afficher
best_roc_auc = best_run["metrics.roc_auc"]

print(f"Meilleur run : {best_run_id}  (roc_auc={best_roc_auc:.3f})")

# --- Étape 2 : enregistrer le modèle dans le Registry ---

# L'URI dit à MLflow où trouver le modèle dans les artefacts du run
# "runs:/<id>/model" → le dossier "model" qu'on a loggé avec log_model()
model_uri = f"runs:/{best_run_id}/model"

# register_model crée une version dans le Registry sous le nom "churnguard"
# Si "churnguard" n'existe pas encore, MLflow le crée automatiquement
# Retourne un objet avec .version (ex: "1" ou "2" ou "3"...)
model_version = mlflow.register_model(model_uri=model_uri, name="churnguard")

print(f"Modèle enregistré — version {model_version.version}")

# --- Étape 3 : promouvoir en Staging ---

# Staging = "prêt à être testé, pas encore en production"
# C'est une convention MLflow pour organiser le cycle de vie du modèle
client.transition_model_version_stage(
    name="churnguard",               # nom du modèle dans le Registry
    version=model_version.version,   # numéro de version à promouvoir
    stage="Staging",                 # étape cible
)

print("Promu en Staging ✓")

# --- Étape 4 : promouvoir en Production ---

# Production = "ce modèle est utilisé par l'API"
# C'est ce stage que l'API FastAPI chargera au démarrage
client.transition_model_version_stage(
    name="churnguard",
    version=model_version.version,
    stage="Production",
)

print("Promu en Production ✓")

# --- Étape 5 : vérification ---

# On recharge le modèle directement depuis le Registry
# Si ça ne plante pas, c'est que tout fonctionne
loaded_model = mlflow.pyfunc.load_model("models:/churnguard/Production")

print("Verification OK — modele charge depuis models:/churnguard/Production")
