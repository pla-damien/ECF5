"""API FastAPI pour la prédiction de churn client."""

# asynccontextmanager : permet de définir le lifespan (démarrage/arrêt)
from contextlib import asynccontextmanager

# FastAPI : le framework web principal
# HTTPException : pour retourner des erreurs HTTP (400, 503, etc.)
from fastapi import FastAPI, HTTPException

# BaseModel : classe de base pour définir les schémas de données Pydantic
# Field : permet d'ajouter des contraintes de validation (≥ 0, ≤ 1, etc.)
from pydantic import BaseModel, Field

# Pour construire un DataFrame à partir des données reçues
import pandas as pd

# Pour charger le modèle depuis le MLflow Model Registry
import mlflow.pyfunc
import mlflow.tracking

# Pour lire la variable d'environnement MLFLOW_TRACKING_URI
import os


# Dictionnaire global qui stockera le modèle et sa version
# On utilise un dict car Python ne permet pas de réassigner
# une variable globale facilement depuis une fonction imbriquée
model_store: dict = {}


# ── Schémas Pydantic ────────────────────────────────────────────────────────

class CustomerIn(BaseModel):
    """Données d'un client envoyées à l'API pour prédiction."""

    # Chaque champ correspond à une colonne du dataset (après preprocess)
    # Field(ge=0) = "greater or equal to 0" → valeur minimale autorisée
    # Une valeur invalide déclenche automatiquement une erreur 422

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)        # 0 ou 1 uniquement
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)                      # nombre de mois ≥ 0
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0.0)          # montant mensuel ≥ 0
    TotalCharges: float = Field(ge=0.0)            # montant total ≥ 0


class PredictionOut(BaseModel):
    """Réponse de l'API pour une prédiction."""

    # True si le client va résilier, False sinon
    churn: bool

    # Probabilité de résiliation entre 0.0 et 1.0
    probability: float = Field(ge=0.0, le=1.0)


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, libère la mémoire à l'arrêt."""

    # On lit l'URI MLflow depuis la variable d'environnement
    # En local : MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    # Dans Docker : MLFLOW_TRACKING_URI=http://mlflow:5000 (valeur par défaut)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        # Chargement du modèle depuis le stage "Production" du Registry
        # Cette opération prend ~1-2 secondes — on la fait une seule fois ici
        model_store["model"] = mlflow.pyfunc.load_model("models:/churnguard/Production")

        # On récupère le numéro de version pour le retourner dans /health
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("churnguard", stages=["Production"])

        # Si une version existe on la stocke, sinon on met "unknown"
        model_store["version"] = versions[0].version if versions else "unknown"

        print(f"Modèle churnguard v{model_store['version']} chargé depuis {tracking_uri}")

    except Exception as e:
        # Si MLflow est indisponible, on continue sans modèle
        # Les routes /predict retourneront une erreur 503
        print(f"Impossible de charger le modèle : {e}")

    # L'API tourne tant qu'on est sur le yield
    yield

    # À l'arrêt de l'API, on vide le model_store pour libérer la mémoire
    model_store.clear()


# ── Application ─────────────────────────────────────────────────────────────

# On crée l'application FastAPI en lui passant le lifespan
app = FastAPI(
    title="ChurnGuard API",
    description="API de prédiction de churn client TelcoFr",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    """Vérifie que l'API tourne et retourne la version du modèle."""
    return {
        "status": "ok",
        "model": "churnguard",
        # .get() retourne "unknown" si la clé n'existe pas (modèle non chargé)
        "version": model_store.get("version", "unknown"),
    }


@app.post("/predict", response_model=PredictionOut)
def predict(customer: CustomerIn) -> PredictionOut:
    """Prédit le churn pour un seul client."""

    # Si le modèle n'est pas dans model_store → il n'a pas pu être chargé
    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    # model_dump() convertit l'objet Pydantic en dictionnaire Python
    # pd.DataFrame([...]) crée un DataFrame d'une seule ligne
    df = pd.DataFrame([customer.model_dump()])

    # predict() retourne un tableau numpy → on prend le premier (et seul) élément
    # float() convertit le numpy.float64 en float Python standard
    proba = float(model_store["model"].predict(df)[0])

    # churn = True si la probabilité dépasse 50%
    return PredictionOut(churn=proba > 0.5, probability=proba)


@app.post("/predict/batch", response_model=list[PredictionOut])
def predict_batch(customers: list[CustomerIn]) -> list[PredictionOut]:
    """Prédit le churn pour une liste de clients (max 100)."""

    # Erreur 400 si la liste est vide (pas de données à prédire)
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="La liste de clients est vide")

    # Erreur 400 si trop de clients (protection contre les abus)
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 clients par requête")

    # Erreur 503 si le modèle n'est pas chargé
    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    # On convertit la liste complète en DataFrame (N lignes d'un coup)
    # C'est plus efficace qu'appeler predict() en boucle sur chaque client
    df = pd.DataFrame([c.model_dump() for c in customers])

    # predict() retourne un tableau de N probabilités
    probas = model_store["model"].predict(df)

    # On construit une liste de PredictionOut, une par client
    return [
        PredictionOut(churn=float(p) > 0.5, probability=float(p))
        for p in probas
    ]
