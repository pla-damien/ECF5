"""API FastAPI pour la prédiction de churn client."""

import os
from contextlib import asynccontextmanager

import mlflow.pyfunc
import mlflow.tracking
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Dict pour permettre la réassignation depuis le lifespan imbriqué
model_store: dict = {}


class CustomerIn(BaseModel):
    """Données d'un client envoyées à l'API pour prédiction."""

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
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
    MonthlyCharges: float = Field(ge=0.0)
    TotalCharges: float = Field(ge=0.0)


class PredictionOut(BaseModel):
    """Réponse de l'API pour une prédiction."""

    churn: bool
    probability: float = Field(ge=0.0, le=1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, libère la mémoire à l'arrêt."""

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        model_store["model"] = mlflow.pyfunc.load_model("models:/churnguard/Production")

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("churnguard", stages=["Production"])
        model_store["version"] = versions[0].version if versions else "unknown"

        print(
            f"Modèle churnguard v{model_store['version']} chargé depuis {tracking_uri}"
        )

    except Exception as e:
        print(f"Impossible de charger le modèle : {e}")

    yield

    model_store.clear()


app = FastAPI(
    title="ChurnGuard API",
    description="API de prédiction de churn client TelcoFr",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Vérifie que l'API tourne et retourne la version du modèle."""
    return {
        "status": "ok",
        "model": "churnguard",
        "version": model_store.get("version", "unknown"),
    }


@app.post("/predict", response_model=PredictionOut)
def predict(customer: CustomerIn) -> PredictionOut:
    """Prédit le churn pour un seul client."""

    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    df = pd.DataFrame([customer.model_dump()])
    proba = float(model_store["model"].predict(df)[0])

    return PredictionOut(churn=proba > 0.5, probability=proba)


@app.post("/predict/batch", response_model=list[PredictionOut])
def predict_batch(customers: list[CustomerIn]) -> list[PredictionOut]:
    """Prédit le churn pour une liste de clients (max 100)."""

    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="La liste de clients est vide")

    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 clients par requête")

    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    df = pd.DataFrame([c.model_dump() for c in customers])
    probas = model_store["model"].predict(df)

    return [PredictionOut(churn=float(p) > 0.5, probability=float(p)) for p in probas]
