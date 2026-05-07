# ChurnGuard MLOps

[![CI](https://github.com/pla-damien/ECF5/actions/workflows/ci.yml/badge.svg)](https://github.com/pla-damien/ECF5/actions/workflows/ci.yml)

API de prédiction de churn client pour TelcoFr — modèle Random Forest versionné dans MLflow, servi par FastAPI, containerisé avec Docker.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  docker compose up                    │
│                                                       │
│  ┌──────────────┐  train+register  ┌──────────────┐  │
│  │   trainer    │─────────────────►│    MLflow    │  │
│  │  (one-shot)  │                  │   UI :5000   │  │
│  └──────────────┘                  └──────┬───────┘  │
│                                           │ registry  │
│  ┌──────────────┐◄──────────────────────── │          │
│  │  API FastAPI │   load model             │          │
│  │    :8000     │                  ┌───────▼───────┐  │
│  └──────────────┘                  │ mlflow-data   │  │
│                                    │ (volume)      │  │
│                                    └───────────────┘  │
└──────────────────────────────────────────────────────┘
```

## Quickstart

```bash
git clone https://github.com/pla-damien/ECF5.git
cd ECF5
docker compose up --build
```

L'API est disponible sur `http://localhost:8000` après que le trainer ait terminé (~2 min).

## Endpoints

### Santé
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model": "churnguard", "version": "1"}
```

### Prédiction simple
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0, "TotalCharges": 840.0
  }'
```
```json
{"churn": true, "probability": 0.73}
```

### Prédiction batch (max 100 clients)
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{...}, {...}]'
```

## Documentation interactive

`http://localhost:8000/docs` — Swagger UI généré automatiquement par FastAPI.

## MLflow UI

`http://localhost:5000` — suivi des expériences, métriques, modèles enregistrés.

## Image Docker

```
ghcr.io/pla-damien/ecf5:latest
```

## Licence

MIT
