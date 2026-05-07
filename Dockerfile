# ══════════════════════════════════════════════════════
# STAGE 1 — builder
# Rôle : installer toutes les dépendances dans un venv
# Ce stage sera jeté à la fin, il ne sera pas dans l'image finale
# ══════════════════════════════════════════════════════

# On part d'une image Python légère ("slim" = sans les outils système inutiles)
FROM python:3.11-slim AS builder

# On définit le dossier de travail dans le conteneur
WORKDIR /app

# On copie uniquement les fichiers nécessaires à l'installation des dépendances
# On copie pyproject.toml AVANT le code source pour profiter du cache Docker :
# si pyproject.toml ne change pas, Docker ne réinstalle pas les dépendances
COPY pyproject.toml .
COPY churnguard/__init__.py churnguard/__init__.py

# On crée un environnement virtuel Python isolé dans /venv
RUN python -m venv /venv

# On active le venv en ajoutant son dossier bin au PATH
ENV PATH="/venv/bin:$PATH"

# On met à jour pip en silence (-q) puis on installe le package et ses dépendances
RUN pip install -q --upgrade pip && pip install -q .


# ══════════════════════════════════════════════════════
# STAGE 2 — runtime
# Rôle : image finale légère avec uniquement ce qu'il faut pour tourner
# ══════════════════════════════════════════════════════

FROM python:3.11-slim AS runtime

WORKDIR /app

# On crée un utilisateur non-root "appuser" pour des raisons de sécurité
# Par défaut Docker tourne en root — c'est une mauvaise pratique en production
RUN useradd --create-home appuser

# On copie le venv complet depuis le stage builder
# --from=builder indique qu'on prend le fichier du stage précédent (pas du disque local)
COPY --from=builder /venv /venv

# On copie le code source du package churnguard
COPY churnguard/ ./churnguard/

# On copie le code de l'API FastAPI
COPY api/ ./api/

# On active le venv dans ce stage aussi
ENV PATH="/venv/bin:$PATH"

# On indique à Python de ne pas écrire les fichiers .pyc (inutiles dans un conteneur)
ENV PYTHONDONTWRITEBYTECODE=1

# On désactive le buffer Python pour que les logs apparaissent immédiatement
ENV PYTHONUNBUFFERED=1

# On déclare que l'application écoute sur le port 8000
EXPOSE 8000

# On bascule vers l'utilisateur non-root avant de lancer l'app
USER appuser

# Healthcheck Docker : vérifie toutes les 30s que /health répond
# --interval=30s : fréquence de vérification
# --timeout=10s  : si pas de réponse en 10s → échec
# --retries=3    : 3 échecs consécutifs → conteneur marqué "unhealthy"
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Commande de démarrage de l'API
# --host 0.0.0.0 : écoute sur toutes les interfaces réseau du conteneur (obligatoire)
# --port 8000    : port d'écoute
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
