# Dockerfile pour déploiement du modèle de classification de pneumothorax
FROM python:3.10.13-slim

# Définir le répertoire de travail
WORKDIR /app

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/outputs/trained_model.h5
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flask gunicorn

# Copier le code de l'application
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p /app/outputs /app/logs /app/temp

# Copier le modèle et les artifacts (seront montés au runtime)
# COPY outputs/ /app/outputs/

# Exposer le port
EXPOSE 5000

# Créer un utilisateur non-root pour la sécurité
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app

USER appuser

# Commande de démarrage
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "app:app"]