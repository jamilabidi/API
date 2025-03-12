FROM python:3.11-slim
WORKDIR /app

ENV ENABLE_METRICS=true
# Installer les dépendances
COPY requirements.txt .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'API
COPY . .

# Monter les volumes et activer le mode dev
VOLUME [ "/app" ]

# Exposer le port 8000 pour FastAPI
EXPOSE 8000

# Lancer l'API avec Uvicorn en mode dev
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
