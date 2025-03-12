FROM python:3.11-slim

WORKDIR /locust

# Installer Locust
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les scripts de test
COPY . .

# Monter les volumes pour modifier les tests en live
VOLUME [ "/locust" ]

# Exposer le port de Locust
EXPOSE 8089

# Lancer Locust en mode standalone
CMD ["locust", "--host=http://api:8000"]
