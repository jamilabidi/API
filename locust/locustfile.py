import base64
import requests
from io import BytesIO
from locust import HttpUser, task, between

# URL de l'image à encoder en base64
IMAGE_URL = "https://media.istockphoto.com/vectors/number-three-3-hand-drawn-with-dry-brush-vector-id484207302?k=6&m=484207302&s=170667a&w=0&h=s3YANDyuLS8u2so-uJbMA2uW6fYyyRkabc1a6OTq7iI="

def get_base64_encoded_image():
    """Télécharge l'image et la convertit en base64"""
    response = requests.get(IMAGE_URL)
    if response.status_code == 200:
        image_data = base64.b64encode(response.content).decode("utf-8")
        return f"data:image/png;base64,{image_data}"
    return None

class APITestUser(HttpUser):
    wait_time = between(1, 3)  # Temps d'attente aléatoire entre les requêtes

    @task
    def test_predict(self):
        """Test de l'endpoint avec une image en base64"""
        base64_image = get_base64_encoded_image()
        if base64_image:
            payload = {"image": base64_image}
            headers = {"Content-Type": "application/json"}
            self.client.post("/api/predict", json=payload, headers=headers)

