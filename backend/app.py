from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
import base64
import io
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps, ImageChops, ImageEnhance, ImageFilter
import cv2
import joblib
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import time
from datetime import datetime
from prometheus_client import Counter, Gauge, Histogram, generate_latest, Summary, make_asgi_app, Info, CONTENT_TYPE_LATEST
from retrain import retrain




# Create a metric to track time spent and requests made.
REQUEST_TIME = Gauge(
    'request_processing_seconds',
    'Time spent processing request')
PREDICTION_RESULT = Gauge(
    'prediction_result',
    'Prediction result by the model')
REQUEST_DATE= Info(
    'request_date',
    'Date of the request')

# Définir le chemin du modèle
MODEL_PATH = "knn_model.pkl"
# Création du dossier pour stocker les images si inexistant
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Vérifier et créer le dossier de stockage si besoin
HELP_DATA_DIR = "help_data"
os.makedirs(HELP_DATA_DIR, exist_ok=True)




def save_debug_image(image, step_name):
    """ Enregistre l'image avec un timestamp et le nom de l'étape """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{DEBUG_DIR}/{step_name}_{timestamp}.png"
    image.save(filename)
    print(f"📸 Image sauvegardée : {filename}")
    
    
# Charger le modèle au démarrage
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"❌ Modèle introuvable : {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    
    print("✅ Modèle chargé avec succès.")

    yield

    print("ℹ️ Application en cours d'arrêt...")

# Définition de l'API FastAPI
app = FastAPI(lifespan=lifespan)

# Add prometheus asgi middleware to route /metrics requests
# metrics_app = make_asgi_app()
# app.mount("/metrics", metrics_app)

# Servir les images de debug
app.mount("/debug_images", StaticFiles(directory=DEBUG_DIR), name="debug_images")

# Configuration CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#############################class helpdata(BaseModel):
class HelpDataRequest(BaseModel):
    image: str  # Image en base64
    number: int  # Label attendu

################## Classe image ######################################################################################
# Définition du modèle de requête
class ImageData(BaseModel):
    image: str  # Image en base64

### === Fonctions de traitement d'image === ###
def replace_transparent_background(image):
    """ Remplace le fond transparent par du blanc """
    image_arr = np.array(image)
    if len(image_arr.shape) == 2:
        return image  # L'image est déjà en niveaux de gris
    
    alpha_channel = image_arr[:, :, 3]  # Canal alpha
    mask = (alpha_channel == 0)
    
    # Remplacer les pixels transparents par du blanc
    image_arr[mask] = [255, 255, 255, 255]

    return Image.fromarray(image_arr)

def trim_borders(image):
    """ Supprime les bordures inutiles autour du chiffre """
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    
    return image.crop(bbox) if bbox else image

def pad_image(image):
    """ Ajoute une marge blanche autour du chiffre """
    return ImageOps.expand(image, border=30, fill=255)

def to_grayscale(image):
    """ Convertit l'image en niveaux de gris """
    return image.convert('L')

def invert_colors(image):
    """ Inverse les couleurs (le chiffre devient noir sur fond blanc) """
    return ImageOps.invert(image)

def resize_image(image):
    """ Redimensionne l'image en 8x8 pixels sans déformation en ajoutant des bordures """
    # Trouver le côté le plus long
    width, height = image.size
    max_side = max(width, height)
    
    # Calculer les coordonnées pour centrer l'image
    left = (width - max_side) // 2
    top = (height - max_side) // 2
    right = (width + max_side) // 2
    bottom = (height + max_side) // 2
    
    # Recadrer l'image pour qu'elle soit carrée
    cropped_image = image.crop((left+5, top+5, right+5, bottom+5))
    
    # Redimensionner l'image à 8x8 pixels
    return cropped_image.resize((8, 8), Image.LANCZOS)

def crop_non_white_area(image):
    # Convert to numpy array
    image_array = np.array(image, dtype=np.float32)

    # Find non-white rows and columns
    non_white_rows = np.where(image_array.min(axis=1) < 255)[0]
    non_white_cols = np.where(image_array.min(axis=0) < 255)[0]

    if non_white_rows.size == 0 or non_white_cols.size == 0:
        return image  # Return original if no non-white pixels found

    # Get bounding box
    top, bottom = non_white_rows[0], non_white_rows[-1]
    left, right = non_white_cols[0], non_white_cols[-1]

    # Calculate width and height
    height, width = bottom - top + 1, right - left + 1
    max_side = max(width, height)

    # Compute new top, bottom, left, and right to keep square shape
    center_x, center_y = (left + right) // 2, (top + bottom) // 2
    half_side = max_side // 2

    new_left = max(0, center_x - half_side)
    new_right = min(image.width, center_x + half_side)
    new_top = max(0, center_y - half_side)
    new_bottom = min(image.height, center_y + half_side)

    # Crop and return the image
    cropped_image = image.crop((new_left, new_top, new_right, new_bottom))
    return cropped_image



# Fonction pour augmenter le contraste
def enhance_contrast(image, factor=1.5):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

# Fonction pour épaissir les lignes
def thicken_lines(image):
    return image.filter(ImageFilter.MaxFilter(3)) 

# Fonction pour passer l'image en noir et blanc
def image_contrasted(image):
    return image.point(lambda x: 0 if x < 3 else 255)
 

 
### === Pipeline de prétraitement d'image === ###
def preprocess_image(image):
    print("🎨 Début du prétraitement...")
    
    
    #save_debug_image(image, f"step1_original")

    if image.mode == "RGBA":
        image = replace_transparent_background(image)
        save_debug_image(image, "step2_no_transparency")
        print("🔄 Fond transparent remplacé")

    image = crop_non_white_area(image)
    #save_debug_image(image, "step3a_cropped")
    print("🌾 Zone non blanche recadrée")

    image = trim_borders(image)
    #save_debug_image(image, "step3b_trimmed")
    print("✂️ Bordures supprimées")
    
    #image = Image.open(io.BytesIO(file.read()))
    #image_array = np.array(image)[:,:,3]
    
    # size = 37
    # center_weight = size**2/2
    # edge_weight = size**2/((size**2-1)*2)
    # core = np.full((size, size), edge_weight)
    # center = size // 2
    # core[center, center] = center_weight
    # core /= core.sum()   
    # convoled_image = np.copy(image_array)
    # convoled_image = cv2.filter2D(image_array, -1, core)
    # image_array = cv2.resize(convoled_image, (8, 8))
    # image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255
    
    
    image = to_grayscale(image)
    #save_debug_image(image, "step4_grayscale")
    print("🌑 Converti en niveaux de gris")
    
    image = pad_image(image)
    #save_debug_image(image, "step5_padded")
    print("📏 Marges ajoutées")

    image = invert_colors(image)
    #save_debug_image(image, "step6_inverted")
    print("🎨 Couleurs inversées")

    # Ajout des nouvelles étapes
    image = enhance_contrast(image)
    #save_debug_image(image, "step6b_contrast_enhanced")
    print("🔲 Contraste amélioré")

    image = thicken_lines(image)
    #save_debug_image(image, "step6c_thickened")
    print("✍️ Traits épaissis")
    
    image = resize_image(image)
    save_debug_image(image, f"step7_resized")
    print("📐 Image redimensionnée en 8x8")

    #return image_array
    return image
### === Route pour les métriques === ###

@app.get("/metrics")
async def get_metrics():
    """ Route pour les métriques Prometheus """
    # cette route est instrumentée par Prometheus et renvoie la dernière valeur des métriques
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

###=== Route pour pour récupérer les helpdata === ###
# elles contienns l'image en tableau de valeur npy et le label attendu
@app.post("/api/helpdata")
async def save_help_data(request: HelpDataRequest):
    try:
        print("📥 Requête reçue sur /api/helpdata")

        # Vérifier que l'image et le label sont bien reçus
        if not request.image or not request.number:
            raise ValueError("L'image ou le label est manquant.")

        print("🖼️ Image Base64 reçue. Début du décodage...")

        # Identifier si le format de l’image est correct
        if "," not in request.image:
            raise ValueError("Format Base64 incorrect. Vérifiez le préfixe de l’image.")

        image_data = base64.b64decode(request.image.split(",")[1])  
        
        print("✅ Image décodée avec succès. Chargement avec Pillow...")

        # Charger l’image avec Pillow
        image = Image.open(BytesIO(image_data))

        # Vérifier le format
        print(f"📏 Dimensions de l’image : {image.size}, Format : {image.format}")

        if image.format not in ["PNG", "JPEG"]:
            raise ValueError(f"Format d'image non supporté : {image.format}")

        image_array = np.array(image)[:,:,3]

        # convertir l'image en array numpy
        size = 37
        center_weight = size**2/2
        edge_weight = size**2/((size**2-1)*2)
        core = np.full((size, size), edge_weight)
        center = size // 2
        core[center, center] = center_weight
        core /= core.sum()   
        convoled_image = np.copy(image_array)
        convoled_image = cv2.filter2D(image_array, -1, core)
        image_array = cv2.resize(convoled_image, (8, 8))
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255
        # Générer le nom de fichier
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{request.number}"
        filepath = os.path.join(HELP_DATA_DIR, filename)

        print(f"💾 Enregistrement de l’image sous : {filepath}")

        # convertir l'image en array numpy
        # image_array = np.array(image)
        # sauvegarder le tableau numpy
        np.save(filepath, image_array)
        # image.save(filepath, format="PNG")

        print("✅ Image sauvegardée avec succès.")
        return {"message": "Image sauvegardée", "filename": filename}

    except Exception as e:
        print(f"❌ Erreur dans save_help_data: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la sauvegarde: {str(e)}")










### === Route API pour la prédiction === ###

@app.post(
    "/api/predict",
    summary="Predict a digit from a base64-encoded image",
    description="Receives a base64 image, preprocesses it, and uses the KNN model to predict the digit."
)

async def predict_image(data: ImageData):
    """
    Predict a digit from a base64-encoded image.

    Parameters:
        data (ImageData): Input data containing the base64-encoded image.

    Returns:
        dict: A JSON response containing the predicted digit.
    """
    try:
        if not data.image :
        #if not data.image and not awaited_result:
            raise HTTPException(status_code=400, detail="Aucune image fournie.")
        t0 = datetime.now()    
        print("prediction commencée")

        print("📷 Image reçue")

        # Décoder l'image Base64
        img_data = base64.b64decode(data.image.split(",")[1])
        img = Image.open(io.BytesIO(img_data))

        print("🖼️ Image décodée avec succès")

        # Appliquer le pipeline de prétraitement
        img = preprocess_image(img)

        # Convertir en array numpy et normaliser les valeurs entre 0 et 16
        img_array = np.array(img, dtype=np.float32)
        img_array = (16 * (img_array / 255)).astype(int)

        print(f"🔢 Image convertie en array : {img_array.shape}")

        # Aplatir l'image pour correspondre au format attendu par le modèle
        img_flatten = img_array.flatten().reshape(1, -1)

        # Vérifier que le modèle est bien chargé
        if model is None:
            raise HTTPException(status_code=500, detail="Modèle non chargé.")

        # Faire la prédiction
        prediction = model.predict(img_flatten)

        print(f"✅ Prédiction effectuée : {prediction[0]}")
        
        t1 = datetime.now()
        latency = (t1 - t0).total_seconds()
        REQUEST_TIME.set(latency)
        
        PREDICTION_RESULT.set(prediction[0])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        REQUEST_DATE.info({"date": timestamp})
        # Enregistrer les métriques
        print(f"⏱️ Latence de la prédiction : {latency:.2f} secondes")
        
        return {"prediction": int(prediction[0])}

    except Exception as e:
        print(f"❌ Erreur dans predict_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/debug_images")
async def list_debug_images():
    try:
        files = os.listdir(DEBUG_DIR)
        files.sort(reverse=True)  # Trier pour afficher les plus récentes en premier
        return {"images": [f"/debug_images/{file}" for file in files]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des images: {str(e)}")
    
@app.post("/api/retrain")
async def retrain_model():
    try:
        print("🔄 Début de la ré-entraînement du modèle...")
        retrain()
        print("✅ Ré-entraînement terminé avec succès.")
        return {"message": "Ré-entraînement terminé avec succès."}
    except Exception as e:
        print(f"❌ Erreur dans retrain_model: {e}")
        raise HTTPException(status_code=500, detail=str(e))