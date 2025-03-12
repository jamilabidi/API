# on va concatener les help_data/*.npy  avec le modèle de sklearn
import numpy as np
import os
import re
from sklearn.datasets import load_digits
from sklearn.utils import Bunch

def generate_new_dataset():
    # Charger le dataset existant
    digits = load_digits()

    # Dossier contenant les fichiers .npy
    data_dir = "help_data"

    # Lister tous les fichiers .npy dans le dossier
    npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

    # Charger les images et extraire les labels
    new_images = []
    new_labels = []

    for file in npy_files:
        # Extraire la target (dernier nombre avant ".npy")
        match = re.search(r"-(\d+)\.npy$", file)
        if match:
            label = int(match.group(1))  # Convertir en entier
            new_labels.append(label)
            
            # Charger l'image et l'ajouter à la liste
            image = np.load(os.path.join(data_dir, file))
            new_images.append(image)

    # Conversion en numpy array
    new_images = np.array(new_images)
    new_labels = np.array(new_labels)
    print(f"Shape des images de load_digits: {digits.images.shape[1:]}")  # (8, 8)
    print(f"Shape des nouvelles images: {new_images.shape[1:]}")

    # Vérifier la forme des nouvelles images
    if new_images.shape[1:] != digits.images.shape[1:]:
        raise ValueError("Les dimensions des nouvelles images ne correspondent pas à celles de load_digits.")

    # Aplatir les images
    new_data = new_images.reshape((new_images.shape[0], -1))

    # Fusionner les données
    merged_data = np.vstack([digits.data, new_data])
    merged_target = np.hstack([digits.target, new_labels])
    merged_images = np.vstack([digits.images, new_images])

    # Créer un nouvel objet Bunch
    extended_digits = Bunch(
        data=merged_data,
        target=merged_target,
        images=merged_images,
        target_names=np.unique(merged_target)
    )

    print(f"Nouvelle taille du dataset: {extended_digits.data.shape}")
    # on nettoye le dossier help_data
    # for file in npy_files:
    #     os.remove(os.path.join(data_dir, file))
    # print("Les fichiers .npy ont été supprimés avec succès.")
    
    
    return extended_digits

