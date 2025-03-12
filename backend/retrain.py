import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from concatanate_data import generate_new_dataset
#fonction pour entrainer le modele
def retrain():
    d = generate_new_dataset()
    # Comptage des occurrences par classe
    class_counts = np.bincount(d.target)
    ### on rend les données équiprobables par sous échantillaonnage
    min_count = min(class_counts)  # Nombre minimum d’images parmi les classes

    np.random.seed(42)  # Fixe la graine pour garantir le même échantillonnage à chaque exécution
    # Création d'un index équilibré
    balanced_indices = np.hstack([
        np.random.choice(np.where(d.target == i)[0], min_count, replace=False)
        for i in range(10)  # Pour chaque classe 0 à 9
    ])

    # Tri des indices pour garder l’ordre des données
    balanced_indices.sort()

    # Nouveau dataset équilibré
    balanced_data = d.data[balanced_indices]
    balanced_target = d.target[balanced_indices]
    #on reconverti les données en dataset:
    # Conversion en DataFrame
    df_balanced = pd.DataFrame(balanced_data)
    df_balanced['target'] = balanced_target  # Ajout de la colonne cible
    ## on découpe les données
    number_x = df_balanced.copy()
    # print( number_x.head)
    number_y = number_x.pop("target")
    #### on split le jeu de donnée en train et test
    X_train, X_test, y_train, y_test = train_test_split(number_x,number_y, test_size=0.20, random_state=42) # utilise la méthode pour diviser le jeu de données en deux
    # print(X_train.shape, X_test.shape)


    ### on scale les données
    scaler =  StandardScaler() #instancie le StandardSCaler
    scaler.fit(X_train) # utilisation de fit sur le jeu de données d'entrainement
    X_train_scaled = scaler.transform(X_train) # transform
    X_test_scaled = scaler.transform(X_test) # transform

    # Affichez les données standardisées
    # print(X_train_scaled)
    neighbors_model = KNeighborsClassifier() # instancie un modèle KNN avec les paramètres par défault

    parameters = {'n_neighbors': range(3, 30)}
    # dictionnaire comprenant pour chaque paramètre (clé du dictionnaire) une liste de valeurs attendues.
    gridsearch_models = GridSearchCV(neighbors_model, param_grid=parameters, cv=5, scoring='accuracy') # instancie un gridsearchCV avec le modèle et le jeu de paramètres
    gridsearch_models.fit(X_train_scaled, y_train)
    gridsearch_accuracy = gridsearch_models.cv_results_
    # # Récupération des résultats du GridSearchCV
    # results = pd.DataFrame(gridsearch_models.cv_results_)

    # # Sélection des colonnes pertinentes : paramètre testé + accuracy moyenne
    # results = results[["param_n_neighbors", "mean_test_score"]]
    best_k = gridsearch_models.best_params_['n_neighbors']

    knn_clf = KNeighborsClassifier(n_neighbors=best_k) # instancie l'objet KNeighborsClassifier avec best_k voisins
    knn_clf.fit(X_train_scaled, y_train)# utilise l'objet et sa méthode fit pour entrainer le modèle sur le jeu de données d'entrainement
    y_pred = knn_clf.predict( X_test_scaled) # utiliser la fonction predict du modèle pour faire prédire le modèle
    y_true = y_test
    accuracy = accuracy_score(y_true, y_pred)  # Calcul de l'accuracy
    print("l'acuracy est ",accuracy)
    # Création de la matrice de confusion
    confusion_matrix_values = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None)

    # Affichage avec ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_values, display_labels=knn_clf.classes_)

    # Définir le chemin du fichier de sauvegarde
    save_dir = "models/confusion_matrix"
    os.makedirs(save_dir, exist_ok=True)  # Créer le dossier s'il n'existe pas
    save_path = os.path.join(save_dir, f"confusion_matrix_{int(time.time())}.png")

    # Tracer la matrice de confusion
    fig, ax = plt.subplots(figsize=(6,6))  # Création d'une nouvelle figure pour éviter les problèmes de sauvegarde
    disp.plot(cmap="Blues", ax=ax)

    # Enregistrer la figure
    plt.title("Matrice de Confusion")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)  # Fermer la figure pour éviter d'afficher plusieurs plots

    print(f"Matrice de confusion sauvegardée dans {save_path}")
    
    
    # Définir le dossier de sauvegarde
    save_dir = "models"
    # S'assurer que le dossier existe
    os.makedirs(save_dir, exist_ok=True)
    
    
    
    
    
    
    # Sauvegarder le modèle en lui donnant un nom type "{timespamp}_model.pkl" dans /models
    # Générer le nom du fichier
    filename = os.path.join(save_dir, f"model_{int(time.time())}.pkl")
    joblib.dump(knn_clf, filename, compress=3)
    print(f"Modèle sauvegardé sous le nom {filename}")
    
    
if __name__ == "__main__":
    retrain()
  