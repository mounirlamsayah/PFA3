import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import mlflow
from config import IMG_DIR, CSV_PATH

import sys

# Ajoute le dossier parent (racine du projet) au chemin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Maintenant, tu peux importer depuis src
from config import IMG_DIR
#from src.data_loader import load_and_explore_data


def load_and_explore_data(n_samples=6):
    # Initialisation de MLflow
    mlflow.set_experiment("Image Data Exploration")
    # Création des dossiers si nécessaire
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR, exist_ok=True)

    # Chargement du fichier CSV
    labels = pd.read_csv(CSV_PATH)

    # Nettoyage des colonnes inutiles
    labels.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)

    # Affichage d'infos de base
    num_images = labels.shape[0]
    num_classes = labels.target.nunique()
    unique_classes = labels.target.unique()
    print(f'Nombre total d\'images : {num_images}')
    print(f'Nombre de classes : {num_classes}')
    print(f'Classes : {unique_classes}')

    # Logging dans MLflow
    mlflow.log_param("csv_path", CSV_PATH)
    mlflow.log_param("image_directory", IMG_DIR)
    mlflow.log_metric("number_of_images", num_images)
    mlflow.log_metric("number_of_classes", num_classes)

    # Visualisation d'images
    visualization_path = "visualizations/class_samples.png"
    plt.figure(figsize=(20, 40))
    for i, (_, row) in enumerate(labels.head(n_samples).iterrows(), 1):
        img_path = os.path.join(IMG_DIR, row['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = plt.subplot(n_samples, 2, i)
        ax.imshow(img)
        ax.set_title(row['target'])

    plt.tight_layout()
    os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
    plt.savefig(visualization_path)
    plt.close()

    # Enregistrement de la visualisation dans MLflow
    mlflow.log_artifact(visualization_path, "visualizations")
    return labels
    


