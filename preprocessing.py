import os
import numpy as np
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from config import IMG_DIR
from data_loader import load_and_explore_data

# Démarrage d'une expérimentation MLflow
mlflow.set_experiment("MLOPS Pipeline")

with mlflow.start_run(run_name="Préparation Dataset"):
    labels = load_and_explore_data()

    # Mapping des classes
    classes = sorted(labels['target'].unique())
    classes_to_num = dict(zip(classes, range(len(classes))))

    # Log d'infos sur le dataset
    mlflow.log_param("Nombre_de_classes", len(classes))
    mlflow.log_param("Classes", classes)
    mlflow.log_param("Taille_du_dataset", len(labels))

    # Visualisation : histogramme des classes
    class_counts = labels['target'].value_counts()
    plt.figure(figsize=(8, 5))
    class_counts.plot(kind='bar', title='Nombre_images_par_classe')
    plt.xlabel('Classe')
    plt.ylabel('Nombre_images')
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    mlflow.log_artifact("class_distribution.png")

    # Chargement et conversion des images en tableau numpy
    def images_to_array(data_dir, df, image_size):
        image_names = df['file_name']
        image_labels = df['target']
        data_size = len(image_names)

        X = np.zeros([data_size, *image_size], dtype=np.uint8)
        y = np.zeros([data_size, 1], dtype=np.uint8)

        for i in range(data_size):
            img_path = os.path.join(data_dir, image_names[i])
            img = load_img(img_path, target_size=image_size[:2])
            X[i] = img
            y[i] = classes_to_num[image_labels[i]]

        y = to_categorical(y)

        # Log forme des données
        mlflow.log_param("Image_size", image_size)
        mlflow.log_metric("Nombre_total_images", data_size)
        mlflow.log_metric("Dimensions_X", X.shape[1])
        mlflow.log_metric("Dimensions_Y", X.shape[2])
        mlflow.log_metric("Canaux", X.shape[3])

        ind = np.random.permutation(data_size)
        return X[ind], y[ind]

    # Exemple d'appel avec image_size (ex: (128, 128, 3))
    image_size = (128, 128, 3)
    X, y = images_to_array(IMG_DIR, labels, image_size)
