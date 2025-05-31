from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.models import Model
from keras.layers import Input, Lambda, GlobalAveragePooling2D
from preprocessing import images_to_array
from config import IMG_DIR
from sklearn.model_selection import train_test_split
from data_loader import load_and_explore_data
import pickle
import os
import mlflow
import mlflow.keras

# Démarrer une expérience MLflow
mlflow.set_experiment("MLOPS Pipeline")

with mlflow.start_run(run_name="Xception_Feature_Extraction"):

    labels = load_and_explore_data()

    # Fonction d'extraction de features
    def get_features(model_name, data_preprocessor, weight, input_size, data):
        input_layer = Input(shape=input_size)
        x = Lambda(data_preprocessor)(input_layer)
        base_model = model_name(weights=weight, include_top=False, input_shape=input_size)(x)
        x = GlobalAveragePooling2D()(base_model)
        feature_extractor = Model(inputs=input_layer, outputs=x)
        
        features = feature_extractor.predict(data, batch_size=128, verbose=1)
        return features

    # Chargement des images
    img_size = (299, 299, 3)
    X, y = images_to_array(IMG_DIR, labels, img_size)

    mlflow.log_param("Image size", img_size)
    mlflow.log_param("Total images", X.shape[0])
    mlflow.log_param("Number of classes", y.shape[1])

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    mlflow.log_metric("Train size", len(X_train))
    mlflow.log_metric("Test size", len(X_test))

    # Extraction des features
    Xception_features_train = get_features(
        Xception, preprocess_input, 'imagenet', img_size, X_train)
    Xception_features_test = get_features(
        Xception, preprocess_input, 'imagenet', img_size, X_test)

    # Sauvegarde des features
    os.makedirs('../outputs/features', exist_ok=True)

    with open('../outputs/features/xception_features_train.pkl', 'wb') as f:
        pickle.dump(Xception_features_train, f)
    with open('../outputs/features/xception_features_test.pkl', 'wb') as f:
        pickle.dump(Xception_features_test, f)
    with open('../outputs/features/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('../outputs/features/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)

    # Log des fichiers comme artéfacts
    mlflow.log_artifact('../outputs/features/xception_features_train.pkl')
    mlflow.log_artifact('../outputs/features/xception_features_test.pkl')
    mlflow.log_artifact('../outputs/features/y_train.pkl')
    mlflow.log_artifact('../outputs/features/y_test.pkl')
