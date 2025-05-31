from keras.models import Sequential
from keras.layers import Dropout, Dense, InputLayer
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.metrics import AUC
from keras.regularizers import l2
import matplotlib.pyplot as plt
from preprocessing import images_to_array
from config import IMG_DIR
from data_loader import load_and_explore_data
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pickle
import mlflow
import mlflow.keras
import os

# Pour éviter que MLflow écrase les logs précédents
mlflow.set_experiment("MLOPS Pipeline")

def plot_training_history(history):
    plt.figure(figsize=(10, 4))

    # Courbe de perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])

    # Courbe AUC
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend(['Train', 'Validation'])

    plt.tight_layout()
    plot_path = "training_curves.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)  # Log le graphique dans MLflow
    plt.show()

def train_model(X_features, y, epochs=30, batch_size=128):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model = Sequential()
    model.add(InputLayer(X_features.shape[1:]))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[AUC(name='auc')])
    model.summary()

    y_labels = np.argmax(y, axis=1)
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(y_labels),
                                         y=y_labels)
    class_weights_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weights_dict)

    # Log des paramètres
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("loss", "categorical_crossentropy")
    mlflow.log_param("class_weights", class_weights_dict)

    history = model.fit(
        X_features, y,
        validation_split=0.2,
        callbacks=[early_stop],
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights_dict
    )

    # Log des métriques à chaque époque
    for i, (loss, val_loss, auc, val_auc) in enumerate(zip(
        history.history['loss'],
        history.history['val_loss'],
        history.history['auc'],
        history.history['val_auc']
    )):
        mlflow.log_metric("loss", loss, step=i)
        mlflow.log_metric("val_loss", val_loss, step=i)
        mlflow.log_metric("auc", auc, step=i)
        mlflow.log_metric("val_auc", val_auc, step=i)

    return model, history

# ---------- MLflow run ----------
with mlflow.start_run(run_name="Train Model"):

    with open('../outputs/features/xception_features_train.pkl', 'rb') as f:
        Xception_features_train = pickle.load(f)

    labels = load_and_explore_data()
    IMG_SIZE = (299, 299, 3)
    X, y_train = images_to_array(IMG_DIR, labels, IMG_SIZE)

    model, history = train_model(Xception_features_train, y_train)

    print(history.history['loss'])
    plot_training_history(history)

    # Sauvegarde du modèle
    os.makedirs("../outputs", exist_ok=True)
    model_path = "../outputs/trained_model.h5"
    model.save(model_path)
    mlflow.log_artifact(model_path)
