import pickle
from keras.models import load_model
from keras.metrics import AUC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des features
with open('../outputs/features/xception_features_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('../outputs/features/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Chargement du modèle
model = load_model('../outputs/trained_model.h5', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[AUC(name='auc')])

# --------- MLflow run ----------
mlflow.set_experiment("MLOPS Pipeline")

with mlflow.start_run(run_name="evaluation du modele"):

    # Évaluation
    test_loss, test_auc = model.evaluate(X_test, y_test, verbose=1)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_auc", test_auc)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Rapport de classification
    report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))

    # Log des métriques par classe
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)

    # Matrice de confusion (optionnelle)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice de confusion")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.show()
