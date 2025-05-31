# app.py - Application Flask pour servir le modèle de classification de pneumothorax

import os
import json
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template_string
from keras.models import load_model
from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.models import Model
from keras.layers import Input, Lambda, GlobalAveragePooling2D
import cv2
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MODEL_PATH = os.environ.get('MODEL_PATH', '../outputs/trained_model.h5')
IMG_SIZE = (299, 299, 3)

# Variables globales pour les modèles
model = None
feature_extractor = None

def load_models():
    """Charge le modèle principal et l'extracteur de features"""
    global model, feature_extractor
    
    try:
        # Charger le modèle principal
        model = load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("✅ Modèle principal chargé avec succès")
        
        # Créer l'extracteur de features Xception
        input_layer = Input(shape=IMG_SIZE)
        x = Lambda(preprocess_input)(input_layer)
        base_model = Xception(weights='imagenet', include_top=False, input_shape=IMG_SIZE)(x)
        x = GlobalAveragePooling2D()(base_model)
        feature_extractor = Model(inputs=input_layer, outputs=x)
        logger.info("✅ Extracteur de features Xception chargé avec succès")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
        return False

def extract_features(image_array):
    """Extrait les features d'une image avec Xception"""
    try:
        # Redimensionner l'image si nécessaire
        if image_array.shape != IMG_SIZE:
            image_array = cv2.resize(image_array, IMG_SIZE[:2])
        
        # Ajouter la dimension batch
        image_batch = np.expand_dims(image_array, axis=0)
        
        # Extraire les features
        features = feature_extractor.predict(image_batch, verbose=0)
        return features
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'extraction de features: {e}")
        return None

def predict_pneumothorax(image_array):
    """Prédit la présence de pneumothorax dans une image"""
    try:
        # Extraire les features
        features = extract_features(image_array)
        if features is None:
            return None
        
        # Faire la prédiction
        prediction = model.predict(features, verbose=0)
        
        # Interpréter les résultats
        prob_no_pneumothorax = float(prediction[0][0])
        prob_pneumothorax = float(prediction[0][1])
        
        result = {
            'no_pneumothorax_probability': prob_no_pneumothorax,
            'pneumothorax_probability': prob_pneumothorax,
            'predicted_class': 'Pneumothorax' if prob_pneumothorax > prob_no_pneumothorax else 'No Pneumothorax',
            'confidence': max(prob_no_pneumothorax, prob_pneumothorax),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction: {e}")
        return None

# Template HTML pour l'interface web
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🏥 Pneumothorax Classifier</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container { 
            background: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header { 
            text-align: center; 
            color: #2c3e50; 
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            background-color: #ecf0f1;
        }
        .btn {
            background-color: #3498db;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn:hover { background-color: #2980b9; }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 5px;
            font-weight: bold;
        }
        .positive { background-color: #e74c3c; color: white; }
        .negative { background-color: #27ae60; color: white; }
        .info { background-color: #f39c12; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Pneumothorax Classification System</h1>
            <p>Upload a chest X-ray image for pneumothorax detection</p>
        </div>
        
        <form action="/predict" method="post" enctype="multipart/form-data">
            <div class="upload-area">
                <h3>📤 Select Chest X-ray Image</h3>
                <input type="file" name="image" accept="image/*" required>
                <br><br>
                <button type="submit" class="btn">🔍 Analyze Image</button>
            </div>
        </form>
        
        <div style="margin-top: 30px;">
            <h3>📊 API Endpoints:</h3>
            <ul>
                <li><strong>GET /health</strong> - Health check</li>
                <li><strong>POST /predict</strong> - Image classification</li>
                <li><strong>GET /model/info</strong> - Model information</li>
            </ul>
        </div>
        
        <div style="margin-top: 20px; text-align: center; color: #7f8c8d;">
            <small>⚠️ This is a research tool. Always consult healthcare professionals for medical diagnosis.</small>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    """Page d'accueil avec interface de upload"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de santé pour vérifier le statut du service"""
    try:
        if model is None or feature_extractor is None:
            return jsonify({
                'status': 'unhealthy',
                'message': 'Models not loaded',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        return jsonify({
            'status': 'healthy',
            'message': 'Service is running',
            'model_loaded': True,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Informations sur le modèle"""
    try:
        info = {
            'model_architecture': 'Xception + Dense layers',
            'task': 'Binary Classification',
            'classes': ['No Pneumothorax', 'Pneumothorax'],
            'input_size': IMG_SIZE,
            'model_path': MODEL_PATH,
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
        
        if model is not None:
            info['model_summary'] = {
                'total_params': model.count_params(),
                'trainable_params': sum([np.prod(v.get_shape()) for v in model.trainable_weights]),
            }
        
        return jsonify(info), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to get model info',
            'message': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    try:
        # Vérifier si un fichier a été uploadé
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'message': 'Please upload an image file'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select an image file'
            }), 400
        
        # Lire et traiter l'image
        image_bytes = file.read()
        image_array = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8), 
            cv2.IMREAD_COLOR
        )
        
        if image_array is None:
            return jsonify({
                'error': 'Invalid image format',
                'message': 'Could not decode the image file'
            }), 400
        
        # Convertir BGR en RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Faire la prédiction
        result = predict_pneumothorax(image_array)
        
        if result is None:
            return jsonify({
                'error': 'Prediction failed',
                'message': 'Could not process the image'
            }), 500
        
        # Retourner les résultats
        if request.content_type == 'application/json':
            return jsonify(result), 200
        else:
            # Interface web - retourner HTML
            result_class = 'positive' if result['predicted_class'] == 'Pneumothorax' else 'negative'
            result_html = f"""
            <div class="result {result_class}">
                <h3>🔍 Analysis Results</h3>
                <p><strong>Prediction:</strong> {result['predicted_class']}</p>
                <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                <p><strong>Pneumothorax Probability:</strong> {result['pneumothorax_probability']:.2%}</p>
                <p><strong>Analysis Time:</strong> {result['timestamp']}</p>
            </div>
            """
            
            return render_template_string(HTML_TEMPLATE + result_html)
        
    except Exception as e:
        logger.error(f"❌ Erreur dans /predict: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    # Charger les modèles au démarrage
    logger.info("🚀 Démarrage de l'application Pneumothorax Classifier...")
    
    if load_models():
        logger.info("✅ Application prête à recevoir des requêtes")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("❌ Impossible de démarrer l'application - échec du chargement des modèles")
        exit(1)