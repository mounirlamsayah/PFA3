#!/bin/bash

# deploy.sh - Script de déploiement automatisé pour le modèle de classification de pneumothorax

set -e  # Arrêter le script en cas d'erreur

# Configuration
PROJECT_NAME="pneumothorax-classifier"
VERSION=${1:-"latest"}
ENVIRONMENT=${2:-"staging"}
PORT=${3:-5000}

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonctions utilitaires
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
cat << "EOF"
🏥 ====================================================
   PNEUMOTHORAX CLASSIFIER - DEPLOYMENT SCRIPT
   ====================================================
EOF

log_info "Starting deployment for environment: $ENVIRONMENT"
log_info "Version: $VERSION"
log_info "Port: $PORT"

# Vérification des prérequis
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Vérifier Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Vérifier Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Vérifier les fichiers nécessaires
    if [ ! -f "Dockerfile" ]; then
        log_error "Dockerfile not found in current directory"
        exit 1
    fi
    
    if [ ! -f "app.py" ]; then
        log_error "app.py not found in current directory"
        exit 1
    fi
    
    if [ ! -f "outputs/trained_model.h5" ]; then
        log_error "Model file (outputs/trained_model.h5) not found"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Créer les répertoires nécessaires
setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p logs
    mkdir -p monitoring/grafana
    mkdir -p nginx
    mkdir -p mlflow-artifacts
    
    log_success "Directories created"
}

# Créer la configuration Nginx
setup_nginx() {
    log_info "Setting up Nginx configuration..."
    
    cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream pneumothorax_api {
        server pneumothorax-api:5000;
    }
    
    upstream mlflow_server {
        server mlflow-tracking:5000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        # API principal
        location /api/ {
            proxy_pass http://pneumothorax_api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeout pour les prédictions longues
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }
        
        # MLflow UI
        location /mlflow/ {
            proxy_pass http://mlflow_server/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Health check
        location /health {
            proxy_pass http://pneumothorax_api/health;
        }
        
        # Page d'accueil
        location / {
            proxy_pass http://pneumothorax_api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
EOF
    
    log_success "Nginx configuration created"
}

# Configuration Prometheus
setup_monitoring() {
    log_info "Setting up monitoring configuration..."
    
    mkdir -p monitoring
    
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'pneumothorax-api'
    static_configs:
      - targets: ['pneumothorax-api:5000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'mlflow'
    static_configs:
      - targets: ['mlflow-tracking:5000']
    scrape_interval: 60s
EOF
    
    log_success "Monitoring configuration created"
}

# Build de l'image Docker
build_image() {
    log_info "Building Docker image..."
    
    docker build \
        --tag $PROJECT_NAME:$VERSION \
        --tag $PROJECT_NAME:latest \
        --build-arg VERSION=$VERSION \
        .
    
    log_success "Docker image built successfully"
}

# Test du modèle avant déploiement
test_model() {
    log_info "Testing model before deployment..."
    
    python3 << 'EOF'
import os
import numpy as np
from keras.models import load_model

try:
    # Test du chargement du modèle
    model_path = './outputs/trained_model.h5'
    model = load_model(model_path, compile=False)
    print("✅ Model loaded successfully")
    
    # Test de prédiction avec données factices
    dummy_input = np.random.random((1, 2048))
    prediction = model.predict(dummy_input, verbose=0)
    print(f"✅ Prediction test successful: {prediction.shape}")
    
    # Validation format de sortie
    assert prediction.shape == (1, 2), f"Expected (1, 2), got {prediction.shape}"
    print("✅ Model validation passed")
    
except Exception as e:
    print(f"❌ Model test failed: {e}")
    exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        log_success "Model tests passed"
    else
        log_error "Model tests failed"
        exit 1
    fi
}

# Déploiement selon l'environnement
deploy_staging() {
    log_info "Deploying to staging environment..."
    
    # Arrêter les conteneurs existants
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Démarrer les services
    docker-compose up -d
    
    # Attendre que les services démarrent
    log_info "Waiting for services to start..."
    sleep 30
    
    # Vérification de santé
    check_health_staging
}

deploy_production() {
    log_info "Deploying to production environment..."
    log_warning "Production deployment requires additional validation"
    
    # Validation supplémentaire pour la production
    read -p "Are you sure you want to deploy to production? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log_info "Production deployment cancelled"
        exit 0
    fi
    
    # Backup de la version précédente
    docker tag $PROJECT_NAME:latest $PROJECT_NAME:backup-$(date +%Y%m%d-%H%M%S) 2>/dev/null || true
    
    # Déploiement avec rollback automatique en cas d'échec
    if ! deploy_staging; then
        log_error "Deployment failed, rolling back..."
        docker tag $PROJECT_NAME:backup-$(date +%Y%m%d-%H%M%S) $PROJECT_NAME:latest 2>/dev/null || true
        docker-compose up -d
        exit 1
    fi
    
    log_success "Production deployment completed"
}

# Tests de santé
check_health_staging() {
    log_info "Running health checks..."
    
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log_info "Health check attempt $attempt/$max_attempts"
        
        # Test API principale
        if curl -f -s http://localhost:$PORT/health > /dev/null; then
            log_success "API health check passed"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            log_error "Health checks failed after $max_attempts attempts"
            docker-compose logs
            exit 1
        fi
        
        sleep 10
        ((attempt++))
    done
    
    # Test de prédiction
    log_info "Testing prediction endpoint..."
    
    # Créer une image de test simple
    python3 << 'EOF'
import requests
import numpy as np
from PIL import Image
import io

# Créer une image de test
test_image = Image.new('RGB', (299, 299), color='gray')
img_byte_arr = io.BytesIO()
test_image.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)

try:
    # Test de l'endpoint de prédiction
    response = requests.post(
        'http://localhost:5000/predict',
        files={'image': ('test.png', img_byte_arr, 'image/png')},
        timeout=60
    )
    
    if response.status_code == 200:
        print("✅ Prediction endpoint test passed")
    else:
        print(f"❌ Prediction endpoint test failed: {response.status_code}")
        print(response.text)
        exit(1)
        
except Exception as e:
    print(f"❌ Prediction test failed: {e}")
    exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        log_success "All health checks passed"
    else
        log_error "Health checks failed"
        exit 1
    fi
}

# Performance benchmarking
run_benchmarks() {
    log_info "Running performance benchmarks..."
    
    python3 << 'EOF'
import time
import requests
import numpy as np
from PIL import Image
import io

def create_test_image():
    """Créer une image de test"""
    test_image = Image.new('RGB', (299, 299), color='gray')
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def benchmark_prediction():
    """Benchmark des prédictions"""
    times = []
    
    for i in range(5):
        test_image = create_test_image()
        
        start_time = time.time()
        response = requests.post(
            'http://localhost:5000/predict',
            files={'image': ('test.png', test_image, 'image/png')},
            timeout=60
        )
        end_time = time.time()
        
        if response.status_code == 200:
            times.append(end_time - start_time)
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
    
    avg_time = np.mean(times)
    print(f"📊 Average prediction time: {avg_time:.3f}s")
    print(f"📊 Min time: {min(times):.3f}s")
    print(f"📊 Max time: {max(times):.3f}s")
    
    # Seuil de performance
    if avg_time > 10.0:  # 10 secondes max
        print("⚠️  Warning: Average prediction time exceeds 10 seconds")
        return False
    
    print("✅ Performance benchmarks passed")
    return True

if not benchmark_prediction():
    exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        log_success "Performance benchmarks completed"
    else
        log_warning "Performance benchmarks failed"
    fi
}

# Nettoyage
cleanup() {
    log_info "Cleaning up temporary files..."
    
    # Supprimer les images Docker non utilisées
    docker image prune -f > /dev/null 2>&1 || true
    
    log_success "Cleanup completed"
}

# Afficher les informations de déploiement
show_deployment_info() {
    log_success "🎉 Deployment completed successfully!"
    echo ""
    echo "🔗 Service URLs:"
    echo "   • Main API: http://localhost:$PORT"
    echo "   • Health Check: http://localhost:$PORT/health"
    echo "   • Model Info: http://localhost:$PORT/model/info"
    echo "   • MLflow UI: http://localhost:5001"
    echo "   • Prometheus: http://localhost:9090"
    echo "   • Grafana: http://localhost:3000 (admin/admin123)"
    echo ""
    echo "📊 Monitoring:"
    echo "   • Logs: docker-compose logs -f"
    echo "   • Status: docker-compose ps"
    echo ""
    echo "🛑 To stop services: docker-compose down"
}

# Menu d'aide
show_help() {
    cat << EOF
Usage: $0 [VERSION] [ENVIRONMENT] [PORT]

Arguments:
  VERSION      Docker image version (default: latest)
  ENVIRONMENT  Deployment environment: staging|production (default: staging)
  PORT         Service port (default: 5000)

Examples:
  $0                           # Deploy latest to staging on port 5000
  $0 v1.2.0 staging 8080      # Deploy v1.2.0 to staging on port 8080
  $0 latest production 5000    # Deploy to production

Options:
  --help, -h   Show this help message
  --test-only  Run tests only without deployment
  --cleanup    Cleanup Docker resources only

Environment Variables:
  PROJECT_NAME  Project name for Docker images (default: pneumothorax-classifier)
EOF
}

# Fonction principale
main() {
    case "${1:-}" in
        --help|-h)
            show_help
            exit 0
            ;;
        --test-only)
            check_prerequisites
            test_model
            exit 0
            ;;
        --cleanup)
            cleanup
            exit 0
            ;;
    esac
    
    log_info "🚀 Starting deployment process..."
    
    # Étapes de déploiement
    check_prerequisites
    setup_directories
    setup_nginx
    setup_monitoring
    test_model
    build_image
    
    # Déploiement selon l'environnement
    case $ENVIRONMENT in
        staging)
            deploy_staging
            ;;
        production)
            deploy_production
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            show_help
            exit 1
            ;;
    esac
    
    # Tests post-déploiement
    run_benchmarks
    cleanup
    show_deployment_info
}

# Gestion des signaux pour nettoyage en cas d'interruption
trap cleanup EXIT INT TERM

# Exécution du script principal
main "$@"