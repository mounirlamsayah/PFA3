# test_deployment.py - Tests automatisés pour valider le déploiement

import requests
import time
import json
import numpy as np
from PIL import Image
import io
import os
import sys
from datetime import datetime

class DeploymentTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
        
    def log_test(self, test_name, success, message="", details=None):
        """Enregistrer le résultat d'un test"""
        result = {
            'test_name': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        
    def create_test_image(self, size=(299, 299), color='gray'):
        """Créer une image de test"""
        test_image = Image.new('RGB', size, color=color)
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr
        
    def test_service_availability(self):
        """Test 1: Vérifier que le service est accessible"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                self.log_test("Service Availability", True, "Service is accessible")
                return True
            else:
                self.log_test("Service Availability", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Service Availability", False, f"Connection failed: {str(e)}")
            return False
            
    def test_health_endpoint(self):
        """Test 2: Vérifier l'endpoint de santé"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    self.log_test("Health Endpoint", True, "Service is healthy", data)
                    return True
                else:
                    self.log_test("Health Endpoint", False, f"Unhealthy status: {data.get('status')}")
                    return False
            else:
                self.log_test("Health Endpoint", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Health Endpoint", False, f"Request failed: {str(e)}")
            return False
            
    def test_model_info_endpoint(self):
        """Test 3: Vérifier l'endpoint d'informations du modèle"""
        try:
            response = requests.get(f"{self.base_url}/model/info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                required_fields = ['model_architecture', 'task', 'classes', 'input_size']
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    self.log_test("Model Info Endpoint", True, "All required fields present", data)
                    return True
                else:
                    self.log_test("Model Info Endpoint", False, f"Missing fields: {missing_fields}")
                    return False
            else:
                self.log_test("Model Info Endpoint", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Model Info Endpoint", False, f"Request failed: {str(e)}")
            return False
            
    def test_prediction_endpoint(self):
        """Test 4: Vérifier l'endpoint de prédiction avec une image de test"""
        try:
            test_image = self.create_test_image()
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/predict",
                files={'image': ('test.png', test_image, 'image/png')},
                timeout=60
            )
            end_time = time.time()
            
            prediction_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['no_pneumothorax_probability', 'pneumothorax_probability', 
                                 'predicted_class', 'confidence', 'timestamp']
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    details = {
                        'prediction_time': prediction_time,
                        'prediction_data': data
                    }
                    self.log_test("Prediction Endpoint", True, 
                                f"Prediction successful in {prediction_time:.2f}s", details)
                    return True
                else:
                    self.log_test("Prediction Endpoint", False, f"Missing fields: {missing_fields}")
                    return False
            else:
                self.log_test("Prediction Endpoint", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Prediction Endpoint", False, f"Request failed: {str(e)}")
            return False
            
    def test_prediction_validation(self):
        """Test 5: Valider le format des prédictions"""
        try:
            test_image = self.create_test_image()
            
            response = requests.post(
                f"{self.base_url}/predict",
                files={'image': ('test.png', test_image, 'image/png')},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Vérifier les probabilités
                prob_no_pneumo = data.get('no_pneumothorax_probability', 0)
                prob_pneumo = data.get('pneumothorax_probability', 0)
                
                # Validation des probabilités
                if not (0 <= prob_no_pneumo <= 1 and 0 <= prob_pneumo <= 1):
                    self.log_test("Prediction Validation", False, "Probabilities out of range [0,1]")
                    return False
                    
                # Vérifier que les probabilités somment à ~1
                prob_sum = prob_no_pneumo + prob_pneumo
                if not (0.9 <= prob_sum <= 1.1):
                    self.log_test("Prediction Validation", False, f"Probability sum invalid: {prob_sum}")
                    return False
                    
                # Vérifier la classe prédite
                predicted_class = data.get('predicted_class')
                if predicted_class not in ['No Pneumothorax', 'Pneumothorax']:
                    self.log_test("Prediction Validation", False, f"Invalid predicted class: {predicted_class}")
                    return False
                    
                # Vérifier la confiance
                confidence = data.get('confidence', 0)
                expected_confidence = max(prob_no_pneumo, prob_pneumo)
                if abs(confidence - expected_confidence) > 0.01:
                    self.log_test("Prediction Validation", False, "Confidence value inconsistent")
                    return False
                    
                self.log_test("Prediction Validation", True, "All validation checks passed")
                return True
            else:
                self.log_test("Prediction Validation", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Prediction Validation", False, f"Validation failed: {str(e)}")
            return False
            
    def test_error_handling(self):
        """Test 6: Vérifier la gestion d'erreurs"""
        tests_passed = 0
        total_tests = 3
        
        # Test 1: Requête sans image
        try:
            response = requests.post(f"{self.base_url}/predict", timeout=10)
            if response.status_code == 400:
                tests_passed += 1
                print("  ✅ No image error handling: OK")
            else:
                print(f"  ❌ No image error handling: Expected 400, got {response.status_code}")
        except Exception as e:
            print(f"  ❌ No image error handling failed: {str(e)}")
            
        # Test 2: Image invalide
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                files={'image': ('test.txt', io.BytesIO(b'not an image'), 'text/plain')},
                timeout=10
            )
            if response.status_code == 400:
                tests_passed += 1
                print("  ✅ Invalid image error handling: OK")
            else:
                print(f"  ❌ Invalid image error handling: Expected 400, got {response.status_code}")
        except Exception as e:
            print(f"  ❌ Invalid image error handling failed: {str(e)}")
            
        # Test 3: Endpoint inexistant
        try:
            response = requests.get(f"{self.base_url}/nonexistent", timeout=10)
            if response.status_code == 404:
                tests_passed += 1
                print("  ✅ 404 error handling: OK")
            else:
                print(f"  ❌ 404 error handling: Expected 404, got {response.status_code}")
        except Exception as e:
            print(f"  ❌ 404 error handling failed: {str(e)}")
            
        success = tests_passed == total_tests
        self.log_test("Error Handling", success, f"{tests_passed}/{total_tests} error handling tests passed")
        return success
        
    def test_performance_benchmark(self, num_requests=10):
        """Test 7: Benchmark de performance"""
        try:
            prediction_times = []
            successful_requests = 0
            
            print(f"  Running {num_requests} prediction requests...")
            
            for i in range(num_requests):
                test_image = self.create_test_image()
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/predict",
                    files={'image': ('test.png', test_image, 'image/png')},
                    timeout=60
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    prediction_times.append(end_time - start_time)
                    successful_requests += 1
                    
                # Petite pause entre les requêtes
                time.sleep(0.1)
                
            if successful_requests > 0:
                avg_time = np.mean(prediction_times)
                min_time = np.min(prediction_times)
                max_time = np.max(prediction_times)
                
                details = {
                    'successful_requests': successful_requests,
                    'total_requests': num_requests,
                    'avg_prediction_time': avg_time,
                    'min_prediction_time': min_time,
                    'max_prediction_time': max_time,
                    'success_rate': successful_requests / num_requests
                }
                
                # Critères de performance
                performance_ok = (
                    avg_time < 10.0 and  # Moins de 10 secondes en moyenne
                    successful_requests / num_requests > 0.9  # Taux de succès > 90%
                )
                
                message = f"Avg: {avg_time:.2f}s, Success rate: {successful_requests}/{num_requests}"
                self.log_test("Performance Benchmark", performance_ok, message, details)
                return performance_ok
            else:
                self.log_test("Performance Benchmark", False, "No successful requests")
                return False
                
        except Exception as e:
            self.log_test("Performance Benchmark", False, f"Benchmark failed: {str(e)}")
            return False
            
    def test_concurrent_requests(self, num_concurrent=5):
        """Test 8: Requêtes concurrentes"""
        import threading
        
        results = []
        
        def make_request():
            try:
                test_image = self.create_test_image()
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/predict",
                    files={'image': ('test.png', test_image, 'image/png')},
                    timeout=60
                )
                end_time = time.time()
                
                results.append({
                    'success': response.status_code == 200,
                    'time': end_time - start_time,
                    'status_code': response.status_code
                })
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e)
                })
                
        print(f"  Running {num_concurrent} concurrent requests...")
        
        # Lancer les requêtes concurrentes
        threads = []
        for _ in range(num_concurrent):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            
        # Démarrer tous les threads
        start_time = time.time()
        for thread in threads:
            thread.start()
            
        # Attendre que tous les threads se terminent
        for thread in threads:
            thread.join()
        end_time = time.time()
        
        # Analyser les résultats
        successful = sum(1 for r in results if r.get('success', False))
        total_time = end_time - start_time
        
        details = {
            'concurrent_requests': num_concurrent,
            'successful_requests': successful,
            'total_time': total_time,
            'success_rate': successful / num_concurrent
        }
        
        success = successful / num_concurrent > 0.8  # Taux de succès > 80%
        message = f"{successful}/{num_concurrent} successful in {total_time:.2f}s"
        
        self.log_test("Concurrent Requests", success, message, details)
        return success

    def test_stress_load(self, duration_seconds=30):
        """Test 9: Test de charge/stress"""
        import threading
        import time
        
        results = []
        stop_test = False
        
        def make_continuous_requests():
            while not stop_test:
                try:
                    test_image = self.create_test_image()
                    start_time = time.time()
                    response = requests.post(
                        f"{self.base_url}/predict",
                        files={'image': ('test.png', test_image, 'image/png')},
                        timeout=10
                    )
                    end_time = time.time()
                    
                    results.append({
                        'success': response.status_code == 200,
                        'time': end_time - start_time,
                        'status_code': response.status_code,
                        'timestamp': time.time()
                    })
                except Exception as e:
                    results.append({
                        'success': False,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                    
                time.sleep(0.1)  # Petite pause
                
        print(f"  Running stress test for {duration_seconds} seconds...")
        
        # Lancer plusieurs threads pour la charge
        threads = []
        for _ in range(3):  # 3 threads concurrents
            thread = threading.Thread(target=make_continuous_requests)
            threads.append(thread)
            thread.start()
            
        # Laisser tourner pendant la durée spécifiée
        time.sleep(duration_seconds)
        stop_test = True
        
        # Attendre que tous les threads se terminent
        for thread in threads:
            thread.join()
            
        # Analyser les résultats
        successful = sum(1 for r in results if r.get('success', False))
        total_requests = len(results)
        
        if total_requests > 0:
            success_rate = successful / total_requests
            avg_response_time = np.mean([r['time'] for r in results if 'time' in r])
            
            details = {
                'duration_seconds': duration_seconds,
                'total_requests': total_requests,
                'successful_requests': successful,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'requests_per_second': total_requests / duration_seconds
            }
            
            # Critères de succès pour le test de stress
            stress_test_success = (
                success_rate > 0.85 and  # Taux de succès > 85%
                avg_response_time < 15.0  # Temps de réponse moyen < 15s
            )
            
            message = f"RPS: {details['requests_per_second']:.1f}, Success: {success_rate:.1%}"
            self.log_test("Stress Load Test", stress_test_success, message, details)
            return stress_test_success
        else:
            self.log_test("Stress Load Test", False, "No requests completed")
            return False

    def test_memory_usage_monitoring(self):
        """Test 10: Surveillance de l'utilisation mémoire (si possible)"""
        try:
            # Faire plusieurs prédictions et vérifier la cohérence
            initial_prediction = None
            consistent_predictions = 0
            
            for i in range(5):
                test_image = self.create_test_image()
                response = requests.post(
                    f"{self.base_url}/predict",
                    files={'image': ('test.png', test_image, 'image/png')},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if initial_prediction is None:
                        initial_prediction = data
                        consistent_predictions = 1
                    else:
                        # Vérifier la cohérence des prédictions pour la même image
                        prob_diff = abs(
                            data['no_pneumothorax_probability'] - 
                            initial_prediction['no_pneumothorax_probability']
                        )
                        if prob_diff < 0.01:  # Tolérance de 1%
                            consistent_predictions += 1
                            
                time.sleep(1)  # Pause entre les requêtes
                
            consistency_rate = consistent_predictions / 5
            success = consistency_rate > 0.8  # 80% de cohérence
            
            details = {
                'consistency_rate': consistency_rate,
                'consistent_predictions': consistent_predictions,
                'total_predictions': 5
            }
            
            message = f"Consistency: {consistency_rate:.1%}"
            self.log_test("Memory Usage Monitoring", success, message, details)
            return success
            
        except Exception as e:
            self.log_test("Memory Usage Monitoring", False, f"Test failed: {str(e)}")
            return False
        
    def run_all_tests(self):
        """Exécuter tous les tests"""
        print("🧪 Starting deployment tests...")
        print("=" * 50)
        
        tests = [
            self.test_service_availability,
            self.test_health_endpoint,
            self.test_model_info_endpoint,
            self.test_prediction_endpoint,
            self.test_prediction_validation,
            self.test_error_handling,
            self.test_performance_benchmark,
            self.test_concurrent_requests,
            self.test_stress_load,
            self.test_memory_usage_monitoring
        ]
        
        passed_tests = 0
        
        for test in tests:
            try:
                if test():
                    passed_tests += 1
            except Exception as e:
                print(f"❌ Test {test.__name__} crashed: {str(e)}")
                
        print("=" * 50)
        print(f"📊 Test Results: {passed_tests}/{len(tests)} tests passed")
        
        if passed_tests == len(tests):
            print("🎉 All tests passed! Deployment is ready.")
            return True
        else:
            print("⚠️  Some tests failed. Please check the deployment.")
            return False
            
    def generate_report(self):
        """Générer un rapport de test"""
        report = {
            'test_summary': {
                'total_tests': len(self.test_results),
                'passed_tests': sum(1 for r in self.test_results if r['success']),
                'failed_tests': sum(1 for r in self.test_results if not r['success']),
                'timestamp': datetime.now().isoformat()
            },
            'detailed_results': self.test_results
        }
        
        # Sauvegarder le rapport
        with open('deployment_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"📄 Test report saved to: deployment_test_report.json")
        return report

def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test deployment of Pneumothorax Classifier')
    parser.add_argument('--url', default='http://localhost:5000', 
                       help='Base URL of the service (default: http://localhost:5000)')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick tests only (skip performance benchmarks)')
    
    args = parser.parse_args()
    
    tester = DeploymentTester(args.url)
    
    if args.quick:
        # Tests rapides seulement
        tests = [
            tester.test_service_availability,
            tester.test_health_endpoint,
            tester.test_model_info_endpoint,
            tester.test_prediction_endpoint,
            tester.test_prediction_validation,
            tester.test_error_handling
        ]
        
        print("🧪 Running quick tests...")
        print("=" * 50)
        
        passed_tests = 0
        for test in tests:
            try:
                if test():
                    passed_tests += 1
            except Exception as e:
                print(f"❌ Test {test.__name__} crashed: {str(e)}")
                
        print("=" * 50)
        print(f"📊 Quick Test Results: {passed_tests}/{len(tests)} tests passed")
        
        if passed_tests == len(tests):
            print("🎉 All quick tests passed!")
        else:
            print("⚠️  Some tests failed.")
    else:
        # Tests complets
        success = tester.run_all_tests()
        
    # Générer le rapport
    tester.generate_report()
    
    return 0 if (args.quick and passed_tests == len(tests)) or (not args.quick and success) else 1

if __name__ == "__main__":
    exit(main())