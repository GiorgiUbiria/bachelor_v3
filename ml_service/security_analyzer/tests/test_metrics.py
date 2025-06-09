import unittest
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from ..model.classifier import SecurityAnalyzer
from ..data.synthetic_generator import SyntheticDataGenerator
from ..utils.schema import SecurityAnalysisRequest

class TestSecurityMetrics(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = SecurityAnalyzer()
        self.generator = SyntheticDataGenerator()
        
        self.test_data = self.generator.generate_evaluation_set(100)  # 400 samples total
        
        if not self.analyzer.is_trained:
            training_data = self.generator.generate_dataset(1000)
            self.analyzer.train(training_data)
    
    def test_accuracy_calculation(self):
        predictions = []
        true_labels = []
        
        for sample in self.test_data[:50]:
            request = SecurityAnalysisRequest(**sample)
            result = self.analyzer.analyze(request)
            
            predictions.append(result.attack_type)
            true_labels.append(sample['attack_type'])
        
        accuracy = accuracy_score(true_labels, predictions)
        
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        self.assertGreater(accuracy, 0.5)
        
        print(f"Model Accuracy: {accuracy:.4f}")
    
    def test_precision_recall_fscore(self):
        predictions = []
        true_labels = []
        
        for sample in self.test_data[:50]:
            request = SecurityAnalysisRequest(**sample)
            result = self.analyzer.analyze(request)
            
            predictions.append(result.attack_type)
            true_labels.append(sample['attack_type'])
        
        precision, recall, fscore, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        self.assertIsInstance(precision, float)
        self.assertIsInstance(recall, float)
        self.assertIsInstance(fscore, float)
        
        self.assertGreaterEqual(precision, 0.0)
        self.assertGreaterEqual(recall, 0.0)
        self.assertGreaterEqual(fscore, 0.0)
        
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {fscore:.4f}")
    
    def test_confusion_matrix_structure(self):
        predictions = []
        true_labels = []
        
        for sample in self.test_data[:50]:
            request = SecurityAnalysisRequest(**sample)
            result = self.analyzer.analyze(request)
            
            predictions.append(result.attack_type)
            true_labels.append(sample['attack_type'])
        
        labels = ['benign', 'csrf', 'sqli', 'xss']
        cm = confusion_matrix(true_labels, predictions, labels=labels)
        
        self.assertEqual(cm.shape, (4, 4))
        self.assertGreaterEqual(cm.sum(), len(predictions))
        
        diagonal_sum = np.trace(cm)
        self.assertGreater(diagonal_sum, 0)
        
        print(f"Confusion Matrix:\n{cm}")
    
    def test_attack_type_distribution(self):
        attack_type_results = {'benign': 0, 'xss': 0, 'sqli': 0, 'csrf': 0}
        
        for sample in self.test_data[:100]:
            request = SecurityAnalysisRequest(**sample)
            result = self.analyzer.analyze(request)
            
            if result.attack_type in attack_type_results:
                attack_type_results[result.attack_type] += 1
        
        detected_types = sum(1 for count in attack_type_results.values() if count > 0)
        self.assertGreaterEqual(detected_types, 3)
        
        print(f"Attack Type Distribution: {attack_type_results}")
    
    def test_confidence_score_validity(self):
        confidence_scores = []
        
        for sample in self.test_data[:50]:
            request = SecurityAnalysisRequest(**sample)
            result = self.analyzer.analyze(request)
            confidence_scores.append(result.confidence)
        
        for confidence in confidence_scores:
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
        
        avg_confidence = np.mean(confidence_scores)
        self.assertGreater(avg_confidence, 0.3)
        
        print(f"Average Confidence: {avg_confidence:.4f}")
    
    def test_attack_score_validity(self):
        benign_scores = []
        malicious_scores = []
        
        for sample in self.test_data[:50]:
            request = SecurityAnalysisRequest(**sample)
            result = self.analyzer.analyze(request)
            
            if result.is_malicious:
                malicious_scores.append(result.attack_score)
            else:
                benign_scores.append(result.attack_score)
        
        all_scores = benign_scores + malicious_scores
        for score in all_scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        if benign_scores and malicious_scores:
            avg_benign = np.mean(benign_scores)
            avg_malicious = np.mean(malicious_scores)
            self.assertGreater(avg_malicious, avg_benign)
            
            print(f"Average Benign Score: {avg_benign:.4f}")
            print(f"Average Malicious Score: {avg_malicious:.4f}")
    
    def test_response_time_metrics(self):
        import time
        response_times = []
        
        for sample in self.test_data[:20]:
            request = SecurityAnalysisRequest(**sample)
            
            start_time = time.time()
            result = self.analyzer.analyze(request)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
        
        avg_response_time = np.mean(response_times)
        max_response_time = max(response_times)
        
        self.assertLess(avg_response_time, 1.0)
        self.assertLess(max_response_time, 2.0)
        
        print(f"Average Response Time: {avg_response_time:.4f}s")
        print(f"Max Response Time: {max_response_time:.4f}s")

if __name__ == '__main__':
    unittest.main() 