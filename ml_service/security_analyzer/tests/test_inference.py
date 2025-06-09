import unittest
import json
from ..model.classifier import SecurityAnalyzer
from ..utils.schema import SecurityAnalysisRequest
from ..data.synthetic_generator import SyntheticDataGenerator

class TestSecurityInference(unittest.TestCase):
    def setUp(self):
        self.analyzer = SecurityAnalyzer()
        self.generator = SyntheticDataGenerator()
        
        # Always train with fresh data for each test
        training_data = self.generator.generate_dataset(2000)  # Increase training size
        training_success = self.analyzer.train(training_data)
        
        if not training_success or not self.analyzer.is_trained:
            self.fail("Failed to train analyzer for testing")
    
    def test_xss_detection(self):
        """Test XSS attack detection"""
        request = SecurityAnalysisRequest(
            method="POST",
            path="/comment",
            headers={"Content-Type": "application/json"},  # Ensure headers is dict
            body='{"comment": "<script>alert(\'XSS\')</script>"}',  # Use JSON string
            ip_address="192.168.1.100"
        )
        
        result = self.analyzer.analyze(request)
        
        self.assertTrue(result.is_malicious)
        self.assertEqual(result.attack_type, "xss")
        self.assertGreater(result.attack_score, 0.5)
    
    def test_sqli_detection(self):
        """Test SQL injection detection"""
        request = SecurityAnalysisRequest(
            method="POST",
            path="/login",
            headers={"Content-Type": "application/json"},
            body='{"username": "admin", "password": "\' OR 1=1 --"}',  # Use JSON string
            ip_address="203.0.113.42"
        )
        
        result = self.analyzer.analyze(request)
        
        self.assertTrue(result.is_malicious)
        self.assertEqual(result.attack_type, "sqli")
        self.assertGreater(result.attack_score, 0.7)
    
    def test_csrf_detection(self):
        """Test CSRF attack detection"""
        request = SecurityAnalysisRequest(
            method="POST",
            path="/transfer",
            headers={  # Ensure headers is properly formatted dict
                "Referer": "https://attacker.com",
                "Content-Type": "application/json"
            },
            body='{"amount": "1000", "to_account": "attacker"}',  # Use JSON string
            cookies={},  # Empty dict instead of None
            ip_address="198.51.100.42"
        )
        
        result = self.analyzer.analyze(request)
        
        self.assertTrue(result.is_malicious)
        self.assertEqual(result.attack_type, "csrf")
    
    def test_benign_request(self):
        """Test benign request handling"""
        request = SecurityAnalysisRequest(
            method="GET",
            path="/products",
            query_params="category=electronics&page=1",  # Use string format
            headers={"User-Agent": "Mozilla/5.0"},
            body="",  # Empty string
            cookies={"session": "valid_session"},  # Provide valid cookies
            ip_address="192.168.1.1"
        )
        
        result = self.analyzer.analyze(request)
        
        self.assertFalse(result.is_malicious)
        self.assertEqual(result.attack_type, "benign")
    
    def test_ensemble_decision(self):
        """Test ensemble decision making"""
        request = SecurityAnalysisRequest(
            method="GET",
            path="/search",
            query_params="q=select from products where price > 100",  # String format
            headers={"User-Agent": "Mozilla/5.0"},
            body="",
            ip_address="10.0.0.1"
        )
        
        result = self.analyzer.analyze(request)
        
        self.assertIsNotNone(result.confidence)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

if __name__ == '__main__':
    unittest.main() 