import unittest
from ..pattern_matching.xss import XSSDetector
from ..pattern_matching.sql_injection import SQLIDetector
from ..pattern_matching.csrf import CSRFDetector
from ..pattern_matching.matcher import PatternMatcher

class TestPatternDetection(unittest.TestCase):
    
    def setUp(self):
        self.xss_detector = XSSDetector()
        self.sqli_detector = SQLIDetector()
        self.csrf_detector = CSRFDetector()
        self.pattern_matcher = PatternMatcher()
    
    def test_xss_detection_positive_cases(self):
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg/onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
            "'\"><script>alert('XSS')</script>"
        ]
        
        for payload in xss_payloads:
            detected, matches = self.xss_detector.detect(payload)
            self.assertTrue(detected, f"XSS payload not detected: {payload}")
            self.assertGreater(len(matches), 0)
    
    def test_xss_detection_negative_cases(self):
        benign_inputs = [
            "normal search query",
            "product description with <b>bold</b> text",
            "email@example.com",
            "price: $19.99",
            "user profile information"
        ]
        
        for input_text in benign_inputs:
            detected, matches = self.xss_detector.detect(input_text)
            self.assertFalse(detected, f"False positive XSS detection: {input_text}")
    
    def test_sqli_detection_positive_cases(self):
        sqli_payloads = [
            "' OR '1'='1",
            "1 OR 1=1 --",
            "' UNION SELECT null, null--",
            "'; DROP TABLE users; --",
            "admin'--",
            "' OR 1=1#",
            "1' AND '1'='1"
        ]
        
        for payload in sqli_payloads:
            detected, matches = self.sqli_detector.detect(payload)
            self.assertTrue(detected, f"SQLi payload not detected: {payload}")
            self.assertGreater(len(matches), 0)
    
    def test_sqli_detection_negative_cases(self):
        benign_inputs = [
            "normal search query",
            "product name: Widget 2000",
            "user comment about products",
            "price comparison data",
            "category: electronics"
        ]
        
        for input_text in benign_inputs:
            detected, matches = self.sqli_detector.detect(input_text)
            self.assertFalse(detected, f"False positive SQLi detection: {input_text}")
    
    def test_csrf_detection_positive_cases(self):
        csrf_scenarios = [
            {
                "method": "POST",
                "path": "/api/transfer",
                "headers": {"Referer": "https://attacker.com"},
                "cookies": {}  # Missing CSRF token
            },
            {
                "method": "POST",
                "path": "/api/delete",
                "headers": {"Origin": "https://malicious-site.com"},
                "cookies": {}
            },
            {
                "method": "POST",
                "path": "/api/update-profile",
                "headers": {},  # Missing referer
                "cookies": {}
            }
        ]
        
        for scenario in csrf_scenarios:
            detected, indicators = self.csrf_detector.detect(scenario)
            self.assertTrue(detected, f"CSRF scenario not detected: {scenario}")
            self.assertGreater(len(indicators), 0)
    
    def test_csrf_detection_negative_cases(self):
        legitimate_requests = [
            {
                "method": "GET",
                "path": "/api/products",
                "headers": {"Referer": "https://legitimate-site.com"},
                "cookies": {"csrftoken": "valid_token"}
            },
            {
                "method": "POST",
                "path": "/api/login",
                "headers": {"Referer": "https://same-origin.com"},
                "cookies": {"csrftoken": "valid_token"}
            }
        ]
        
        for request in legitimate_requests:
            detected, indicators = self.csrf_detector.detect(request)
            self.assertFalse(detected, f"False positive CSRF detection: {request}")
    
    def test_pattern_matcher_integration(self):
        test_cases = [
            {
                "text": "<script>alert('XSS')</script>",
                "request_data": {"method": "GET", "path": "/search"},
                "expected_type": "xss"
            },
            {
                "text": "' OR 1=1 --",
                "request_data": {"method": "POST", "path": "/login"},
                "expected_type": "sqli"
            },
            {
                "text": "normal query",
                "request_data": {
                    "method": "POST",
                    "path": "/transfer",
                    "headers": {"Referer": "https://attacker.com"},
                    "cookies": {}
                },
                "expected_type": "csrf"
            },
            {
                "text": "benign search query",
                "request_data": {
                    "method": "GET",
                    "path": "/products",
                    "cookies": {"csrftoken": "valid"}
                },
                "expected_type": "benign"
            }
        ]
        
        for case in test_cases:
            detected_type, matches = self.pattern_matcher.analyze(
                case["text"], case["request_data"]
            )
            self.assertEqual(
                detected_type, case["expected_type"],
                f"Pattern matching failed for case: {case}"
            )
    
    def test_pattern_priority(self):
        # Test that SQL injection takes priority over XSS when both are present
        combined_payload = "' OR 1=1 -- <script>alert('xss')</script>"
        request_data = {"method": "POST", "path": "/login", "body": combined_payload}
        
        detected_type, matches = self.pattern_matcher.analyze(combined_payload, request_data)
        
        # SQLi should be detected first due to pattern ordering
        self.assertEqual(detected_type, "sqli")
        self.assertGreater(len(matches), 0)
    
    def test_edge_cases(self):
        edge_cases = [
            ("", {"method": "GET"}),
            ("   ", {"method": "GET"}),
            ("a" * 10000, {"method": "GET"}),
            ("special chars: !@#$%^&*()", {"method": "GET"}),
            ("unicode: 测试 ñoël", {"method": "GET"})
        ]
        
        for text, request_data in edge_cases:
            try:
                detected_type, matches = self.pattern_matcher.analyze(text, request_data)
                self.assertIn(detected_type, ["benign", "xss", "sqli", "csrf"])
                self.assertIsInstance(matches, list)
            except Exception as e:
                self.fail(f"Pattern matcher failed on edge case '{text[:50]}...': {e}")

if __name__ == '__main__':
    unittest.main() 