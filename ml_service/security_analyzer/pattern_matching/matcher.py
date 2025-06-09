from .xss import XSSDetector
from .sql_injection import SQLIDetector
from .csrf import CSRFDetector
from typing import Dict, Any, Tuple, List
from ..utils.logger import setup_logger

class PatternMatcher:
    def __init__(self):
        """Initialize the pattern matcher with all detectors"""
        self.logger = setup_logger('PatternMatcher')
        
        # Initialize detectors
        self.xss_detector = XSSDetector()
        self.sqli_detector = SQLIDetector()
        self.csrf_detector = CSRFDetector()
    
    def analyze(self, text: str, request_data: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Analyze text and request data for security patterns"""
        try:
            # Check patterns in priority order: SQLi -> XSS -> CSRF
            # SQLi should take highest priority due to severity
            
            # Check SQL Injection first (highest priority)
            sqli_detected, sqli_matches = self.sqli_detector.detect(text)
            if sqli_detected:
                self.logger.info(f"SQLi pattern detected: {len(sqli_matches)} matches")
                return "sqli", sqli_matches
            
            # Check XSS second
            xss_detected, xss_matches = self.xss_detector.detect(text)
            if xss_detected:
                self.logger.info(f"XSS pattern detected: {len(xss_matches)} matches")
                return "xss", xss_matches
            
            # Check CSRF last
            csrf_detected, csrf_indicators = self.csrf_detector.detect(request_data)
            if csrf_detected:
                self.logger.info(f"CSRF pattern detected: {len(csrf_indicators)} indicators")
                return "csrf", csrf_indicators
            
            return "benign", []
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            return "benign", []

    def get_pattern_confidence(self, attack_type, matches):
        """Calculate confidence based on pattern matches"""
        if attack_type == 'benign':
            return 0.7
        
        # Base confidence by attack type
        base_confidence = {
            'xss': 0.8,
            'sqli': 0.9,
            'csrf': 0.7
        }.get(attack_type, 0.6)
        
        # Increase confidence based on number of matches
        confidence_boost = min(0.15, len(matches) * 0.05)
        
        return min(0.95, base_confidence + confidence_boost) 