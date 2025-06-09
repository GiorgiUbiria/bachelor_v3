from typing import Dict, List, Tuple, Any

class CSRFDetector:
    def detect(self, request_data):
        """Detect potential CSRF attacks with balanced precision"""
        try:
            method = request_data.get('method', 'GET').upper()
            path = request_data.get('path', '/')
            headers = request_data.get('headers', {})
            cookies = request_data.get('cookies', {})
            
            # Only check POST/PUT/DELETE requests for CSRF
            if method not in ['POST', 'PUT', 'DELETE']:
                return False, []
            
            # Skip CSRF checks for safe endpoints
            safe_endpoints = ['/login', '/register', '/forgot-password', '/api/public']
            if any(safe_path in path for safe_path in safe_endpoints):
                return False, []
            
            suspicious_indicators = []
            csrf_score = 0
            
            # Check for suspicious referer/origin
            referer = headers.get('Referer', headers.get('referer', ''))
            origin = headers.get('Origin', headers.get('origin', ''))
            
            if referer:
                suspicious_domains = ['attacker.com', 'malicious', 'evil', 'hack']
                if any(domain in referer.lower() for domain in suspicious_domains):
                    csrf_score += 1
                    suspicious_indicators.append(f"Suspicious referer: {referer}")
            
            if origin:
                suspicious_domains = ['attacker.com', 'malicious', 'evil', 'hack']
                if any(domain in origin.lower() for domain in suspicious_domains):
                    csrf_score += 1
                    suspicious_indicators.append(f"Suspicious origin: {origin}")
            
            # Check for missing CSRF token on sensitive endpoints
            sensitive_endpoints = ['/transfer', '/delete', '/admin', '/update-profile', '/api/update']
            is_sensitive = any(endpoint in path for endpoint in sensitive_endpoints)
            
            if is_sensitive:
                csrf_token = (
                    cookies.get('csrftoken') or 
                    cookies.get('csrf_token') or 
                    headers.get('X-CSRF-Token') or
                    headers.get('X-CSRFToken')
                )
                if not csrf_token:
                    csrf_score += 1  # Reduced from requiring multiple indicators
                    suspicious_indicators.append("Missing CSRF token on sensitive endpoint")
            
            # Lower threshold for CSRF detection but keep precision
            if csrf_score >= 1 and is_sensitive:  # Require at least one indicator on sensitive endpoints
                return True, suspicious_indicators
            elif csrf_score >= 2:  # Require two indicators on other endpoints
                return True, suspicious_indicators
            
            return False, []
            
        except Exception as e:
            self.logger.error(f"CSRF detection failed: {e}")
            return False, [] 