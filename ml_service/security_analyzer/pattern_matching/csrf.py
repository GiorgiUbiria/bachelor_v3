from typing import Dict, List, Tuple, Any

class CSRFDetector:
    def detect(self, request_data):
        """Detect potential CSRF attacks"""
        method = request_data.get('method', 'GET')
        headers = request_data.get('headers', {})
        cookies = request_data.get('cookies', {}) or {}  # Handle None case
        path = request_data.get('path', '/')
        
        indicators = []
        
        # Check for missing CSRF token in POST requests
        if method == 'POST':
            if 'csrftoken' not in cookies and 'csrf_token' not in cookies:
                indicators.append("Missing CSRF token in POST request")
        
        # Check for suspicious referer
        referer = headers.get('Referer', headers.get('referer', ''))
        if referer:
            try:
                from urllib.parse import urlparse
                referer_domain = urlparse(referer).netloc
                # Add basic domain validation - should be more sophisticated in production
                suspicious_domains = ['attacker.com', 'malicious-site.com', 'evil.com']
                if any(domain in referer_domain for domain in suspicious_domains):
                    indicators.append(f"Suspicious referer domain: {referer_domain}")
            except:
                indicators.append("Invalid referer format")
        
        # Check for suspicious origin
        origin = headers.get('Origin', headers.get('origin', ''))
        if origin:
            try:
                from urllib.parse import urlparse
                origin_domain = urlparse(origin).netloc
                suspicious_domains = ['attacker.com', 'malicious-site.com', 'evil.com']
                if any(domain in origin_domain for domain in suspicious_domains):
                    indicators.append(f"Suspicious origin domain: {origin_domain}")
            except:
                indicators.append("Invalid origin format")
        
        # For legitimate same-origin requests with CSRF tokens, return False
        if method == 'POST' and ('csrftoken' in cookies or 'csrf_token' in cookies):
            if referer and 'same-origin.com' in referer:
                return False, []
        
        return len(indicators) > 0, indicators 