from ..utils.logger import setup_logger

class BenignDetector:
    def __init__(self):
        self.logger = setup_logger('BenignDetector')
        
        # Common benign patterns
        self.benign_paths = [
            '/api/products', '/search', '/categories', '/help',
            '/contact', '/about', '/profile', '/dashboard',
            '/static/', '/assets/', '/images/', '/css/', '/js/'
        ]
        
        self.benign_query_patterns = [
            r'^[a-zA-Z0-9\s\-_.]+$',  # Simple alphanumeric queries
            r'^category=\w+$',         # Category filters
            r'^page=\d+$',            # Pagination
            r'^sort=\w+$',            # Sorting
            r'^limit=\d+$',           # Limits
        ]
        
        self.benign_user_agents = [
            'Mozilla/5.0', 'Chrome/', 'Safari/', 'Firefox/',
            'Edge/', 'Opera/', 'Googlebot', 'Bingbot'
        ]
    
    def is_benign(self, request_data):
        """Check if request shows clear benign indicators with adjusted threshold"""
        try:
            method = request_data.get('method', 'GET')
            path = request_data.get('path', '/')
            query_params = str(request_data.get('query_params', ''))
            user_agent = request_data.get('user_agent', '')
            headers = request_data.get('headers', {})
            
            benign_score = 0
            
            # Check for benign paths
            if any(benign_path in path for benign_path in self.benign_paths):
                benign_score += 2
            
            # GET requests are generally safer
            if method == 'GET':
                benign_score += 1
            
            # Check for legitimate user agents
            if any(ua in user_agent for ua in self.benign_user_agents):
                benign_score += 1
            
            # Check for simple, benign query patterns
            import re
            if query_params:
                for pattern in self.benign_query_patterns:
                    if re.match(pattern, query_params.strip()):
                        benign_score += 1
                        break
            
            # Check for standard headers
            if headers.get('Accept') and 'text/html' in headers.get('Accept', ''):
                benign_score += 1
            
            # Increase threshold to reduce false negatives
            return benign_score >= 4  # Increased from 3
            
        except Exception as e:
            self.logger.error(f"Benign detection failed: {e}")
            return False 