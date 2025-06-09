from typing import Dict, List, Any, Optional
from sqlalchemy.orm import sessionmaker
from database import DatabaseConnection
from ..database.models import AttackMitigation
from ..utils.logger import setup_logger

class MitigationAdvisor:
    def __init__(self):
        self.logger = setup_logger('MitigationAdvisor')
        self.db_connection = DatabaseConnection()
        
        if self.db_connection.SessionLocal:
            self.SessionLocal = self.db_connection.SessionLocal
            self.db_available = True
        else:
            self.db_available = False
    
    def get_mitigation_advice(self, attack_type: str, attack_patterns: List[str], 
                            request_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific mitigation advice for detected attacks"""
        
        if not self.db_available:
            return self._get_fallback_advice(attack_type)
        
        try:
            session = self.SessionLocal()
            
            # Get mitigation strategies from database
            mitigations = session.query(AttackMitigation).filter(
                AttackMitigation.attack_type == attack_type
            ).all()
            
            session.close()
            
            advice = {
                'attack_type': attack_type,
                'severity': self._calculate_severity(attack_type, attack_patterns),
                'immediate_actions': [],
                'preventive_measures': [],
                'code_examples': [],
                'custom_recommendations': []
            }
            
            for mitigation in mitigations:
                # Check if this mitigation applies to detected patterns
                if self._pattern_matches_mitigation(attack_patterns, mitigation.attack_pattern):
                    advice['immediate_actions'].append({
                        'action': mitigation.mitigation_strategy,
                        'priority': mitigation.severity_level,
                        'description': mitigation.prevention_tips
                    })
                    
                    if mitigation.sanitization_code:
                        advice['code_examples'].append({
                            'language': 'python',
                            'code': mitigation.sanitization_code,
                            'description': mitigation.mitigation_strategy
                        })
            
            # Add context-specific recommendations
            advice['custom_recommendations'] = self._get_context_specific_advice(
                attack_type, request_context
            )
            
            return advice
            
        except Exception as e:
            self.logger.error(f"Failed to get mitigation advice: {e}")
            return self._get_fallback_advice(attack_type)
    
    def _get_fallback_advice(self, attack_type: str) -> Dict[str, Any]:
        """Fallback advice when database is unavailable"""
        fallback_advice = {
            'xss': {
                'immediate_actions': ['Sanitize user input', 'Implement Content Security Policy'],
                'code_examples': ['html.escape(user_input)', 'Content-Security-Policy: default-src \'self\'']
            },
            'sqli': {
                'immediate_actions': ['Use parameterized queries', 'Validate input'],
                'code_examples': ['cursor.execute("SELECT * WHERE id = %s", (id,))']
            },
            'csrf': {
                'immediate_actions': ['Implement CSRF tokens', 'Validate request origin'],
                'code_examples': ['@csrf_protect', 'check request.headers.get("Origin")']
            }
        }
        
        return fallback_advice.get(attack_type, {'immediate_actions': ['Review and validate input']})
    
    def _pattern_matches_mitigation(self, detected_patterns: List[str], mitigation_pattern: str) -> bool:
        """Check if detected patterns match mitigation strategy"""
        if not detected_patterns or not mitigation_pattern:
            return True  # Apply general mitigations
        
        return any(mitigation_pattern.lower() in pattern.lower() for pattern in detected_patterns)
    
    def _calculate_severity(self, attack_type: str, patterns: List[str]) -> str:
        """Calculate severity based on attack type and patterns"""
        base_severity = {
            'sqli': 'critical',
            'xss': 'high', 
            'csrf': 'medium',
            'benign': 'low'
        }
        
        severity = base_severity.get(attack_type, 'medium')
        
        # Increase severity based on pattern complexity
        if len(patterns) > 2:
            severity_levels = ['low', 'medium', 'high', 'critical']
            current_index = severity_levels.index(severity)
            if current_index < len(severity_levels) - 1:
                severity = severity_levels[current_index + 1]
        
        return severity
    
    def _get_context_specific_advice(self, attack_type: str, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate context-specific recommendations"""
        recommendations = []
        
        path = context.get('path', '')
        method = context.get('method', '')
        
        if '/admin' in path:
            recommendations.append({
                'type': 'access_control',
                'recommendation': 'Implement additional authentication for admin endpoints',
                'priority': 'high'
            })
        
        if method == 'POST' and attack_type == 'csrf':
            recommendations.append({
                'type': 'csrf_protection',
                'recommendation': 'Ensure CSRF tokens are required for all POST requests',
                'priority': 'high'
            })
        
        if '/api/' in path:
            recommendations.append({
                'type': 'api_security',
                'recommendation': 'Implement rate limiting and API key validation',
                'priority': 'medium'
            })
        
        return recommendations 