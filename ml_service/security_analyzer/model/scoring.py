from ..config import DANGER_THRESHOLDS, ATTACK_SEVERITY
from typing import Dict, Any

class DangerScorer:
    def calculate_score(self, attack_type: str, ml_confidence: float, 
                       pattern_matches: list, request_context: Dict[str, Any]) -> float:
        base_score = ATTACK_SEVERITY.get(attack_type, 0.0)
        
        if attack_type == 'benign':
            return 0.0
        
        confidence_weight = ml_confidence
        pattern_weight = min(len(pattern_matches) * 0.2, 0.6)
        
        context_weight = 0.0
        if request_context.get('method') == 'POST':
            context_weight += 0.1
        if 'admin' in request_context.get('path', ''):
            context_weight += 0.1
        
        final_score = base_score * confidence_weight + pattern_weight + context_weight
        return min(final_score, 1.0)
    
    def get_danger_level(self, score: float) -> str:
        if score >= DANGER_THRESHOLDS['high']:
            return 'high'
        elif score >= DANGER_THRESHOLDS['medium']:
            return 'medium'
        elif score >= DANGER_THRESHOLDS['low']:
            return 'low'
        return 'minimal' 