import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class FeedbackSystem:
    def __init__(self):
        self.feedback_data = []
        self.model_performance = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
        self.user_feedback = {}
        
    def record_feedback(self, request_id: str, user_feedback: str, 
                       actual_threat: bool, predicted_threat: bool,
                       confidence: float, metadata: Optional[Dict] = None):
        """Record user feedback for model improvement"""
        try:
            feedback_entry = {
                'request_id': request_id,
                'timestamp': datetime.now(),
                'user_feedback': user_feedback,  # 'correct', 'incorrect', 'uncertain'
                'actual_threat': actual_threat,
                'predicted_threat': predicted_threat,
                'confidence': confidence,
                'metadata': metadata or {}
            }
            
            self.feedback_data.append(feedback_entry)
            
            # Update performance metrics
            if actual_threat and predicted_threat:
                self.model_performance['true_positives'] += 1
            elif not actual_threat and not predicted_threat:
                self.model_performance['true_negatives'] += 1
            elif not actual_threat and predicted_threat:
                self.model_performance['false_positives'] += 1
            elif actual_threat and not predicted_threat:
                self.model_performance['false_negatives'] += 1
                
            logger.info(f"Feedback recorded for request {request_id}: {user_feedback}")
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            
    def get_model_accuracy(self) -> Dict[str, float]:
        """Calculate current model accuracy metrics"""
        try:
            tp = self.model_performance['true_positives']
            tn = self.model_performance['true_negatives']
            fp = self.model_performance['false_positives']
            fn = self.model_performance['false_negatives']
            
            total = tp + tn + fp + fn
            
            if total == 0:
                return {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'total_samples': 0
                }
            
            accuracy = (tp + tn) / total
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'total_samples': total,
                'confusion_matrix': {
                    'true_positives': tp,
                    'true_negatives': tn,
                    'false_positives': fp,
                    'false_negatives': fn
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating model accuracy: {e}")
            return {'error': str(e)}
            
    def get_feedback_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of feedback received in last N days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_feedback = [
                f for f in self.feedback_data 
                if f['timestamp'] >= cutoff_date
            ]
            
            if not recent_feedback:
                return {'period_days': days, 'total_feedback': 0}
            
            feedback_counts = {}
            confidence_scores = []
            
            for feedback in recent_feedback:
                user_feedback = feedback['user_feedback']
                feedback_counts[user_feedback] = feedback_counts.get(user_feedback, 0) + 1
                confidence_scores.append(feedback['confidence'])
                
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            return {
                'period_days': days,
                'total_feedback': len(recent_feedback),
                'feedback_breakdown': feedback_counts,
                'average_confidence': avg_confidence,
                'accuracy_metrics': self.get_model_accuracy()
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback summary: {e}")
            return {'error': str(e)}
            
    def identify_improvement_areas(self) -> List[Dict[str, Any]]:
        """Identify areas where the model needs improvement"""
        try:
            improvement_areas = []
            
            # Analyze false positives
            fp_count = self.model_performance['false_positives']
            if fp_count > 5:  # Threshold for concern
                improvement_areas.append({
                    'area': 'false_positives',
                    'severity': 'high' if fp_count > 10 else 'medium',
                    'description': f'High false positive rate: {fp_count} instances',
                    'recommendation': 'Review and refine detection patterns, adjust confidence thresholds'
                })
                
            # Analyze false negatives
            fn_count = self.model_performance['false_negatives']
            if fn_count > 3:  # Lower threshold as these are more critical
                improvement_areas.append({
                    'area': 'false_negatives',
                    'severity': 'critical' if fn_count > 5 else 'high',
                    'description': f'Missing threats: {fn_count} instances',
                    'recommendation': 'Expand threat detection patterns, improve model sensitivity'
                })
                
            # Analyze confidence scores
            recent_feedback = self.feedback_data[-50:]  # Last 50 entries
            if recent_feedback:
                low_confidence = [f for f in recent_feedback if f['confidence'] < 0.6]
                if len(low_confidence) / len(recent_feedback) > 0.3:  # More than 30% low confidence
                    improvement_areas.append({
                        'area': 'confidence_calibration',
                        'severity': 'medium',
                        'description': 'Many predictions have low confidence scores',
                        'recommendation': 'Retrain model with more diverse dataset, improve feature engineering'
                    })
                    
            return improvement_areas
            
        except Exception as e:
            logger.error(f"Error identifying improvement areas: {e}")
            return [{'error': str(e)}]
            
    def get_training_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for model retraining"""
        try:
            accuracy_metrics = self.get_model_accuracy()
            total_samples = accuracy_metrics.get('total_samples', 0)
            accuracy = accuracy_metrics.get('accuracy', 0.0)
            
            recommendations = {
                'should_retrain': False,
                'reasons': [],
                'priority': 'low',
                'recommended_actions': []
            }
            
            # Check if retraining is needed
            if total_samples >= 100:  # Enough data for meaningful analysis
                if accuracy < 0.85:
                    recommendations['should_retrain'] = True
                    recommendations['reasons'].append(f'Low accuracy: {accuracy:.2%}')
                    recommendations['priority'] = 'high'
                    
                if self.model_performance['false_negatives'] > 5:
                    recommendations['should_retrain'] = True
                    recommendations['reasons'].append('High false negative rate')
                    recommendations['priority'] = 'critical'
                    
                if self.model_performance['false_positives'] > 20:
                    recommendations['should_retrain'] = True
                    recommendations['reasons'].append('High false positive rate')
                    if recommendations['priority'] == 'low':
                        recommendations['priority'] = 'medium'
                        
            # Recommend specific actions
            if recommendations['should_retrain']:
                recommendations['recommended_actions'] = [
                    'Collect additional training data',
                    'Review and update feature extraction',
                    'Experiment with different algorithms',
                    'Perform hyperparameter tuning',
                    'Validate on independent test set'
                ]
            else:
                recommendations['recommended_actions'] = [
                    'Continue monitoring performance',
                    'Collect more feedback data',
                    'Regular performance reviews'
                ]
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting training recommendations: {e}")
            return {'error': str(e)}
            
    def export_feedback_data(self, format: str = 'json') -> Any:
        """Export feedback data for analysis"""
        try:
            if format.lower() == 'json':
                # Convert datetime objects to strings for JSON serialization
                export_data = []
                for entry in self.feedback_data:
                    export_entry = entry.copy()
                    export_entry['timestamp'] = entry['timestamp'].isoformat()
                    export_data.append(export_entry)
                return json.dumps(export_data, indent=2)
            else:
                return self.feedback_data
                
        except Exception as e:
            logger.error(f"Error exporting feedback data: {e}")
            return {'error': str(e)}
            
    def reset_metrics(self):
        """Reset all metrics and feedback data"""
        try:
            self.feedback_data.clear()
            self.model_performance = {
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0
            }
            self.user_feedback.clear()
            logger.info("Feedback system metrics reset")
            
        except Exception as e:
            logger.error(f"Error resetting metrics: {e}")

# Global feedback system instance
feedback_system = FeedbackSystem() 