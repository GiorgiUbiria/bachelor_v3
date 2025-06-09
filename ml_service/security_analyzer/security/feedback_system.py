from typing import Dict, List, Any, Optional
from sqlalchemy.orm import sessionmaker
from database import DatabaseConnection
from ..database.models import SecurityFeedback, HttpRequestLog
from ..utils.logger import setup_logger
from datetime import datetime

class SecurityFeedbackSystem:
    def __init__(self):
        self.logger = setup_logger('SecurityFeedbackSystem')
        self.db_connection = DatabaseConnection()
        
        if self.db_connection.SessionLocal:
            self.SessionLocal = self.db_connection.SessionLocal
            self.db_available = True
        else:
            self.db_available = False
    
    def submit_feedback(self, request_log_id: str, feedback_type: str,
                       corrected_label: Optional[str] = None,
                       feedback_reason: Optional[str] = None,
                       user_id: Optional[str] = None) -> bool:
        """Submit feedback for a security analysis result"""
        
        if not self.db_available:
            self.logger.warning("Database unavailable, feedback not stored")
            return False
        
        try:
            session = self.SessionLocal()
            
            # Get original prediction
            request_log = session.query(HttpRequestLog).filter(
                HttpRequestLog.id == request_log_id
            ).first()
            
            if not request_log:
                self.logger.warning(f"Request log {request_log_id} not found")
                session.close()
                return False
            
            feedback = SecurityFeedback(
                request_log_id=request_log_id,
                feedback_type=feedback_type,
                original_prediction=request_log.suspected_attack_type,
                corrected_label=corrected_label,
                feedback_reason=feedback_reason,
                feedback_source='user' if user_id else 'automated',
                created_by=user_id
            )
            
            session.add(feedback)
            session.commit()
            session.close()
            
            self.logger.info(f"Feedback submitted for request {request_log_id}: {feedback_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit feedback: {e}")
            return False
    
    def get_feedback_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback statistics for model improvement"""
        
        if not self.db_available:
            return {"error": "Database unavailable"}
        
        try:
            session = self.SessionLocal()
            from sqlalchemy import func
            from datetime import timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Get feedback counts by type
            feedback_counts = session.query(
                SecurityFeedback.feedback_type,
                func.count(SecurityFeedback.id)
            ).filter(
                SecurityFeedback.created_at >= cutoff_date
            ).group_by(SecurityFeedback.feedback_type).all()
            
            # Get most common false positives
            false_positives = session.query(
                SecurityFeedback.original_prediction,
                SecurityFeedback.corrected_label,
                func.count(SecurityFeedback.id)
            ).filter(
                SecurityFeedback.feedback_type == 'false_positive',
                SecurityFeedback.created_at >= cutoff_date
            ).group_by(
                SecurityFeedback.original_prediction,
                SecurityFeedback.corrected_label
            ).order_by(func.count(SecurityFeedback.id).desc()).limit(10).all()
            
            session.close()
            
            return {
                'period_days': days,
                'feedback_counts': dict(feedback_counts),
                'total_feedback': sum(count for _, count in feedback_counts),
                'common_false_positives': [
                    {
                        'predicted': pred,
                        'actual': actual,
                        'count': count
                    } for pred, actual, count in false_positives
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get feedback statistics: {e}")
            return {"error": str(e)}
    
    def get_correction_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for model corrections based on feedback"""
        
        if not self.db_available:
            return []
        
        try:
            session = self.SessionLocal()
            from sqlalchemy import func
            
            # Find patterns in false positives/negatives
            correction_patterns = session.query(
                SecurityFeedback.original_prediction,
                SecurityFeedback.corrected_label,
                SecurityFeedback.feedback_reason,
                func.count(SecurityFeedback.id).label('count')
            ).filter(
                SecurityFeedback.feedback_type.in_(['false_positive', 'false_negative'])
            ).group_by(
                SecurityFeedback.original_prediction,
                SecurityFeedback.corrected_label,
                SecurityFeedback.feedback_reason
            ).having(func.count(SecurityFeedback.id) >= 3).all()
            
            suggestions = []
            for pattern in correction_patterns:
                suggestions.append({
                    'original_prediction': pattern.original_prediction,
                    'suggested_correction': pattern.corrected_label,
                    'reason': pattern.feedback_reason,
                    'frequency': pattern.count,
                    'confidence': min(pattern.count / 10.0, 1.0)  # Scale confidence
                })
            
            session.close()
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Failed to get correction suggestions: {e}")
            return [] 