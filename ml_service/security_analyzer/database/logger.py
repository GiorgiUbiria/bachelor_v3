from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Dict, Any, Optional
import json

from database import DatabaseConnection
from .models import HttpRequestLog, MLAnalysisLog, SecurityMetrics
from ..utils.logger import setup_logger
from ..config import ENABLE_DB_LOGGING

class SecurityDatabaseLogger:
    def __init__(self):
        self.logger = setup_logger('SecurityDatabaseLogger')
        self.db_available = False
        try:
            if ENABLE_DB_LOGGING:  # Only try to connect if explicitly enabled
                self.db_connection = DatabaseConnection()
                
                if self.db_connection.SessionLocal:
                    self.SessionLocal = self.db_connection.SessionLocal
                    self.db_available = True
                else:
                    self.logger.warning("Database not available, logging disabled")
            else:
                self.logger.info("Database logging disabled by configuration")
        except Exception as e:
            self.logger.warning(f"Database not available, logging disabled: {e}")
            self.db_available = False
    
    def log_request_analysis(self, request_data: Dict[str, Any], 
                           analysis_result: Dict[str, Any],
                           processing_time: float) -> Optional[str]:
        if not self.db_available:
            # Optionally log to file instead
            self.logger.debug(f"Analysis result (DB unavailable): {analysis_result}")
            return
        
        try:
            session = self.SessionLocal()
            
            request_log = HttpRequestLog(
                ip_address=request_data.get('ip_address'),
                user_agent=request_data.get('user_agent'),
                path=request_data.get('path'),
                method=request_data.get('method'),
                query_params=request_data.get('query_params'),
                headers=request_data.get('headers', {}),
                body=request_data.get('body', {}),
                cookies=request_data.get('cookies', {}),
                duration_ms=int(processing_time * 1000),
                suspected_attack_type=analysis_result.get('attack_type'),
                attack_score=analysis_result.get('attack_score'),
                confidence_score=analysis_result.get('confidence'),
                is_malicious=analysis_result.get('is_malicious'),
                pattern_matches=analysis_result.get('details', {}).get('matched_patterns', []),
                ml_prediction=analysis_result.get('details', {}).get('ml_prediction'),
                ensemble_weights=analysis_result.get('details', {}).get('ensemble_weights', {})
            )
            
            session.add(request_log)
            session.commit()
            
            request_log_id = request_log.id
            session.close()
            
            self.logger.debug(f"Logged request analysis: {request_log_id}")
            return str(request_log_id)
            
        except Exception as e:
            self.logger.error(f"Failed to log request analysis: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return None
    
    def log_ml_analysis(self, request_log_id: str, analysis_details: Dict[str, Any],
                       processing_time: float) -> Optional[str]:
        if not self.db_available or not request_log_id:
            return None
        
        try:
            session = self.SessionLocal()
            
            ml_log = MLAnalysisLog(
                request_log_id=request_log_id,
                analysis_type="security_analysis",
                input_features=analysis_details.get('input_features', {}),
                ml_probabilities=analysis_details.get('ml_probabilities', {}),
                pattern_results=analysis_details.get('pattern_results', {}),
                ensemble_decision=analysis_details.get('ensemble_decision', {}),
                feature_importance=analysis_details.get('feature_importance', {}),
                explanation_data=analysis_details.get('explanation_data', {}),
                processing_time_ms=int(processing_time * 1000)
            )
            
            session.add(ml_log)
            session.commit()
            
            ml_log_id = ml_log.id
            session.close()
            
            return str(ml_log_id)
            
        except Exception as e:
            self.logger.error(f"Failed to log ML analysis: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return None
    
    def update_security_metrics(self):
        if not self.db_available:
            return
        
        try:
            session = self.SessionLocal()
            today = datetime.utcnow().date()
            
            from sqlalchemy import func, and_
            
            total_requests = session.query(func.count(HttpRequestLog.id)).filter(
                func.date(HttpRequestLog.timestamp) == today
            ).scalar() or 0
            
            malicious_requests = session.query(func.count(HttpRequestLog.id)).filter(
                and_(
                    func.date(HttpRequestLog.timestamp) == today,
                    HttpRequestLog.is_malicious == True
                )
            ).scalar() or 0
            
            attack_types = session.query(
                HttpRequestLog.suspected_attack_type,
                func.count(HttpRequestLog.id)
            ).filter(
                and_(
                    func.date(HttpRequestLog.timestamp) == today,
                    HttpRequestLog.is_malicious == True
                )
            ).group_by(HttpRequestLog.suspected_attack_type).all()
            
            attack_type_counts = {attack_type: count for attack_type, count in attack_types}
            
            avg_confidence = session.query(func.avg(HttpRequestLog.confidence_score)).filter(
                func.date(HttpRequestLog.timestamp) == today
            ).scalar() or 0.0
            
            avg_processing_time = session.query(func.avg(HttpRequestLog.duration_ms)).filter(
                func.date(HttpRequestLog.timestamp) == today
            ).scalar() or 0.0
            
            existing_metrics = session.query(SecurityMetrics).filter(
                func.date(SecurityMetrics.date) == today
            ).first()
            
            if existing_metrics:
                existing_metrics.total_requests = total_requests
                existing_metrics.malicious_requests = malicious_requests
                existing_metrics.attack_type_counts = attack_type_counts
                existing_metrics.average_confidence = float(avg_confidence)
                existing_metrics.average_processing_time = float(avg_processing_time)
            else:
                metrics = SecurityMetrics(
                    total_requests=total_requests,
                    malicious_requests=malicious_requests,
                    attack_type_counts=attack_type_counts,
                    average_confidence=float(avg_confidence),
                    average_processing_time=float(avg_processing_time)
                )
                session.add(metrics)
            
            session.commit()
            session.close()
            
            self.logger.info(f"Updated security metrics for {today}")
            
        except Exception as e:
            self.logger.error(f"Failed to update security metrics: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
    
    def get_security_dashboard_data(self, days: int = 7) -> Dict[str, Any]:
        if not self.db_available:
            return {"error": "Database not available"}
        
        try:
            session = self.SessionLocal()
            from sqlalchemy import func, and_
            from datetime import timedelta
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            metrics = session.query(SecurityMetrics).filter(
                SecurityMetrics.date >= start_date
            ).order_by(SecurityMetrics.date).all()
            
            dashboard_data = {
                'period_days': days,
                'total_requests': sum(m.total_requests for m in metrics),
                'total_threats': sum(m.malicious_requests for m in metrics),
                'threat_rate': 0.0,
                'daily_stats': [],
                'attack_distribution': {},
                'average_confidence': 0.0,
                'average_response_time': 0.0
            }
            
            if dashboard_data['total_requests'] > 0:
                dashboard_data['threat_rate'] = dashboard_data['total_threats'] / dashboard_data['total_requests']
            
            for metric in metrics:
                dashboard_data['daily_stats'].append({
                    'date': metric.date.isoformat(),
                    'requests': metric.total_requests,
                    'threats': metric.malicious_requests,
                    'threat_rate': metric.malicious_requests / max(1, metric.total_requests)
                })
            
            all_attack_counts = {}
            confidence_sum = 0
            response_time_sum = 0
            
            for metric in metrics:
                if metric.attack_type_counts:
                    for attack_type, count in metric.attack_type_counts.items():
                        all_attack_counts[attack_type] = all_attack_counts.get(attack_type, 0) + count
                
                confidence_sum += metric.average_confidence or 0
                response_time_sum += metric.average_processing_time or 0
            
            dashboard_data['attack_distribution'] = all_attack_counts
            dashboard_data['average_confidence'] = confidence_sum / max(1, len(metrics))
            dashboard_data['average_response_time'] = response_time_sum / max(1, len(metrics))
            
            session.close()
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard data: {e}")
            if 'session' in locals():
                session.close()
            return {"error": str(e)}

    def log_analysis(self, request_data, response_data):
        """Log security analysis results"""
        if not self.db_available:
            return
        
        try:
            # Extract relevant information from request and response
            analysis_type = "security_analysis"
            input_data = json.dumps(request_data) if isinstance(request_data, dict) else str(request_data)
            output_data = json.dumps(response_data) if isinstance(response_data, dict) else str(response_data)
            
            # Get processing time and confidence from response
            processing_time = response_data.get('processing_time_ms', 0) if isinstance(response_data, dict) else 0
            confidence = response_data.get('confidence', 0.0) if isinstance(response_data, dict) else 0.0
            
            # Create log entry
            log_entry = MLAnalysisLog(
                analysis_type=analysis_type,
                input_data=input_data,
                output_data=output_data,
                processing_time_ms=int(processing_time),
                confidence_score=float(confidence)
            )
            
            # Save to database
            with self.SessionLocal() as session:
                session.add(log_entry)
                session.commit()
                self.logger.info(f"Analysis logged: {analysis_type}")
        
        except Exception as e:
            self.logger.error(f"Failed to log analysis: {e}")

    def log_request(self, request_data, analysis_result=None):
        """Log HTTP request with optional analysis result"""
        if not self.db_available:
            return
        
        try:
            # Extract request information
            method = request_data.get('method', 'GET')
            path = request_data.get('path', '/')
            ip_address = request_data.get('ip_address', '127.0.0.1')
            user_agent = request_data.get('user_agent', '')
            
            # Extract analysis results if provided
            suspected_attack_type = None
            attack_score = 0.0
            
            if analysis_result:
                if hasattr(analysis_result, 'attack_type'):
                    suspected_attack_type = analysis_result.attack_type if analysis_result.is_malicious else None
                    attack_score = analysis_result.attack_score
                elif isinstance(analysis_result, dict):
                    suspected_attack_type = analysis_result.get('attack_type') if analysis_result.get('is_malicious') else None
                    attack_score = analysis_result.get('attack_score', 0.0)
            
            # Create log entry
            log_entry = HttpRequestLog(
                ip_address=ip_address,
                user_agent=user_agent,
                path=path,
                method=method,
                suspected_attack_type=suspected_attack_type,
                attack_score=attack_score
            )
            
            # Save to database
            with self.SessionLocal() as session:
                session.add(log_entry)
                session.commit()
                self.logger.info(f"Request logged: {method} {path}")
        
        except Exception as e:
            self.logger.error(f"Failed to log request: {e}") 