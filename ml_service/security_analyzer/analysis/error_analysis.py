import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
import re
from datetime import datetime, timedelta

from ..database.logger import SecurityDatabaseLogger
from ..utils.logger import setup_logger
from ..model.classifier import SecurityAnalyzer

class AdvancedErrorAnalyzer:
    def __init__(self):
        self.logger = setup_logger('AdvancedErrorAnalyzer')
        self.db_logger = SecurityDatabaseLogger()
    
    def collect_high_confidence_errors(self, confidence_threshold: float = 0.8,
                                     days: int = 30) -> Dict[str, Any]:
        """Collect high-confidence prediction errors for analysis"""
        
        if not self.db_logger.db_available:
            return {"error": "Database not available"}
        
        try:
            session = self.db_logger.SessionLocal()
            from sqlalchemy import and_, func
            from ..database.models import HttpRequestLog, SecurityFeedback
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Get high-confidence predictions that received negative feedback
            high_conf_errors = session.query(
                HttpRequestLog.suspected_attack_type,
                HttpRequestLog.confidence_score,
                HttpRequestLog.path,
                HttpRequestLog.query_params,
                HttpRequestLog.pattern_matches,
                SecurityFeedback.corrected_label,
                SecurityFeedback.feedback_reason
            ).join(SecurityFeedback).filter(
                and_(
                    HttpRequestLog.confidence_score >= confidence_threshold,
                    HttpRequestLog.timestamp >= cutoff_date,
                    SecurityFeedback.feedback_type.in_(['false_positive', 'false_negative'])
                )
            ).all()
            
            session.close()
            
            # Analyze error patterns
            error_patterns = self._analyze_error_patterns(high_conf_errors)
            
            return {
                'collection_period_days': days,
                'confidence_threshold': confidence_threshold,
                'total_high_confidence_errors': len(high_conf_errors),
                'error_patterns': error_patterns,
                'recommendations': self._generate_error_recommendations(error_patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect high-confidence errors: {e}")
            return {"error": str(e)}
    
    def _analyze_error_patterns(self, errors: List[Tuple]) -> Dict[str, Any]:
        """Analyze patterns in prediction errors"""
        
        patterns = {
            'prediction_errors': defaultdict(int),
            'common_false_positive_patterns': [],
            'common_false_negative_patterns': [],
            'path_based_errors': defaultdict(int),
            'pattern_match_errors': defaultdict(int)
        }
        
        false_positives = []
        false_negatives = []
        
        for error in errors:
            predicted, confidence, path, query_params, pattern_matches, corrected, reason = error
            
            # Count prediction error types
            error_key = f"{predicted} -> {corrected}"
            patterns['prediction_errors'][error_key] += 1
            
            # Separate false positives and negatives
            if predicted != 'benign' and corrected == 'benign':
                false_positives.append({
                    'predicted': predicted,
                    'path': path,
                    'query_params': query_params,
                    'pattern_matches': pattern_matches,
                    'reason': reason
                })
            elif predicted == 'benign' and corrected != 'benign':
                false_negatives.append({
                    'actual': corrected,
                    'path': path,
                    'query_params': query_params,
                    'pattern_matches': pattern_matches,
                    'reason': reason
                })
            
            # Analyze path-based errors
            if path:
                path_pattern = self._extract_path_pattern(path)
                patterns['path_based_errors'][path_pattern] += 1
            
            # Analyze pattern matching errors
            if pattern_matches:
                for pattern in pattern_matches:
                    patterns['pattern_match_errors'][pattern] += 1
        
        # Find common patterns
        patterns['common_false_positive_patterns'] = self._find_common_patterns(false_positives)
        patterns['common_false_negative_patterns'] = self._find_common_patterns(false_negatives)
        
        return patterns
    
    def _extract_path_pattern(self, path: str) -> str:
        """Extract generalized pattern from request path"""
        # Replace IDs and numbers with placeholders
        pattern = re.sub(r'/\d+', '/{id}', path)
        pattern = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '{uuid}', pattern)
        return pattern
    
    def _find_common_patterns(self, errors: List[Dict]) -> List[Dict[str, Any]]:
        """Find common patterns in error cases"""
        
        if not errors:
            return []
        
        # Group by similar characteristics
        path_groups = defaultdict(list)
        query_groups = defaultdict(list)
        
        for error in errors:
            path_pattern = self._extract_path_pattern(error.get('path', ''))
            path_groups[path_pattern].append(error)
            
            # Extract query parameter patterns
            query_params = error.get('query_params', '')
            if query_params:
                query_pattern = self._extract_query_pattern(query_params)
                query_groups[query_pattern].append(error)
        
        common_patterns = []
        
        # Find frequent path patterns
        for path_pattern, group in path_groups.items():
            if len(group) >= 3:  # At least 3 occurrences
                common_patterns.append({
                    'type': 'path_pattern',
                    'pattern': path_pattern,
                    'frequency': len(group),
                    'examples': group[:3]
                })
        
        # Find frequent query patterns
        for query_pattern, group in query_groups.items():
            if len(group) >= 3:
                common_patterns.append({
                    'type': 'query_pattern',
                    'pattern': query_pattern,
                    'frequency': len(group),
                    'examples': group[:3]
                })
        
        return common_patterns
    
    def _extract_query_pattern(self, query_params: str) -> str:
        """Extract generalized pattern from query parameters"""
        # Replace values with placeholders
        pattern = re.sub(r'=([^&]+)', '={value}', query_params)
        return pattern
    
    def _generate_error_recommendations(self, error_patterns: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on error analysis"""
        
        recommendations = []
        
        # Check for frequent false positives
        fp_patterns = error_patterns.get('common_false_positive_patterns', [])
        if fp_patterns:
            recommendations.append(
                f"Found {len(fp_patterns)} common false positive patterns. "
                "Consider refining pattern matching rules or adjusting confidence thresholds."
            )
        
        # Check for frequent false negatives
        fn_patterns = error_patterns.get('common_false_negative_patterns', [])
        if fn_patterns:
            recommendations.append(
                f"Found {len(fn_patterns)} common false negative patterns. "
                "Consider adding new detection rules or improving feature extraction."
            )
        
        # Check for path-specific errors
        path_errors = error_patterns.get('path_based_errors', {})
        frequent_path_errors = {k: v for k, v in path_errors.items() if v >= 5}
        if frequent_path_errors:
            recommendations.append(
                f"Frequent errors on paths: {list(frequent_path_errors.keys())}. "
                "Consider path-specific validation rules."
            )
        
        # Check for pattern matching issues
        pattern_errors = error_patterns.get('pattern_match_errors', {})
        if pattern_errors:
            most_problematic = max(pattern_errors.items(), key=lambda x: x[1])
            recommendations.append(
                f"Pattern '{most_problematic[0]}' frequently causes errors. "
                "Consider reviewing this pattern's accuracy."
            )
        
        return recommendations
    
    def cluster_error_cases(self, errors: List[Dict], n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster error cases to identify distinct error types"""
        
        if len(errors) < n_clusters:
            return {"error": "Not enough error cases for clustering"}
        
        try:
            # Feature extraction for clustering
            features = []
            labels = []
            
            for error in errors:
                # Extract features: path length, query length, attack type, etc.
                path_len = len(error.get('path', ''))
                query_len = len(error.get('query_params', ''))
                predicted_type = error.get('predicted', 'unknown')
                
                # Encode attack types
                attack_encoding = {
                    'benign': 0, 'xss': 1, 'sqli': 2, 'csrf': 3, 'unknown': 4
                }
                
                feature_vector = [
                    path_len,
                    query_len,
                    attack_encoding.get(predicted_type, 4),
                    error.get('confidence', 0.5)
                ]
                
                features.append(feature_vector)
                labels.append(error)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            
            # Analyze clusters
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(labels[i])
            
            cluster_analysis = {}
            for cluster_id, cluster_errors in clusters.items():
                cluster_analysis[f"cluster_{cluster_id}"] = {
                    'size': len(cluster_errors),
                    'common_characteristics': self._analyze_cluster_characteristics(cluster_errors),
                    'examples': cluster_errors[:3]
                }
            
            return {
                'n_clusters': n_clusters,
                'total_errors': len(errors),
                'clusters': cluster_analysis,
                'cluster_centers': kmeans.cluster_centers_.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to cluster error cases: {e}")
            return {"error": str(e)}
    
    def _analyze_cluster_characteristics(self, cluster_errors: List[Dict]) -> Dict[str, Any]:
        """Analyze common characteristics within an error cluster"""
        
        # Count common attributes
        predicted_types = [e.get('predicted', 'unknown') for e in cluster_errors]
        paths = [self._extract_path_pattern(e.get('path', '')) for e in cluster_errors]
        
        characteristics = {
            'most_common_prediction': Counter(predicted_types).most_common(1)[0] if predicted_types else None,
            'most_common_path_pattern': Counter(paths).most_common(1)[0] if paths else None,
            'avg_confidence': np.mean([e.get('confidence', 0.5) for e in cluster_errors]),
            'size': len(cluster_errors)
        }
        
        return characteristics 