import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Dict, List, Any, Optional
import copy
from datetime import datetime

from ..model.classifier import SecurityAnalyzer
from ..data.synthetic_generator import SyntheticDataGenerator
from ..utils.logger import setup_logger
from ..database.logger import SecurityDatabaseLogger

class AblationStudyFramework:
    def __init__(self):
        self.logger = setup_logger('AblationStudyFramework')
        self.generator = SyntheticDataGenerator()
        self.db_logger = SecurityDatabaseLogger()
        
    def run_comprehensive_ablation_study(self, test_size: int = 2000) -> Dict[str, Any]:
        """Run comprehensive ablation study on security analyzer components"""
        
        self.logger.info("Starting comprehensive ablation study")
        
        # Generate test dataset
        test_data = self.generator.generate_evaluation_set(test_size // 4)
        
        # Get baseline performance
        baseline_analyzer = SecurityAnalyzer()
        if not baseline_analyzer.is_trained:
            training_data = self.generator.generate_dataset(5000)
            baseline_analyzer.train(training_data)
        
        baseline_accuracy = self._evaluate_analyzer(baseline_analyzer, test_data)
        
        study_results = {
            'timestamp': datetime.now().isoformat(),
            'baseline_accuracy': baseline_accuracy,
            'test_samples': len(test_data),
            'component_results': {},
            'component_importance': {}
        }
        
        # Test each component removal
        components_to_test = [
            'pattern_matching',
            'ml_classifier', 
            'tfidf_vectorizer',
            'ensemble_decision',
            'danger_scoring'
        ]
        
        for component in components_to_test:
            self.logger.info(f"Testing ablation of component: {component}")
            
            try:
                reduced_accuracy = self._test_component_ablation(
                    component, test_data, baseline_analyzer
                )
                
                performance_impact = baseline_accuracy - reduced_accuracy
                component_importance = performance_impact / baseline_accuracy if baseline_accuracy > 0 else 0
                
                study_results['component_results'][component] = {
                    'reduced_accuracy': reduced_accuracy,
                    'performance_impact': performance_impact,
                    'importance_score': component_importance
                }
                
                study_results['component_importance'][component] = component_importance
                
                # Store in database
                self._store_ablation_result(
                    component, baseline_accuracy, reduced_accuracy, 
                    performance_impact, len(test_data)
                )
                
            except Exception as e:
                self.logger.error(f"Failed to test component {component}: {e}")
                study_results['component_results'][component] = {'error': str(e)}
        
        # Calculate overall insights
        study_results['insights'] = self._generate_ablation_insights(study_results)
        
        self.logger.info("Ablation study completed")
        return study_results
    
    def _test_component_ablation(self, component: str, test_data: List[Dict], 
                                baseline_analyzer: SecurityAnalyzer) -> float:
        """Test performance with specific component removed/disabled"""
        
        if component == 'pattern_matching':
            return self._test_without_pattern_matching(test_data, baseline_analyzer)
        elif component == 'ml_classifier':
            return self._test_pattern_matching_only(test_data, baseline_analyzer)
        elif component == 'tfidf_vectorizer':
            return self._test_without_tfidf(test_data, baseline_analyzer)
        elif component == 'ensemble_decision':
            return self._test_without_ensemble(test_data, baseline_analyzer)
        elif component == 'danger_scoring':
            return self._test_without_danger_scoring(test_data, baseline_analyzer)
        else:
            raise ValueError(f"Unknown component: {component}")
    
    def _test_without_pattern_matching(self, test_data: List[Dict], 
                                     baseline_analyzer: SecurityAnalyzer) -> float:
        """Test performance without pattern matching"""
        
        correct = 0
        total = len(test_data)
        
        for sample in test_data:
            from ..utils.schema import SecurityAnalysisRequest
            request = SecurityAnalysisRequest(**sample)
            
            # Simulate analysis without pattern matching
            request_dict = request.dict()
            preprocessed_text = baseline_analyzer.preprocessor.preprocess(request_dict)
            
            # Only use ML prediction
            ml_prediction = 'benign'
            if baseline_analyzer.is_trained:
                try:
                    X = baseline_analyzer.vectorizer.transform([preprocessed_text])
                    if hasattr(baseline_analyzer.classifier, 'ensemble_classifier'):
                        prediction = baseline_analyzer.classifier.ensemble_classifier.predict(X)[0]
                    else:
                        prediction = baseline_analyzer.simple_classifier.predict(X)[0]
                    ml_prediction = prediction
                except:
                    pass
            
            if ml_prediction == sample['attack_type']:
                correct += 1
        
        return correct / total
    
    def _test_pattern_matching_only(self, test_data: List[Dict], 
                                  baseline_analyzer: SecurityAnalyzer) -> float:
        """Test performance with only pattern matching (no ML)"""
        
        correct = 0
        total = len(test_data)
        
        for sample in test_data:
            request_dict = sample
            preprocessed_text = baseline_analyzer.preprocessor.preprocess(request_dict)
            
            # Only use pattern matching
            pattern_type, _ = baseline_analyzer.pattern_matcher.analyze(
                preprocessed_text, request_dict
            )
            
            if pattern_type == sample['attack_type']:
                correct += 1
        
        return correct / total
    
    def _test_without_tfidf(self, test_data: List[Dict], 
                          baseline_analyzer: SecurityAnalyzer) -> float:
        """Test performance without TF-IDF (pattern matching only)"""
        # For this implementation, without TF-IDF we fall back to pattern matching
        return self._test_pattern_matching_only(test_data, baseline_analyzer)
    
    def _test_without_ensemble(self, test_data: List[Dict], 
                             baseline_analyzer: SecurityAnalyzer) -> float:
        """Test performance without ensemble decision (ML only)"""
        return self._test_without_pattern_matching(test_data, baseline_analyzer)
    
    def _test_without_danger_scoring(self, test_data: List[Dict], 
                                   baseline_analyzer: SecurityAnalyzer) -> float:
        """Test accuracy without danger scoring (classification accuracy unaffected)"""
        # Danger scoring doesn't affect classification accuracy, only attack scores
        return self._evaluate_analyzer(baseline_analyzer, test_data)
    
    def _evaluate_analyzer(self, analyzer: SecurityAnalyzer, test_data: List[Dict]) -> float:
        """Evaluate analyzer accuracy on test data"""
        
        correct = 0
        total = len(test_data)
        
        for sample in test_data:
            from ..utils.schema import SecurityAnalysisRequest
            request = SecurityAnalysisRequest(**sample)
            result = analyzer.analyze(request)
            
            if result.attack_type == sample['attack_type']:
                correct += 1
        
        return correct / total
    
    def _store_ablation_result(self, component: str, baseline_accuracy: float,
                             reduced_accuracy: float, performance_impact: float,
                             test_samples: int):
        """Store ablation study result in database"""
        
        if not self.db_logger.db_available:
            return
        
        try:
            session = self.db_logger.SessionLocal()
            from ..database.models import AblationStudyResults
            
            result = AblationStudyResults(
                study_name="comprehensive_ablation_study",
                component_removed=component,
                baseline_accuracy=baseline_accuracy,
                reduced_accuracy=reduced_accuracy,
                performance_impact=performance_impact,
                component_importance=performance_impact / baseline_accuracy if baseline_accuracy > 0 else 0,
                test_samples=test_samples
            )
            
            session.add(result)
            session.commit()
            session.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store ablation result: {e}")
    
    def _generate_ablation_insights(self, study_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from ablation study results"""
        
        component_importance = study_results.get('component_importance', {})
        
        if not component_importance:
            return {'error': 'No component importance data available'}
        
        # Rank components by importance
        ranked_components = sorted(
            component_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Find most and least critical components
        most_critical = ranked_components[0] if ranked_components else None
        least_critical = ranked_components[-1] if ranked_components else None
        
        # Calculate redundancy (components with low individual impact)
        redundant_components = [
            comp for comp, importance in component_importance.items() 
            if importance < 0.05  # Less than 5% impact
        ]
        
        return {
            'component_ranking': ranked_components,
            'most_critical_component': {
                'name': most_critical[0] if most_critical else None,
                'importance': most_critical[1] if most_critical else 0
            },
            'least_critical_component': {
                'name': least_critical[0] if least_critical else None,
                'importance': least_critical[1] if least_critical else 0
            },
            'redundant_components': redundant_components,
            'system_robustness': len(redundant_components) / max(len(component_importance), 1),
            'recommendations': self._generate_recommendations(component_importance)
        }
    
    def _generate_recommendations(self, component_importance: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations based on ablation study"""
        
        recommendations = []
        
        # Check for highly important components
        for component, importance in component_importance.items():
            if importance > 0.2:  # More than 20% impact
                recommendations.append(
                    f"Critical component '{component}' has high impact ({importance:.1%}). "
                    f"Consider additional validation and backup strategies."
                )
        
        # Check for low-impact components
        low_impact = [comp for comp, imp in component_importance.items() if imp < 0.05]
        if low_impact:
            recommendations.append(
                f"Components {low_impact} have minimal impact and could be simplified or removed."
            )
        
        # Check for balanced system
        importance_std = np.std(list(component_importance.values()))
        if importance_std < 0.1:
            recommendations.append(
                "System shows balanced component importance, indicating good architectural design."
            )
        
        return recommendations 