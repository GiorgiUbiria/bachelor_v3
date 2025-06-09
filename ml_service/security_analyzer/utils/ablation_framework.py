import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import copy

logger = logging.getLogger(__name__)

class AblationFramework:
    def __init__(self):
        self.ablation_results = {}
        self.baseline_performance = {}
        self.component_contributions = {}
        
    def set_baseline_performance(self, metrics: Dict[str, float]):
        """Set baseline performance metrics for comparison"""
        try:
            self.baseline_performance = metrics.copy()
            logger.info(f"Baseline performance set: {metrics}")
        except Exception as e:
            logger.error(f"Error setting baseline performance: {e}")
            
    def run_component_ablation(self, component_name: str, 
                              test_function, test_data,
                              component_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Run ablation study for a specific component"""
        try:
            logger.info(f"Running ablation study for component: {component_name}")
            
            # Test with component enabled
            with_component = test_function(test_data, include_component=True, 
                                         component_params=component_params)
            
            # Test with component disabled
            without_component = test_function(test_data, include_component=False,
                                            component_params=component_params)
            
            # Calculate component contribution
            contribution = self._calculate_contribution(with_component, without_component)
            
            result = {
                'component_name': component_name,
                'with_component': with_component,
                'without_component': without_component,
                'contribution': contribution,
                'timestamp': datetime.now().isoformat()
            }
            
            self.ablation_results[component_name] = result
            self.component_contributions[component_name] = contribution
            
            return result
            
        except Exception as e:
            logger.error(f"Error running ablation for {component_name}: {e}")
            return {'error': str(e)}
            
    def _calculate_contribution(self, with_metrics: Dict[str, float], 
                              without_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate component contribution to performance"""
        try:
            contribution = {}
            
            for metric_name in with_metrics:
                if metric_name in without_metrics:
                    with_value = with_metrics[metric_name]
                    without_value = without_metrics[metric_name]
                    
                    # Calculate absolute and relative contribution
                    absolute_contrib = with_value - without_value
                    relative_contrib = (absolute_contrib / without_value * 100) if without_value != 0 else 0
                    
                    contribution[f'{metric_name}_absolute'] = absolute_contrib
                    contribution[f'{metric_name}_relative'] = relative_contrib
                    
            return contribution
            
        except Exception as e:
            logger.error(f"Error calculating contribution: {e}")
            return {}
            
    def run_comprehensive_ablation(self, components: List[str],
                                 test_function, test_data) -> Dict[str, Any]:
        """Run ablation study for multiple components"""
        try:
            logger.info(f"Running comprehensive ablation for {len(components)} components")
            
            comprehensive_results = {
                'individual_results': {},
                'component_ranking': [],
                'interaction_effects': {},
                'summary': {}
            }
            
            # Test individual components
            for component in components:
                result = self.run_component_ablation(component, test_function, test_data)
                comprehensive_results['individual_results'][component] = result
                
            # Rank components by contribution
            ranking = self._rank_components_by_contribution()
            comprehensive_results['component_ranking'] = ranking
            
            # Test component interactions (pairwise)
            if len(components) > 1:
                interactions = self._test_component_interactions(components, test_function, test_data)
                comprehensive_results['interaction_effects'] = interactions
                
            # Generate summary
            summary = self._generate_ablation_summary()
            comprehensive_results['summary'] = summary
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Error running comprehensive ablation: {e}")
            return {'error': str(e)}
            
    def _rank_components_by_contribution(self) -> List[Dict[str, Any]]:
        """Rank components by their contribution to performance"""
        try:
            ranking = []
            
            for component_name, contribution in self.component_contributions.items():
                # Use accuracy as primary ranking metric (can be customized)
                primary_metric = contribution.get('accuracy_absolute', 0)
                
                ranking.append({
                    'component': component_name,
                    'contribution_score': primary_metric,
                    'all_contributions': contribution
                })
                
            # Sort by contribution score (descending)
            ranking.sort(key=lambda x: x['contribution_score'], reverse=True)
            
            return ranking
            
        except Exception as e:
            logger.error(f"Error ranking components: {e}")
            return []
            
    def _test_component_interactions(self, components: List[str],
                                   test_function, test_data) -> Dict[str, Any]:
        """Test interactions between component pairs"""
        try:
            interactions = {}
            
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    comp1, comp2 = components[i], components[j]
                    
                    # Test with both components
                    with_both = test_function(test_data, 
                                            include_components=[comp1, comp2])
                    
                    # Test with neither component
                    without_both = test_function(test_data, 
                                               exclude_components=[comp1, comp2])
                    
                    # Calculate interaction effect
                    individual_effects = (
                        self.component_contributions.get(comp1, {}).get('accuracy_absolute', 0) +
                        self.component_contributions.get(comp2, {}).get('accuracy_absolute', 0)
                    )
                    
                    combined_effect = with_both.get('accuracy', 0) - without_both.get('accuracy', 0)
                    interaction_effect = combined_effect - individual_effects
                    
                    interactions[f'{comp1}_{comp2}'] = {
                        'component_1': comp1,
                        'component_2': comp2,
                        'individual_effects_sum': individual_effects,
                        'combined_effect': combined_effect,
                        'interaction_effect': interaction_effect,
                        'synergy': 'positive' if interaction_effect > 0 else 'negative'
                    }
                    
            return interactions
            
        except Exception as e:
            logger.error(f"Error testing component interactions: {e}")
            return {}
            
    def _generate_ablation_summary(self) -> Dict[str, Any]:
        """Generate summary of ablation study results"""
        try:
            if not self.component_contributions:
                return {'message': 'No ablation results available'}
                
            summary = {
                'total_components_tested': len(self.component_contributions),
                'most_impactful_component': None,
                'least_impactful_component': None,
                'average_contribution': {},
                'recommendations': []
            }
            
            # Find most and least impactful components
            contributions = {}
            for comp, metrics in self.component_contributions.items():
                contributions[comp] = metrics.get('accuracy_absolute', 0)
                
            if contributions:
                most_impactful = max(contributions, key=contributions.get)
                least_impactful = min(contributions, key=contributions.get)
                
                summary['most_impactful_component'] = {
                    'name': most_impactful,
                    'contribution': contributions[most_impactful]
                }
                summary['least_impactful_component'] = {
                    'name': least_impactful,
                    'contribution': contributions[least_impactful]
                }
                
            # Calculate average contributions
            all_metrics = set()
            for metrics in self.component_contributions.values():
                all_metrics.update(metrics.keys())
                
            for metric in all_metrics:
                values = [metrics.get(metric, 0) for metrics in self.component_contributions.values()]
                summary['average_contribution'][metric] = np.mean(values) if values else 0
                
            # Generate recommendations
            recommendations = self._generate_recommendations()
            summary['recommendations'] = recommendations
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {'error': str(e)}
            
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on ablation results"""
        try:
            recommendations = []
            
            if not self.component_contributions:
                return ['No ablation data available for recommendations']
                
            # Analyze contributions
            contributions = {}
            for comp, metrics in self.component_contributions.items():
                contributions[comp] = metrics.get('accuracy_absolute', 0)
                
            if contributions:
                max_contrib = max(contributions.values())
                min_contrib = min(contributions.values())
                
                # High impact components
                high_impact = [comp for comp, contrib in contributions.items() 
                             if contrib > max_contrib * 0.7]
                if high_impact:
                    recommendations.append(
                        f"Focus optimization efforts on high-impact components: {', '.join(high_impact)}"
                    )
                
                # Low impact components
                low_impact = [comp for comp, contrib in contributions.items() 
                            if contrib < max_contrib * 0.1]
                if low_impact:
                    recommendations.append(
                        f"Consider simplifying or removing low-impact components: {', '.join(low_impact)}"
                    )
                
                # Negative impact components
                negative_impact = [comp for comp, contrib in contributions.items() if contrib < 0]
                if negative_impact:
                    recommendations.append(
                        f"Investigate components with negative impact: {', '.join(negative_impact)}"
                    )
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ['Error generating recommendations']
            
    def export_results(self, format: str = 'dict') -> Any:
        """Export ablation study results"""
        try:
            export_data = {
                'baseline_performance': self.baseline_performance,
                'ablation_results': self.ablation_results,
                'component_contributions': self.component_contributions,
                'export_timestamp': datetime.now().isoformat()
            }
            
            if format.lower() == 'json':
                import json
                return json.dumps(export_data, indent=2)
            elif format.lower() == 'csv':
                # Convert to DataFrame for CSV export
                df_data = []
                for comp, metrics in self.component_contributions.items():
                    row = {'component': comp}
                    row.update(metrics)
                    df_data.append(row)
                return pd.DataFrame(df_data)
            else:
                return export_data
                
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return {'error': str(e)}
            
    def clear_results(self):
        """Clear all ablation study results"""
        try:
            self.ablation_results.clear()
            self.baseline_performance.clear()
            self.component_contributions.clear()
            logger.info("Ablation study results cleared")
        except Exception as e:
            logger.error(f"Error clearing results: {e}")

# Example test functions for security analyzer ablation
class SecurityAblationTests:
    @staticmethod
    def test_pattern_matching_component(test_data, include_component=True, **kwargs):
        """Test pattern matching component contribution"""
        # Mock implementation - replace with actual security analyzer logic
        base_accuracy = 0.75
        pattern_contribution = 0.15 if include_component else 0
        
        return {
            'accuracy': base_accuracy + pattern_contribution,
            'precision': 0.80 + (0.10 if include_component else 0),
            'recall': 0.70 + (0.12 if include_component else 0),
            'f1_score': 0.75 + (0.11 if include_component else 0)
        }
    
    @staticmethod
    def test_ml_classifier_component(test_data, include_component=True, **kwargs):
        """Test ML classifier component contribution"""
        # Mock implementation
        base_accuracy = 0.60
        ml_contribution = 0.25 if include_component else 0
        
        return {
            'accuracy': base_accuracy + ml_contribution,
            'precision': 0.65 + (0.20 if include_component else 0),
            'recall': 0.55 + (0.25 if include_component else 0),
            'f1_score': 0.60 + (0.22 if include_component else 0)
        }
    
    @staticmethod
    def test_ensemble_component(test_data, include_component=True, **kwargs):
        """Test ensemble method component contribution"""
        # Mock implementation
        base_accuracy = 0.82
        ensemble_contribution = 0.08 if include_component else 0
        
        return {
            'accuracy': base_accuracy + ensemble_contribution,
            'precision': 0.85 + (0.06 if include_component else 0),
            'recall': 0.80 + (0.09 if include_component else 0),
            'f1_score': 0.82 + (0.07 if include_component else 0)
        }

# Global ablation framework instance
ablation_framework = AblationFramework() 