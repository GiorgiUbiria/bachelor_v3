import numpy as np
import pandas as pd
import logging
from ..train.evaluation import ModelEvaluator
from ..model.hybrid_model import HybridModel

logger = logging.getLogger(__name__)

class AblationStudy:
    def __init__(self):
        self.evaluator = ModelEvaluator()
        
    def analyze_hybrid_contributions(self, trained_models, interactions_df, k=10):
        """Analyze contribution of each component in hybrid model"""
        logger.info("Starting ablation study for hybrid model...")
        
        try:
            results = {}
            base_weights = {'collaborative': 0.4, 'content_based': 0.3, 'clustering': 0.3}
            
            # Test removing each component
            component_names = ['collaborative', 'content_based', 'clustering']
            
            for excluded_component in component_names:
                logger.info(f"Testing without {excluded_component}...")
                
                # Create hybrid model without this component
                modified_weights = base_weights.copy()
                excluded_weight = modified_weights.pop(excluded_component)
                
                # Redistribute weight to remaining components
                remaining_components = len(modified_weights)
                if remaining_components > 0:
                    weight_addition = excluded_weight / remaining_components
                    for comp in modified_weights:
                        modified_weights[comp] += weight_addition
                        
                # Create and evaluate modified hybrid model
                hybrid_ablated = HybridModel(weights=modified_weights)
                for name, model in trained_models.items():
                    if name != excluded_component and name != 'hybrid' and name != 'pricing':
                        hybrid_ablated.add_model(name, model)
                        
                # Evaluate performance
                evaluation_results = self.evaluator.evaluate_model(hybrid_ablated, interactions_df, k_values=[k])
                results[f'without_{excluded_component}'] = evaluation_results.get(f'k_{k}', {})
                
            # Evaluate full hybrid model for comparison
            full_hybrid = HybridModel(weights=base_weights)
            for name, model in trained_models.items():
                if name != 'hybrid' and name != 'pricing':
                    full_hybrid.add_model(name, model)
                    
            full_results = self.evaluator.evaluate_model(full_hybrid, interactions_df, k_values=[k])
            results['full_hybrid'] = full_results.get(f'k_{k}', {})
            
            # Calculate component contributions
            contributions = self._calculate_contributions(results)
            
            return {
                'ablation_results': results,
                'component_contributions': contributions,
                'summary': self._generate_ablation_summary(results, contributions)
            }
            
        except Exception as e:
            logger.error(f"Error in ablation study: {e}")
            return {}
            
    def _calculate_contributions(self, results):
        """Calculate each component's contribution to performance"""
        contributions = {}
        
        if 'full_hybrid' not in results:
            return contributions
            
        full_performance = results['full_hybrid']
        
        for metric in ['precision_at_k', 'recall_at_k', 'ndcg_at_k']:
            if metric not in full_performance:
                continue
                
            full_score = full_performance[metric]
            contributions[metric] = {}
            
            for ablation_key in results:
                if ablation_key.startswith('without_'):
                    component = ablation_key.replace('without_', '')
                    ablated_score = results[ablation_key].get(metric, 0)
                    
                    # Contribution = performance drop when component is removed
                    contribution = full_score - ablated_score
                    contributions[metric][component] = contribution
                    
        return contributions
        
    def _generate_ablation_summary(self, results, contributions):
        """Generate human-readable summary of ablation study"""
        summary = ["ABLATION STUDY SUMMARY", "=" * 25, ""]
        
        if 'full_hybrid' in results:
            full_perf = results['full_hybrid']
            summary.append("Full Hybrid Model Performance:")
            for metric, value in full_perf.items():
                summary.append(f"  {metric}: {value:.4f}")
            summary.append("")
            
        summary.append("Component Contributions:")
        summary.append("-" * 25)
        
        for metric, component_contribs in contributions.items():
            summary.append(f"\n{metric.upper()}:")
            
            # Sort components by contribution
            sorted_contribs = sorted(component_contribs.items(), key=lambda x: x[1], reverse=True)
            
            for component, contribution in sorted_contribs:
                impact = "positive" if contribution > 0 else "negative"
                summary.append(f"  {component}: {contribution:+.4f} ({impact} impact)")
                
        return "\n".join(summary)
        
    def test_weight_sensitivity(self, trained_models, interactions_df, weight_ranges=None):
        """Test sensitivity to different weight combinations"""
        if weight_ranges is None:
            weight_ranges = {
                'collaborative': [0.2, 0.4, 0.6],
                'content_based': [0.1, 0.3, 0.5], 
                'clustering': [0.1, 0.3, 0.5]
            }
            
        logger.info("Testing weight sensitivity...")
        
        sensitivity_results = []
        
        # Generate weight combinations
        from itertools import product
        weight_combinations = list(product(*weight_ranges.values()))
        
        for weights_tuple in weight_combinations:
            # Normalize weights to sum to 1
            total_weight = sum(weights_tuple)
            if total_weight == 0:
                continue
                
            normalized_weights = [w / total_weight for w in weights_tuple]
            weight_dict = dict(zip(weight_ranges.keys(), normalized_weights))
            
            try:
                # Create hybrid model with these weights
                hybrid_model = HybridModel(weights=weight_dict)
                for name, model in trained_models.items():
                    if name in weight_dict:
                        hybrid_model.add_model(name, model)
                        
                # Evaluate
                results = self.evaluator.evaluate_model(hybrid_model, interactions_df, k_values=[10])
                performance = results.get('k_10', {})
                
                sensitivity_results.append({
                    'weights': weight_dict,
                    'performance': performance
                })
                
            except Exception as e:
                logger.warning(f"Error testing weights {weight_dict}: {e}")
                continue
                
        return sensitivity_results 