import numpy as np
import logging

logger = logging.getLogger(__name__)

class HybridModel:
    def __init__(self, weights=None):
        self.weights = weights or {
            'collaborative': 0.4,
            'content_based': 0.3,
            'clustering': 0.3
        }
        self.models = {}
        
    def add_model(self, name, model):
        """Add a component model to the hybrid system"""
        self.models[name] = model
        
    def fit(self, **kwargs):
        """Train all component models"""
        try:
            for name, model in self.models.items():
                if hasattr(model, 'fit'):
                    logger.info(f"Training {name} model...")
                    model.fit(**kwargs)
                    
            logger.info("Hybrid model training completed")
            return self
            
        except Exception as e:
            logger.error(f"Error training hybrid model: {e}")
            raise
            
    def recommend(self, user_id, n_recommendations=10, **kwargs):
        """Generate hybrid recommendations by combining multiple models"""
        try:
            all_recommendations = {}
            
            # Get recommendations from each model
            for name, model in self.models.items():
                if hasattr(model, 'recommend'):
                    try:
                        recs = model.recommend(user_id, n_recommendations * 2, **kwargs)
                        all_recommendations[name] = recs
                    except Exception as e:
                        logger.warning(f"Error getting recommendations from {name}: {e}")
                        all_recommendations[name] = []
                        
            # Combine recommendations using weighted scoring
            combined_scores = {}
            
            for method, recommendations in all_recommendations.items():
                weight = self.weights.get(method, 0.0)
                
                for rec in recommendations:
                    item_id = rec['item_id']
                    score = rec['score'] * weight
                    
                    if item_id not in combined_scores:
                        combined_scores[item_id] = {
                            'score': 0.0,
                            'methods': [],
                            'method_scores': {}
                        }
                        
                    combined_scores[item_id]['score'] += score
                    combined_scores[item_id]['methods'].append(method)
                    combined_scores[item_id]['method_scores'][method] = rec['score']
                    
            # Sort by combined score and return top recommendations
            sorted_items = sorted(
                combined_scores.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )
            
            final_recommendations = []
            for item_id, data in sorted_items[:n_recommendations]:
                final_recommendations.append({
                    'item_id': item_id,
                    'score': data['score'],
                    'method': 'hybrid',
                    'contributing_methods': data['methods'],
                    'method_scores': data['method_scores']
                })
                
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error generating hybrid recommendations: {e}")
            return []
            
    def set_weights(self, new_weights):
        """Update model weights"""
        self.weights = new_weights
        logger.info(f"Updated hybrid model weights: {new_weights}")
        
    def get_model_performance(self, user_id, true_items):
        """Evaluate individual model performance"""
        performance = {}
        
        for name, model in self.models.items():
            try:
                recs = model.recommend(user_id, 10)
                recommended_items = [rec['item_id'] for rec in recs]
                
                # Calculate precision@k
                hits = len(set(recommended_items) & set(true_items))
                precision = hits / len(recommended_items) if recommended_items else 0
                recall = hits / len(true_items) if true_items else 0
                
                performance[name] = {
                    'precision': precision,
                    'recall': recall,
                    'recommendations_count': len(recommended_items)
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {name} model: {e}")
                performance[name] = {'precision': 0, 'recall': 0, 'recommendations_count': 0}
                
        return performance 