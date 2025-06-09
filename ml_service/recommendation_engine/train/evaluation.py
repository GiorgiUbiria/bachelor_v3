import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from ..utils.metrics import RecommendationMetrics

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_model(self, model, interactions_df, test_size=0.2, k_values=[5, 10, 20]):
        """Comprehensive model evaluation"""
        try:
            # Create train/test split
            train_df, test_df = train_test_split(
                interactions_df, 
                test_size=test_size, 
                random_state=42,
                stratify=interactions_df['user_id'] if 'user_id' in interactions_df.columns else None
            )
            
            # Train model on training data
            if hasattr(model, 'fit'):
                model.fit(train_df)
                
            # Generate predictions for test users
            test_users = test_df['user_id'].unique()
            predictions = {}
            
            for user_id in test_users:
                try:
                    recommendations = model.recommend(user_id, n_recommendations=max(k_values))
                    predictions[user_id] = [rec['item_id'] for rec in recommendations]
                except Exception as e:
                    logger.warning(f"Could not generate recommendations for user {user_id}: {e}")
                    predictions[user_id] = []
                    
            # Evaluate for different k values
            results = {}
            for k in k_values:
                metrics = RecommendationMetrics.evaluate_recommendations(
                    predictions, test_df, k=k
                )
                results[f'k_{k}'] = metrics
                
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
            
    def compare_models(self, models_dict, interactions_df, k=10):
        """Compare multiple models"""
        comparison_results = {}
        
        for model_name, model in models_dict.items():
            logger.info(f"Evaluating {model_name}...")
            results = self.evaluate_model(model, interactions_df, k_values=[k])
            comparison_results[model_name] = results.get(f'k_{k}', {})
            
        return comparison_results
        
    def cross_validate_model(self, model, interactions_df, n_folds=5, k=10):
        """Cross-validation evaluation"""
        users = interactions_df['user_id'].unique()
        np.random.shuffle(users)
        
        fold_size = len(users) // n_folds
        fold_results = []
        
        for fold in range(n_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(users)
            
            test_users = users[start_idx:end_idx]
            train_users = np.concatenate([users[:start_idx], users[end_idx:]])
            
            train_df = interactions_df[interactions_df['user_id'].isin(train_users)]
            test_df = interactions_df[interactions_df['user_id'].isin(test_users)]
            
            # Train and evaluate
            if hasattr(model, 'fit'):
                model.fit(train_df)
                
            predictions = {}
            for user_id in test_users:
                try:
                    recommendations = model.recommend(user_id, n_recommendations=k)
                    predictions[user_id] = [rec['item_id'] for rec in recommendations]
                except:
                    predictions[user_id] = []
                    
            fold_metrics = RecommendationMetrics.evaluate_recommendations(
                predictions, test_df, k=k
            )
            fold_results.append(fold_metrics)
            
        # Average results across folds
        avg_results = {}
        if fold_results:
            for metric in fold_results[0].keys():
                values = [result.get(metric, 0) for result in fold_results]
                avg_results[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
                
        return avg_results
        
    def evaluate_cold_start(self, model, interactions_df, min_interactions=5):
        """Evaluate model performance on cold start users"""
        user_interaction_counts = interactions_df['user_id'].value_counts()
        
        # Identify cold start users (few interactions)
        cold_start_users = user_interaction_counts[
            user_interaction_counts < min_interactions
        ].index.tolist()
        
        warm_users = user_interaction_counts[
            user_interaction_counts >= min_interactions
        ].index.tolist()
        
        # Evaluate on both groups
        cold_start_df = interactions_df[interactions_df['user_id'].isin(cold_start_users)]
        warm_df = interactions_df[interactions_df['user_id'].isin(warm_users)]
        
        results = {}
        
        if not cold_start_df.empty:
            results['cold_start'] = self.evaluate_model(model, cold_start_df)
            
        if not warm_df.empty:
            results['warm_users'] = self.evaluate_model(model, warm_df)
            
        return results
        
    def evaluate_diversity_and_coverage(self, model, interactions_df, catalog_df):
        """Evaluate recommendation diversity and catalog coverage"""
        users = interactions_df['user_id'].unique()[:100]  # Sample users for efficiency
        all_recommendations = []
        
        for user_id in users:
            try:
                recommendations = model.recommend(user_id, n_recommendations=20)
                recommended_items = [rec['item_id'] for rec in recommendations]
                all_recommendations.append(recommended_items)
            except:
                continue
                
        if not all_recommendations:
            return {}
            
        # Calculate metrics
        total_items = set(catalog_df['item_id'].tolist())
        coverage = RecommendationMetrics.coverage(all_recommendations, total_items)
        diversity = RecommendationMetrics.diversity(all_recommendations)
        
        # Category diversity
        category_diversity = self._calculate_category_diversity(
            all_recommendations, catalog_df
        )
        
        return {
            'catalog_coverage': coverage,
            'intra_list_diversity': diversity,
            'category_diversity': category_diversity,
            'total_unique_recommended': len(set([item for recs in all_recommendations for item in recs]))
        }
        
    def _calculate_category_diversity(self, recommendations_list, catalog_df):
        """Calculate diversity across product categories"""
        category_diversities = []
        
        for recommendations in recommendations_list:
            if not recommendations:
                continue
                
            # Get categories for recommended items
            recommended_categories = []
            for item_id in recommendations:
                item_info = catalog_df[catalog_df['item_id'] == item_id]
                if not item_info.empty:
                    recommended_categories.append(item_info.iloc[0]['category'])
                    
            # Calculate category diversity
            unique_categories = len(set(recommended_categories))
            total_categories = len(recommended_categories)
            
            if total_categories > 0:
                category_diversities.append(unique_categories / total_categories)
                
        return np.mean(category_diversities) if category_diversities else 0.0 