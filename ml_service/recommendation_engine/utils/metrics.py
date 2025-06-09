import numpy as np
from sklearn.metrics import precision_score, recall_score, ndcg_score
import logging

logger = logging.getLogger(__name__)

class RecommendationMetrics:
    @staticmethod
    def precision_at_k(recommended_items, relevant_items, k=10):
        """Calculate Precision@K"""
        if not recommended_items or not relevant_items:
            return 0.0
            
        recommended_k = recommended_items[:k]
        hits = len(set(recommended_k) & set(relevant_items))
        return hits / len(recommended_k)
        
    @staticmethod
    def recall_at_k(recommended_items, relevant_items, k=10):
        """Calculate Recall@K"""
        if not relevant_items:
            return 0.0
            
        recommended_k = recommended_items[:k]
        hits = len(set(recommended_k) & set(relevant_items))
        return hits / len(relevant_items)
        
    @staticmethod
    def ndcg_at_k(recommended_items, relevant_items, k=10):
        """Calculate NDCG@K"""
        if not recommended_items or not relevant_items:
            return 0.0
            
        recommended_k = recommended_items[:k]
        relevance_scores = [1 if item in relevant_items else 0 for item in recommended_k]
        
        if sum(relevance_scores) == 0:
            return 0.0
            
        # DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
        
        # IDCG
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0
        
    @staticmethod
    def map_at_k(recommended_items, relevant_items, k=10):
        """Calculate MAP@K"""
        if not recommended_items or not relevant_items:
            return 0.0
            
        recommended_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        ap = 0.0
        hits = 0
        
        for i, item in enumerate(recommended_k):
            if item in relevant_set:
                hits += 1
                ap += hits / (i + 1)
                
        return ap / len(relevant_items) if relevant_items else 0.0
        
    @staticmethod
    def coverage(recommended_items_list, total_items):
        """Calculate catalog coverage"""
        if not total_items:
            return 0.0
            
        recommended_unique = set()
        for recommendations in recommended_items_list:
            recommended_unique.update(recommendations)
            
        return len(recommended_unique) / len(total_items)
        
    @staticmethod
    def diversity(recommended_items_list):
        """Calculate intra-list diversity (average pairwise distance)"""
        if not recommended_items_list:
            return 0.0
            
        diversities = []
        for recommendations in recommended_items_list:
            if len(recommendations) < 2:
                diversities.append(0.0)
                continue
                
            # Simple diversity measure based on unique items
            unique_items = len(set(recommendations))
            diversity = unique_items / len(recommendations)
            diversities.append(diversity)
            
        return np.mean(diversities)
        
    @staticmethod
    def evaluate_recommendations(predictions, test_interactions, k=10):
        """Comprehensive evaluation of recommendations"""
        metrics = {}
        
        user_metrics = []
        all_recommended = []
        
        for user_id, recommended_items in predictions.items():
            # Get relevant items for this user
            user_interactions = test_interactions[test_interactions['user_id'] == user_id]
            relevant_items = user_interactions['item_id'].tolist()
            
            if not relevant_items:
                continue
                
            # Calculate metrics for this user
            user_precision = RecommendationMetrics.precision_at_k(recommended_items, relevant_items, k)
            user_recall = RecommendationMetrics.recall_at_k(recommended_items, relevant_items, k)
            user_ndcg = RecommendationMetrics.ndcg_at_k(recommended_items, relevant_items, k)
            user_map = RecommendationMetrics.map_at_k(recommended_items, relevant_items, k)
            
            user_metrics.append({
                'precision': user_precision,
                'recall': user_recall,
                'ndcg': user_ndcg,
                'map': user_map
            })
            
            all_recommended.append(recommended_items)
            
        # Calculate average metrics
        if user_metrics:
            metrics['precision_at_k'] = np.mean([m['precision'] for m in user_metrics])
            metrics['recall_at_k'] = np.mean([m['recall'] for m in user_metrics])
            metrics['ndcg_at_k'] = np.mean([m['ndcg'] for m in user_metrics])
            metrics['map_at_k'] = np.mean([m['map'] for m in user_metrics])
            
            # Global metrics
            all_items = test_interactions['item_id'].unique()
            metrics['coverage'] = RecommendationMetrics.coverage(all_recommended, all_items)
            metrics['diversity'] = RecommendationMetrics.diversity(all_recommended)
            
        return metrics 