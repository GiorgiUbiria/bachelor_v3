import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)

class CollaborativeFilteringModel:
    def __init__(self, n_components=50, algorithm='svd'):
        self.n_components = n_components
        self.algorithm = algorithm
        self.model = None
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.user_ids = []
        self.item_ids = []
        
    def fit(self, interactions_data):
        """Train collaborative filtering model on user-item interactions"""
        try:
            user_item_df = interactions_data.pivot_table(
                index='user_id', 
                columns='item_id', 
                values='rating',
                fill_value=0
            )
            
            self.user_ids = user_item_df.index.tolist()
            self.item_ids = user_item_df.columns.tolist()
            self.user_item_matrix = csr_matrix(user_item_df.values)
            
            if self.algorithm == 'svd':
                self.model = TruncatedSVD(n_components=self.n_components, random_state=42)
                self.user_factors = self.model.fit_transform(self.user_item_matrix)
                self.item_factors = self.model.components_.T
                
            elif self.algorithm == 'user_based':
                self.user_similarity = cosine_similarity(self.user_item_matrix)
                
            elif self.algorithm == 'item_based':
                self.item_similarity = cosine_similarity(self.user_item_matrix.T)
                
            logger.info(f"Collaborative filtering model trained with {len(self.user_ids)} users and {len(self.item_ids)} items")
            return self
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering model: {e}")
            raise
            
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if self.model is None:
            return 0.0
            
        try:
            if user_id not in self.user_ids or item_id not in self.item_ids:
                return 0.0
                
            user_idx = self.user_ids.index(user_id)
            item_idx = self.item_ids.index(item_id)
            
            if self.algorithm == 'svd':
                prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                return max(0, min(5, prediction))
                
            elif self.algorithm == 'user_based':
                user_ratings = self.user_item_matrix[user_idx].toarray()[0]
                similar_users = self.user_similarity[user_idx]
                
                numerator = np.dot(similar_users, self.user_item_matrix[:, item_idx].toarray().flatten())
                denominator = np.sum(np.abs(similar_users))
                
                if denominator == 0:
                    return 0.0
                return numerator / denominator
                
            return 0.0
            
        except Exception as e:
            logger.error(f"Error predicting for user {user_id}, item {item_id}: {e}")
            return 0.0
            
    def recommend(self, user_id, n_recommendations=10, exclude_seen=True):
        """Generate recommendations for a user"""
        if user_id not in self.user_ids:
            return []
            
        try:
            user_idx = self.user_ids.index(user_id)
            
            if self.algorithm == 'svd':
                user_vector = self.user_factors[user_idx]
                scores = np.dot(self.item_factors, user_vector)
                
            elif self.algorithm == 'user_based':
                user_ratings = self.user_item_matrix[user_idx].toarray()[0]
                similar_users = self.user_similarity[user_idx]
                scores = np.dot(similar_users, self.user_item_matrix.toarray())
                
            else:
                return []
                
            if exclude_seen:
                seen_items = self.user_item_matrix[user_idx].nonzero()[1]
                scores[seen_items] = -np.inf
                
            top_items = np.argsort(scores)[::-1][:n_recommendations]
            
            recommendations = []
            for item_idx in top_items:
                if scores[item_idx] > -np.inf:
                    recommendations.append({
                        'item_id': self.item_ids[item_idx],
                        'score': float(scores[item_idx]),
                        'method': 'collaborative_filtering'
                    })
                    
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return [] 