import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class ContentBasedModel:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True
        )
        self.item_profiles = None
        self.user_profiles = None
        self.item_similarity_matrix = None
        self.items_data = None
        
    def fit(self, items_data, interactions_data=None):
        """Train content-based model on item features"""
        try:
            self.items_data = items_data.copy()
            
            # Create item content features
            item_content = items_data['name'].fillna('') + ' ' + items_data['description'].fillna('')
            self.item_profiles = self.tfidf_vectorizer.fit_transform(item_content)
            
            # Calculate item similarity matrix
            self.item_similarity_matrix = cosine_similarity(self.item_profiles)
            
            # Build user profiles from interactions if available
            if interactions_data is not None:
                self._build_user_profiles(interactions_data)
                
            logger.info(f"Content-based model trained with {len(items_data)} items")
            return self
            
        except Exception as e:
            logger.error(f"Error training content-based model: {e}")
            raise
            
    def _build_user_profiles(self, interactions_data):
        """Build user profiles from their interaction history"""
        try:
            user_profiles = {}
            
            for user_id in interactions_data['user_id'].unique():
                user_interactions = interactions_data[interactions_data['user_id'] == user_id]
                
                # Weight items by rating/interaction strength
                weighted_profiles = []
                total_weight = 0
                
                for _, interaction in user_interactions.iterrows():
                    item_id = interaction['item_id']
                    rating = interaction.get('rating', 1.0)
                    
                    if item_id in self.items_data['item_id'].values:
                        item_idx = self.items_data[self.items_data['item_id'] == item_id].index[0]
                        item_profile = self.item_profiles[item_idx].toarray()[0]
                        weighted_profiles.append(item_profile * rating)
                        total_weight += rating
                        
                if weighted_profiles and total_weight > 0:
                    user_profile = np.sum(weighted_profiles, axis=0) / total_weight
                    user_profiles[user_id] = user_profile
                    
            self.user_profiles = user_profiles
            
        except Exception as e:
            logger.error(f"Error building user profiles: {e}")
            
    def recommend(self, user_id=None, item_id=None, n_recommendations=10):
        """Generate content-based recommendations"""
        try:
            if item_id is not None:
                # Item-to-item recommendations
                return self._recommend_similar_items(item_id, n_recommendations)
            elif user_id is not None and self.user_profiles:
                # User profile-based recommendations
                return self._recommend_for_user_profile(user_id, n_recommendations)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error generating content-based recommendations: {e}")
            return []
            
    def _recommend_similar_items(self, item_id, n_recommendations):
        """Recommend items similar to given item"""
        if item_id not in self.items_data['item_id'].values:
            return []
            
        item_idx = self.items_data[self.items_data['item_id'] == item_id].index[0]
        similarities = self.item_similarity_matrix[item_idx]
        
        # Get most similar items (excluding the item itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'item_id': self.items_data.iloc[idx]['item_id'],
                'score': float(similarities[idx]),
                'method': 'content_based_similar'
            })
            
        return recommendations
        
    def _recommend_for_user_profile(self, user_id, n_recommendations):
        """Recommend items based on user profile"""
        if user_id not in self.user_profiles:
            return []
            
        user_profile = self.user_profiles[user_id]
        
        # Calculate similarity between user profile and all items
        similarities = cosine_similarity([user_profile], self.item_profiles)[0]
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'item_id': self.items_data.iloc[idx]['item_id'],
                'score': float(similarities[idx]),
                'method': 'content_based_profile'
            })
            
        return recommendations 