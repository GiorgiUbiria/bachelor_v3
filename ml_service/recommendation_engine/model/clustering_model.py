import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class ClusteringModel:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.user_clusters = {}
        self.cluster_profiles = {}
        self.cluster_popular_items = {}
        
    def fit(self, user_features, interactions_data):
        """Train clustering model on user features"""
        try:
            # Scale user features
            user_features_scaled = self.scaler.fit_transform(user_features)
            
            # Fit clustering model
            cluster_labels = self.kmeans.fit_predict(user_features_scaled)
            
            # Store user cluster assignments
            for i, user_id in enumerate(user_features.index):
                self.user_clusters[user_id] = cluster_labels[i]
                
            # Build cluster profiles and popular items
            self._build_cluster_profiles(interactions_data)
            
            logger.info(f"Clustering model trained with {self.n_clusters} clusters for {len(user_features)} users")
            return self
            
        except Exception as e:
            logger.error(f"Error training clustering model: {e}")
            raise
            
    def _build_cluster_profiles(self, interactions_data):
        """Build profiles for each cluster"""
        try:
            for cluster_id in range(self.n_clusters):
                cluster_users = [uid for uid, cid in self.user_clusters.items() if cid == cluster_id]
                
                if not cluster_users:
                    continue
                    
                # Get interactions for users in this cluster
                cluster_interactions = interactions_data[
                    interactions_data['user_id'].isin(cluster_users)
                ]
                
                # Find popular items in this cluster
                if not cluster_interactions.empty:
                    item_popularity = cluster_interactions.groupby('item_id').agg({
                        'rating': ['count', 'mean']
                    }).reset_index()
                    
                    item_popularity.columns = ['item_id', 'count', 'avg_rating']
                    item_popularity['popularity_score'] = (
                        item_popularity['count'] * item_popularity['avg_rating']
                    )
                    
                    popular_items = item_popularity.sort_values(
                        'popularity_score', ascending=False
                    ).head(20)
                    
                    self.cluster_popular_items[cluster_id] = popular_items['item_id'].tolist()
                    
        except Exception as e:
            logger.error(f"Error building cluster profiles: {e}")
            
    def predict_cluster(self, user_features):
        """Predict cluster for new user"""
        try:
            user_features_scaled = self.scaler.transform([user_features])
            return self.kmeans.predict(user_features_scaled)[0]
        except Exception as e:
            logger.error(f"Error predicting cluster: {e}")
            return 0
            
    def recommend(self, user_id, n_recommendations=10):
        """Generate cluster-based recommendations"""
        try:
            if user_id not in self.user_clusters:
                return []
                
            cluster_id = self.user_clusters[user_id]
            
            if cluster_id not in self.cluster_popular_items:
                return []
                
            popular_items = self.cluster_popular_items[cluster_id][:n_recommendations]
            
            recommendations = []
            for i, item_id in enumerate(popular_items):
                recommendations.append({
                    'item_id': item_id,
                    'score': 1.0 - (i * 0.1),  # Decreasing score based on popularity rank
                    'method': 'cluster_based'
                })
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating cluster-based recommendations: {e}")
            return []
            
    def get_cluster_info(self, user_id):
        """Get cluster information for a user"""
        if user_id not in self.user_clusters:
            return None
            
        cluster_id = self.user_clusters[user_id]
        cluster_users = [uid for uid, cid in self.user_clusters.items() if cid == cluster_id]
        
        return {
            'cluster_id': cluster_id,
            'cluster_size': len(cluster_users),
            'popular_items': self.cluster_popular_items.get(cluster_id, [])
        } 