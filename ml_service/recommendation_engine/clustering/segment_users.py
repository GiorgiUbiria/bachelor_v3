import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class UserSegmentation:
    def __init__(self, method='kmeans', n_clusters=5):
        self.method = method
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.feature_names = []
        self.cluster_profiles = {}
        
    def segment_users(self, user_features_df, interactions_df):
        """Segment users into clusters based on behavior and demographics"""
        try:
            # Prepare features
            features_df = self._prepare_clustering_features(user_features_df, interactions_df)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_df)
            
            # Apply PCA for dimensionality reduction
            features_pca = self.pca.fit_transform(features_scaled)
            
            # Apply clustering
            if self.method == 'kmeans':
                self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
                cluster_labels = self.model.fit_predict(features_pca)
            elif self.method == 'dbscan':
                self.model = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = self.model.fit_predict(features_pca)
            else:
                raise ValueError(f"Unknown clustering method: {self.method}")
                
            # Add cluster labels to features
            features_df['cluster'] = cluster_labels
            
            # Build cluster profiles
            self._build_cluster_profiles(features_df, interactions_df)
            
            logger.info(f"User segmentation completed: {len(np.unique(cluster_labels))} clusters")
            return features_df
            
        except Exception as e:
            logger.error(f"Error in user segmentation: {e}")
            return user_features_df
            
    def _prepare_clustering_features(self, user_features_df, interactions_df):
        """Prepare comprehensive features for clustering"""
        features_df = user_features_df.copy()
        
        # Behavioral features from interactions
        user_behavior = interactions_df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std'],
            'item_id': 'nunique',
            'timestamp': ['min', 'max']
        }).reset_index()
        
        user_behavior.columns = [
            'user_id', 'total_interactions', 'avg_rating', 'rating_std', 
            'unique_items', 'first_interaction', 'last_interaction'
        ]
        
        # Calculate interaction span in days
        user_behavior['interaction_span_days'] = (
            user_behavior['last_interaction'] - user_behavior['first_interaction']
        ).dt.days
        
        user_behavior['rating_std'] = user_behavior['rating_std'].fillna(0)
        
        # Category preferences
        category_interactions = interactions_df.merge(
            features_df[['user_id']], on='user_id', how='inner'
        )
        
        if 'category' in category_interactions.columns:
            category_prefs = category_interactions.groupby(['user_id', 'category']).size().unstack(fill_value=0)
            # Normalize to percentages
            category_prefs = category_prefs.div(category_prefs.sum(axis=1), axis=0)
            
            # Merge all features
            features_combined = features_df.merge(user_behavior, on='user_id', how='left')
            features_combined = features_combined.merge(category_prefs, left_on='user_id', right_index=True, how='left')
        else:
            features_combined = features_df.merge(user_behavior, on='user_id', how='left')
            
        # Fill missing values
        features_combined = features_combined.fillna(0)
        
        # Select only numeric columns for clustering
        numeric_cols = features_combined.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'user_id']
        
        self.feature_names = numeric_cols
        return features_combined[['user_id'] + numeric_cols]
        
    def _build_cluster_profiles(self, clustered_features, interactions_df):
        """Build descriptive profiles for each cluster"""
        for cluster_id in clustered_features['cluster'].unique():
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
                
            cluster_users = clustered_features[clustered_features['cluster'] == cluster_id]['user_id'].tolist()
            
            # Demographic profile
            cluster_demo = clustered_features[clustered_features['cluster'] == cluster_id]
            
            # Behavioral profile
            cluster_interactions = interactions_df[interactions_df['user_id'].isin(cluster_users)]
            
            profile = {
                'cluster_id': cluster_id,
                'size': len(cluster_users),
                'demographics': {
                    'avg_age': cluster_demo['age'].mean() if 'age' in cluster_demo.columns else None,
                    'avg_interactions': cluster_demo['total_interactions'].mean() if 'total_interactions' in cluster_demo.columns else None,
                    'avg_rating': cluster_demo['avg_rating'].mean() if 'avg_rating' in cluster_demo.columns else None
                },
                'top_categories': [],
                'characteristics': []
            }
            
            # Category preferences
            if not cluster_interactions.empty and 'category' in cluster_interactions.columns:
                top_categories = cluster_interactions['category'].value_counts().head(5)
                profile['top_categories'] = top_categories.to_dict()
                
            # Characteristics based on feature values
            for feature in self.feature_names:
                if feature in cluster_demo.columns:
                    feature_mean = cluster_demo[feature].mean()
                    overall_mean = clustered_features[feature].mean()
                    
                    if feature_mean > overall_mean * 1.2:
                        profile['characteristics'].append(f"High {feature}")
                    elif feature_mean < overall_mean * 0.8:
                        profile['characteristics'].append(f"Low {feature}")
                        
            self.cluster_profiles[cluster_id] = profile
            
    def get_cluster_summary(self):
        """Get summary of all clusters"""
        return {
            'method': self.method,
            'n_clusters': len(self.cluster_profiles),
            'profiles': self.cluster_profiles
        }
        
    def predict_cluster(self, user_features):
        """Predict cluster for new user"""
        if self.model is None:
            return -1
            
        try:
            # Prepare features in same format as training
            features_scaled = self.scaler.transform([user_features])
            features_pca = self.pca.transform(features_scaled)
            
            if self.method == 'kmeans':
                return self.model.predict(features_pca)[0]
            else:
                return -1  # DBSCAN doesn't support prediction
                
        except Exception as e:
            logger.error(f"Error predicting cluster: {e}")
            return -1 