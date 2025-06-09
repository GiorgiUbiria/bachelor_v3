import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging
from ..model import *
from ..data import DataPreprocessor, FeatureExtractor
from ..config import RecommendationConfig
from ..utils.metrics import RecommendationMetrics

logger = logging.getLogger(__name__)

class HybridTrainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.models = {}
        self.config = RecommendationConfig()
        
    def train_all_models(self, interactions_df, items_df, users_df):
        """Train all recommendation models"""
        logger.info("Starting comprehensive model training...")
        
        try:
            # Preprocess data
            interactions_clean = self.preprocessor.preprocess_interactions(interactions_df)
            items_clean = self.preprocessor.preprocess_items(items_df)
            users_clean = self.preprocessor.preprocess_users(users_df)
            
            # Extract features
            user_features = self.feature_extractor.extract_user_features(users_clean, interactions_clean)
            
            # Train individual models
            self._train_collaborative_model(interactions_clean)
            self._train_content_based_model(items_clean, interactions_clean)
            self._train_clustering_model(user_features, interactions_clean)
            self._train_hybrid_model()
            
            # Generate sample pricing data and train pricing model
            pricing_data = self._generate_sample_pricing_data(items_clean, interactions_clean)
            self._train_pricing_model(pricing_data)
            
            # Save all models
            self._save_models()
            
            logger.info("All models trained successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
            
    def _train_collaborative_model(self, interactions_df):
        """Train collaborative filtering model"""
        logger.info("Training collaborative filtering model...")
        
        model = CollaborativeFilteringModel(
            n_components=self.config.SVD_COMPONENTS,
            algorithm='svd'
        )
        
        model.fit(interactions_df)
        self.models['collaborative'] = model
        
    def _train_content_based_model(self, items_df, interactions_df):
        """Train content-based filtering model"""
        logger.info("Training content-based model...")
        
        model = ContentBasedModel(max_features=self.config.MAX_FEATURES_TFIDF)
        model.fit(items_df, interactions_df)
        self.models['content_based'] = model
        
    def _train_clustering_model(self, user_features, interactions_df):
        """Train clustering model"""
        logger.info("Training clustering model...")
        
        model = ClusteringModel(n_clusters=self.config.N_CLUSTERS)
        model.fit(user_features, interactions_df)
        self.models['clustering'] = model
        
    def _train_hybrid_model(self):
        """Train hybrid model combining all approaches"""
        logger.info("Training hybrid model...")
        
        hybrid_model = HybridModel(weights=self.config.HYBRID_WEIGHTS)
        
        # Add trained models to hybrid system
        for name, model in self.models.items():
            hybrid_model.add_model(name, model)
            
        self.models['hybrid'] = hybrid_model
        
    def _train_pricing_model(self, pricing_data):
        """Train dynamic pricing model"""
        logger.info("Training pricing model...")
        
        model = DynamicPricingModel()
        model.fit(pricing_data)
        self.models['pricing'] = model
        
    def _generate_sample_pricing_data(self, items_df, interactions_df):
        """Generate sample pricing data for training"""
        np.random.seed(42)
        
        pricing_data = []
        
        for _, item in items_df.iterrows():
            item_id = item['item_id']
            base_price = item.get('price', np.random.uniform(10, 100))
            
            # Generate multiple pricing scenarios
            for _ in range(10):
                # Random context variables
                user_interest = np.random.uniform(0.1, 0.9)
                stock_level = np.random.randint(1, 100)
                demand_score = np.random.uniform(0.2, 0.8)
                competitor_price = base_price * np.random.uniform(0.8, 1.2)
                
                # Simple pricing logic for training
                price_multiplier = 1.0
                if user_interest > 0.7:
                    price_multiplier *= 1.1  # Higher price for high interest
                if stock_level < 20:
                    price_multiplier *= 1.05  # Higher price for low stock
                if demand_score > 0.6:
                    price_multiplier *= 1.08  # Higher price for high demand
                    
                final_price = base_price * price_multiplier
                
                pricing_data.append({
                    'item_id': item_id,
                    'base_price': base_price,
                    'user_interest_score': user_interest,
                    'stock_level': stock_level,
                    'demand_score': demand_score,
                    'competitor_price': competitor_price,
                    'final_price': final_price,
                    'hour_of_day': np.random.randint(0, 24),
                    'day_of_week': np.random.randint(0, 7),
                    'category_popularity': np.random.uniform(0.3, 0.9)
                })
                
        return pd.DataFrame(pricing_data)
        
    def _save_models(self):
        """Save trained models to disk"""
        for name, model in self.models.items():
            model_path = getattr(self.config, f'{name.upper()}_MODEL_PATH')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {name} model to {model_path}")
            
    def load_models(self):
        """Load trained models from disk"""
        for name in ['collaborative', 'content_based', 'clustering', 'hybrid', 'pricing']:
            model_path = getattr(self.config, f'{name.upper()}_MODEL_PATH')
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                logger.info(f"Loaded {name} model from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")
                
        return len(self.models) > 0 