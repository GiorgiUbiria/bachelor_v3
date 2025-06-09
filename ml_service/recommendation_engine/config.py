import os
from pathlib import Path

class RecommendationConfig:
    MODEL_DIR = Path(__file__).parent / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    
    COLLABORATIVE_MODEL_PATH = MODEL_DIR / "collaborative_model.pkl"
    CONTENT_MODEL_PATH = MODEL_DIR / "content_model.pkl"
    CLUSTERING_MODEL_PATH = MODEL_DIR / "clustering_model.pkl"
    HYBRID_MODEL_PATH = MODEL_DIR / "hybrid_model.pkl"
    PRICING_MODEL_PATH = MODEL_DIR / "pricing_model.pkl"
    
    SVD_COMPONENTS = 50
    MAX_FEATURES_TFIDF = 5000
    N_CLUSTERS = 5
    
    DEFAULT_RECOMMENDATIONS = 10
    MAX_RECOMMENDATIONS = 50
    MIN_INTERACTIONS_FOR_CF = 5
    
    HYBRID_WEIGHTS = {
        'collaborative': 0.4,
        'content_based': 0.3,
        'clustering': 0.3
    }
    
    MIN_DISCOUNT_PERCENTAGE = 5.0
    MAX_DISCOUNT_PERCENTAGE = 50.0
    PRICE_ADJUSTMENT_BOUNDS = (0.5, 1.5)  # 50% to 150% of base price
    
    SIMILARITY_THRESHOLD = 0.1
    CONFIDENCE_THRESHOLD = 0.5
    
    CACHE_TTL = 3600  # 1 hour
    
    TRAIN_TEST_SPLIT = 0.8
    RANDOM_STATE = 42 