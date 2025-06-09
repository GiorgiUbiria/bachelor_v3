USER_SEGMENTS = {
    0: "Tech Enthusiasts",
    1: "Fashion Lovers", 
    2: "Health & Fitness",
    3: "Home & Garden",
    4: "Book Readers"
}

PRODUCT_CATEGORIES = [
    "Electronics", "Fashion", "Books", "Home", "Sports",
    "Beauty", "Automotive", "Toys", "Grocery", "Health"
]

RECOMMENDATION_METHODS = [
    "collaborative_filtering",
    "content_based",
    "clustering",
    "hybrid"
]

EVALUATION_METRICS = [
    "precision_at_k",
    "recall_at_k", 
    "ndcg_at_k",
    "map_at_k",
    "coverage",
    "diversity"
]

REGIONAL_PREFERENCES = {
    "EU": ["electronics", "books", "fashion"],
    "NA": ["electronics", "sports", "home"],
    "ASIA": ["fashion", "beauty", "electronics"]
}

AGE_PREFERENCES = {
    "young": ["gaming", "fashion", "electronics"],
    "middle": ["home", "books", "sports"],
    "senior": ["health", "books", "home"]
}

PRICING_STRATEGIES = {
    "competitive": {"base_multiplier": 0.95},
    "premium": {"base_multiplier": 1.15},
    "value": {"base_multiplier": 0.85}
} 