import os
from pathlib import Path

# Paths configuration
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

CLASSIFIER_PATH = MODEL_DIR / "security_classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
PATTERN_RULES_PATH = MODEL_DIR / "pattern_rules.json"

# Attack types configuration
ATTACK_TYPES = [
    "XSS",
    "SQL_INJECTION", 
    "CSRF",
    "COMMAND_INJECTION",
    "PATH_TRAVERSAL",
    "NORMAL"
]

# Model parameters
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 3)
MODEL_CONFIDENCE_THRESHOLD = 0.7

# Training parameters
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42

# API configuration
MAX_REQUEST_SIZE = 1024 * 1024  # 1MB
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

DANGER_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 0.8
}

ATTACK_SEVERITY = {
    'sqli': 1.0,
    'xss': 0.8,
    'csrf': 0.6,
    'benign': 0.0
}

MAX_FEATURES = 5000
MIN_DF = 2
MAX_DF = 0.8

# Add database configuration with fallback
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///security_test.db')
USE_DATABASE_LOGGING = os.getenv('USE_DATABASE_LOGGING', 'false').lower() == 'true'

# For testing, disable database logging by default
ENABLE_DB_LOGGING = os.getenv('ENABLE_DB_LOGGING', 'false').lower() == 'true' 