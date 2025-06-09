import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from ..config import VECTORIZER_PATH, MAX_FEATURES, MIN_DF, MAX_DF

class SecurityTfidfVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        self.is_fitted = False
    
    def _load_or_create(self):
        if os.path.exists(VECTORIZER_PATH):
            with open(VECTORIZER_PATH, 'rb') as f:
                self.vectorizer = pickle.load(f)
        else:
            self.vectorizer = TfidfVectorizer(
                max_features=MAX_FEATURES,
                min_df=MIN_DF,
                max_df=MAX_DF,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)
            )
    
    def fit_transform(self, texts):
        result = self.vectorizer.fit_transform(texts)
        self.save()
        return result
    
    def transform(self, texts):
        if hasattr(self.vectorizer, 'vocabulary_'):
            return self.vectorizer.transform(texts)
        else:
            raise ValueError("Vectorizer not fitted yet")
    
    def save(self):
        os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(self.vectorizer, f) 