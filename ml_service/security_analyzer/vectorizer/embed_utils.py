import numpy as np
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import pickle
import os
from ..utils.logger import setup_logger
from ..config import ML_MODEL_PATH

class EmbeddingExtractor:
    def __init__(self, model_type: str = "sentence-transformer"):
        self.logger = setup_logger('EmbeddingExtractor')
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.embedding_dim = 0
        self._load_model()
    
    def _load_model(self):
        try:
            if self.model_type == "sentence-transformer":
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = 384
                self.logger.info("Loaded SentenceTransformer model")
            
            elif self.model_type == "bert":
                model_name = "bert-base-uncased"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.embedding_dim = 768
                self.logger.info("Loaded BERT model")
            
            elif self.model_type == "distilbert":
                model_name = "distilbert-base-uncased"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.embedding_dim = 768
                self.logger.info("Loaded DistilBERT model")
                
        except Exception as e:
            self.logger.error(f"Failed to load {self.model_type} model: {e}")
            self.model = None
    
    def extract_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            if self.model_type == "sentence-transformer":
                return self._extract_sentence_transformer_embeddings(texts, batch_size)
            elif self.model_type in ["bert", "distilbert"]:
                return self._extract_transformer_embeddings(texts, batch_size)
        except Exception as e:
            self.logger.error(f"Embedding extraction failed: {e}")
            raise
    
    def _extract_sentence_transformer_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def _extract_transformer_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        embeddings = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(device)
                
                outputs = self.model(**inputs)
                
                if hasattr(outputs, 'last_hidden_state'):
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                else:
                    batch_embeddings = outputs.pooler_output
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def save_embeddings(self, embeddings: np.ndarray, texts: List[str], filename: str):
        embedding_data = {
            'embeddings': embeddings,
            'texts': texts,
            'model_type': self.model_type,
            'embedding_dim': self.embedding_dim
        }
        
        filepath = os.path.join(ML_MODEL_PATH, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        self.logger.info(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filename: str) -> Dict[str, Any]:
        filepath = os.path.join(ML_MODEL_PATH, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Embedding file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            embedding_data = pickle.load(f)
        
        self.logger.info(f"Loaded embeddings from {filepath}")
        return embedding_data

class HybridVectorizer:
    
    def __init__(self, tfidf_vectorizer, embedding_extractor: EmbeddingExtractor):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.embedding_extractor = embedding_extractor
        self.logger = setup_logger('HybridVectorizer')
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        
        if self.embedding_extractor.model is not None:
            embeddings = self.embedding_extractor.extract_embeddings(texts)
            
            combined_features = np.hstack([tfidf_features.toarray(), embeddings])
            self.logger.info(f"Combined features shape: {combined_features.shape}")
            return combined_features
        else:
            self.logger.warning("Embedding extractor not available, using TF-IDF only")
            return tfidf_features.toarray()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        if self.embedding_extractor.model is not None:
            embeddings = self.embedding_extractor.extract_embeddings(texts)
            
            combined_features = np.hstack([tfidf_features.toarray(), embeddings])
            return combined_features
        else:
            return tfidf_features.toarray() 