import pickle
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime
from collections import Counter

from ..config import CLASSIFIER_PATH, ATTACK_TYPES
from ..data.preprocessing import RequestPreprocessor
from ..vectorizer.tfidf_vectorizer import SecurityTfidfVectorizer
from ..pattern_matching.matcher import PatternMatcher
from ..model.scoring import DangerScorer
from ..model.enhanced_classifier import EnhancedSecurityClassifier
from ..model.explainability import SecurityExplainer
from ..database.logger import SecurityDatabaseLogger
from ..utils.schema import SecurityAnalysisRequest, SecurityAnalysisResponse
from ..utils.logger import setup_logger
from ..pattern_matching.benign_detector import BenignDetector

class SecurityAnalyzer:
    def __init__(self):
        """Initialize the Security Analyzer"""
        self.logger = setup_logger('SecurityAnalyzer')
        
        # Initialize components
        self.preprocessor = RequestPreprocessor()
        self.pattern_matcher = PatternMatcher()
        self.benign_detector = BenignDetector()  # Add benign detector
        
        # Initialize enhanced classifier
        try:
            self.enhanced_classifier = EnhancedSecurityClassifier()
            # Try to load existing model
            if self.enhanced_classifier.load_model():
                self.logger.info("Pre-trained enhanced classifier loaded")
            else:
                self.logger.info("No pre-trained model found, will train on first use")
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced classifier: {e}")
            self.enhanced_classifier = None
        
        # Initialize explainability module AFTER enhanced classifier
        try:
            if self.enhanced_classifier and self.enhanced_classifier.is_trained:
                self.explainer = SecurityExplainer(
                    classifier=self.enhanced_classifier.ensemble,
                    vectorizer=self.enhanced_classifier.vectorizer,
                    preprocessor=self.preprocessor
                )
                self.logger.info("Explainability module initialized")
            else:
                self.explainer = None
                self.logger.info("Explainability module will be initialized after training")
        except Exception as e:
            self.logger.warning(f"Explainability module not available: {e}")
            self.explainer = None
        
        # Initialize database logger
        try:
            self.db_logger = SecurityDatabaseLogger()
        except Exception as e:
            self.logger.warning(f"Database logger not available: {e}")
            self.db_logger = None
        
        self._is_trained = False
    
    def train(self, training_data):
        """Train the security analyzer with provided data"""
        try:
            self.logger.info(f"Training SecurityAnalyzer with {len(training_data)} samples")
            
            # Prepare training data
            texts = []
            labels = []
            
            for sample in training_data:
                if hasattr(sample, 'dict'):
                    sample_dict = sample.dict()
                else:
                    sample_dict = sample
                    
                # Preprocess the sample
                preprocessed_text = self.preprocessor.preprocess(sample_dict)
                texts.append(preprocessed_text)
                labels.append(sample_dict.get('attack_type', 'benign'))
            
            # Validate training data balance
            label_counts = Counter(labels)
            self.logger.info(f"Training data distribution: {dict(label_counts)}")
            
            # Check for severely imbalanced data
            total_samples = len(labels)
            min_class_ratio = min(label_counts.values()) / total_samples
            if min_class_ratio < 0.1:  # Less than 10% of any class
                self.logger.warning(f"Severely imbalanced training data: {dict(label_counts)}")
            
            # Train the enhanced classifier
            if self.enhanced_classifier:
                training_success = self.enhanced_classifier.train(texts, labels)
                if training_success:
                    self.logger.info("Enhanced classifier training completed")
                    self._is_trained = True
                    
                    # Validate model performance
                    self._validate_model_performance(texts[:100], labels[:100])
                    
                    # Initialize explainability module after training
                    try:
                        self.explainer = SecurityExplainer(
                            classifier=self.enhanced_classifier.ensemble,
                            vectorizer=self.enhanced_classifier.vectorizer,
                            preprocessor=self.preprocessor
                        )
                        self.logger.info("Explainability module initialized after training")
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize explainability after training: {e}")
                        self.explainer = None
                else:
                    self.logger.error("Enhanced classifier training failed")
                    return False
            else:
                # Initialize enhanced classifier if not available
                try:
                    self.enhanced_classifier = EnhancedSecurityClassifier()
                    training_success = self.enhanced_classifier.train(texts, labels)
                    if training_success:
                        self.logger.info("Enhanced classifier created and trained")
                        self._is_trained = True
                    else:
                        self.logger.error("Enhanced classifier training failed")
                        return False
                except Exception as e:
                    self.logger.error(f"Failed to create and train enhanced classifier: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False

    def _validate_model_performance(self, texts, labels):
        """Validate that the model is actually learning"""
        try:
            predictions = []
            for text in texts:
                result = self.enhanced_classifier.predict([text])
                if result and len(result) > 0:
                    predictions.append(result[0]['prediction'])
                else:
                    predictions.append('benign')
            
            # Check if model predicts variety of classes
            pred_counts = Counter(predictions)
            label_counts = Counter(labels)
            
            self.logger.info(f"Validation - True labels: {dict(label_counts)}")
            self.logger.info(f"Validation - Predictions: {dict(pred_counts)}")
            
            # Warning if model only predicts one class
            if len(pred_counts) == 1:
                self.logger.error("MODEL VALIDATION FAILED: Model only predicts one class!")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return False

    @property
    def is_trained(self):
        """Check if the model is trained"""
        if hasattr(self, '_is_trained') and self._is_trained:
            return True
        
        # Check if enhanced classifier is trained
        if self.enhanced_classifier and hasattr(self.enhanced_classifier, 'is_trained'):
            return self.enhanced_classifier.is_trained
        
        return False
    
    def _save_simple_model(self):
        os.makedirs(os.path.dirname(CLASSIFIER_PATH), exist_ok=True)
        with open(CLASSIFIER_PATH, 'wb') as f:
            pickle.dump(self.simple_classifier, f)
    
    def _calculate_attack_score(self, attack_type, confidence, pattern_matches):
        """Calculate attack danger score with improved scoring"""
        try:
            if attack_type == 'benign':
                return 0.0
            
            # Enhanced base scores for stronger detection
            base_scores = {
                'xss': 0.75,      # Increased from 0.6
                'sqli': 0.85,     # Increased from 0.7
                'csrf': 0.65,     # Increased from 0.5
            }
            
            base_score = base_scores.get(attack_type, 0.6)
            
            # Boost score based on pattern matches
            pattern_boost = min(0.2, len(pattern_matches) * 0.05)
            
            # Confidence multiplier
            confidence_multiplier = max(0.8, confidence)
            
            final_score = min(0.95, (base_score + pattern_boost) * confidence_multiplier)
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"Attack score calculation failed: {e}")
            return 0.6

    def analyze(self, request):
        """Analyze a security request using ensemble approach"""
        start_time = time.time()
        
        try:
            # Convert request to dict for processing
            if hasattr(request, 'dict'):
                request_dict = request.dict()
            else:
                request_dict = dict(request)
            
            # Quick benign check first - if clearly benign, skip expensive analysis
            if self.benign_detector.is_benign(request_dict):
                processing_time = (time.time() - start_time) * 1000
                
                return SecurityAnalysisResponse(
                    is_malicious=False,
                    attack_type="benign",
                    attack_score=0.0,
                    confidence=0.95,  # High confidence for clear benign patterns
                    details={"reason": "Clear benign pattern detected"},
                    processing_time_ms=processing_time
                )
            
            # Preprocess the request
            preprocessed_text = self.preprocessor.preprocess(request_dict)
            
            # Pattern matching analysis
            pattern_type, pattern_matches = self.pattern_matcher.analyze(
                preprocessed_text, request_dict
            )
            
            # Calculate pattern confidence based on matches
            if pattern_type != 'benign':
                pattern_confidence = min(0.9, 0.6 + (len(pattern_matches) * 0.1))
            else:
                pattern_confidence = 0.7  # Default confidence for benign
            
            # ML prediction
            ml_pred = 'benign'
            ml_confidence = 0.5
            
            try:
                if self.enhanced_classifier and self.enhanced_classifier.is_trained:
                    ml_result = self.enhanced_classifier.predict([preprocessed_text])
                    if ml_result and len(ml_result) > 0:
                        ml_pred = ml_result[0]['prediction']
                        ml_confidence = ml_result[0]['confidence']
                else:
                    self.logger.warning("Enhanced classifier not available, using pattern matching only")
            except Exception as e:
                self.logger.warning(f"ML prediction failed: {e}")
            
            # Ensemble decision
            final_attack_type, final_confidence, final_attack_score = self._ensemble_decision(
                ml_pred, ml_confidence, pattern_type, pattern_matches
            )
            
            # Calculate attack score
            attack_score = self._calculate_attack_score(
                final_attack_type, final_confidence, pattern_matches
            )
            
            # Determine if malicious
            is_malicious = final_attack_type != 'benign'
            
            processing_time = (time.time() - start_time) * 1000
            
            # Create response
            response = SecurityAnalysisResponse(
                is_malicious=is_malicious,
                attack_type=final_attack_type,
                attack_score=attack_score,
                confidence=final_confidence,
                suspected_attack_type=final_attack_type if is_malicious else None,
                details={
                    'pattern_matches': pattern_matches,
                    'ml_prediction': ml_pred,
                    'ml_confidence': ml_confidence,
                    'pattern_confidence': pattern_confidence,
                    'processing_time_ms': processing_time
                },
                processing_time_ms=processing_time
            )
            
            # Log the analysis - with proper error handling
            try:
                if hasattr(self, 'db_logger') and self.db_logger:
                    if hasattr(self.db_logger, 'log_analysis'):
                        self.db_logger.log_analysis(request_dict, response.dict())
                    elif hasattr(self.db_logger, 'log_request'):
                        self.db_logger.log_request(request_dict, response)
            except Exception as log_error:
                self.logger.warning(f"Failed to log analysis: {log_error}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return SecurityAnalysisResponse(
                is_malicious=False,
                attack_type="benign",
                attack_score=0.0,
                confidence=0.5,
                details={"error": str(e)},
                processing_time_ms=processing_time
            )
    
    def _ensemble_decision(self, ml_prediction, ml_confidence, pattern_type, pattern_matches):
        """Make ensemble decision with improved confidence handling"""
        try:
            # If benign detector already flagged as benign, trust it
            if hasattr(self, '_benign_detected') and self._benign_detected:
                return "benign", 0.0, 0.95
            
            # Pattern matching takes precedence if confidence is high
            if pattern_type != "benign" and len(pattern_matches) >= 2:
                # High confidence for multiple pattern matches
                confidence = min(0.95, 0.7 + (len(pattern_matches) * 0.1))
                attack_score = self._calculate_attack_score(pattern_type, confidence, pattern_matches)
                return pattern_type, attack_score, confidence
            
            # Single pattern match - medium confidence
            if pattern_type != "benign" and len(pattern_matches) == 1:
                confidence = 0.75
                attack_score = self._calculate_attack_score(pattern_type, confidence, pattern_matches)
                return pattern_type, attack_score, confidence
            
            # ML prediction with high confidence
            if ml_prediction != "benign" and ml_confidence > 0.8:
                attack_score = self._calculate_attack_score(ml_prediction, ml_confidence, [])
                return ml_prediction, attack_score, ml_confidence
            
            # ML prediction with medium confidence - require higher threshold
            if ml_prediction != "benign" and ml_confidence > 0.65:
                attack_score = self._calculate_attack_score(ml_prediction, ml_confidence, [])
                return ml_prediction, attack_score, ml_confidence * 0.9  # Slight confidence reduction
            
            # Default to benign if no strong indicators
            return "benign", 0.0, 0.8
            
        except Exception as e:
            self.logger.error(f"Ensemble decision failed: {e}")
            return "benign", 0.0, 0.5
    
    def get_model_info(self) -> dict:
        info = {
            'is_trained': self._is_trained,
            'attack_types': ATTACK_TYPES,
            'vectorizer_features': getattr(self.vectorizer.vectorizer, 'max_features', 'Not fitted'),
            'pattern_detectors': ['XSS', 'SQLi', 'CSRF'],
            'database_logging': self.db_logger.db_available,
            'explainability_available': self.explainer is not None
        }
        
        if hasattr(self.enhanced_classifier, 'get_model_performance'):
            performance_info = self.enhanced_classifier.get_model_performance()
            info.update(performance_info)
        
        return info
    
    def get_explanation(self, request: SecurityAnalysisRequest) -> dict:
        if not self.explainer:
            return {'error': 'Explainer not available'}
        
        request_dict = request.dict()
        return self.explainer.explain_prediction(request_dict)
    
    def get_dashboard_data(self, days: int = 7) -> dict:
        return self.db_logger.get_security_dashboard_data(days)

    @property
    def vectorizer(self):
        """Access to vectorizer through enhanced classifier"""
        if self.enhanced_classifier and hasattr(self.enhanced_classifier, 'vectorizer'):
            return self.enhanced_classifier.vectorizer
        return None

    @property
    def ensemble(self):
        """Access to ensemble through enhanced classifier"""
        if self.enhanced_classifier and hasattr(self.enhanced_classifier, 'ensemble'):
            return self.enhanced_classifier.ensemble
        return None 