import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import datetime
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from collections import Counter

from ..config import CLASSIFIER_PATH, ATTACK_TYPES
from ..utils.logger import setup_logger

class EnhancedSecurityClassifier:
    def __init__(self):
        """Initialize the enhanced security classifier"""
        self.logger = setup_logger('EnhancedSecurityClassifier')
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        
        self.label_encoder = LabelEncoder()
        self.ensemble = None
        self.ensemble_weights = {}
        self.training_metadata = {}
        self.is_trained = False
        
        # Set model path - ensure consistency
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        self.model_path = os.path.join(models_dir, 'security_classifier_enhanced.pkl')
        
        # Try to load existing model
        if os.path.exists(self.model_path):
            self.load_model()
    
    def create_ensemble(self):
        rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        lr_classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='liblinear'
        )
        
        nb_classifier = MultinomialNB(alpha=0.1)
        
        self.ensemble_classifier = VotingClassifier(
            estimators=[
                ('rf', rf_classifier),
                ('lr', lr_classifier),
                ('nb', nb_classifier)
            ],
            voting='soft'
        )
        
        self.individual_classifiers = {
            'random_forest': rf_classifier,
            'logistic_regression': lr_classifier,
            'naive_bayes': nb_classifier
        }
    
    def train(self, texts, labels):
        """Train the enhanced security classifier"""
        try:
            self.logger.info(f"Training enhanced classifier with {len(texts)} samples")
            
            # Ensure we have data
            if not texts or not labels:
                raise ValueError("Training data cannot be empty")
            
            # Debug: Check label distribution
            label_counts = Counter(labels)
            self.logger.info(f"Label distribution: {dict(label_counts)}")
            
            # Ensure minimum samples per class
            min_samples = min(label_counts.values())
            if min_samples < 10:
                self.logger.warning(f"Low sample count for some classes: {dict(label_counts)}")
            
            # Fit the vectorizer and transform texts
            self.logger.info("Fitting TF-IDF vectorizer...")
            X = self.vectorizer.fit_transform(texts)
            self.logger.info(f"Vectorizer fitted with {X.shape[1]} features, {X.shape[0]} samples")
            
            # Check if features were extracted
            if X.shape[1] == 0:
                raise ValueError("No features extracted from texts")
            
            # Encode labels
            y = self.label_encoder.fit_transform(labels)
            self.logger.info(f"Label encoder fitted with classes: {self.label_encoder.classes_}")
            
            # Check if we have multiple classes
            if len(self.label_encoder.classes_) < 2:
                raise ValueError("Need at least 2 classes for training")
            
            # Split data with stratification
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError as e:
                self.logger.warning(f"Stratified split failed: {e}, using random split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # Train individual models with balanced class weights
            models = {}
            accuracies = {}
            
            # Random Forest with balanced weights
            rf = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1,
                class_weight='balanced',  # CRITICAL: Balance classes
                max_depth=10,
                min_samples_split=5
            )
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_accuracy = accuracy_score(y_test, rf_pred)
            models['random_forest'] = rf
            accuracies['random_forest'] = rf_accuracy
            self.logger.info(f"random_forest accuracy: {rf_accuracy:.4f}")
            
            # Logistic Regression with balanced weights
            lr = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced',  # CRITICAL: Balance classes
                solver='liblinear'
            )
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            lr_accuracy = accuracy_score(y_test, lr_pred)
            models['logistic_regression'] = lr
            accuracies['logistic_regression'] = lr_accuracy
            self.logger.info(f"logistic_regression accuracy: {lr_accuracy:.4f}")
            
            # Naive Bayes
            nb = MultinomialNB(alpha=1.0)
            nb.fit(X_train, y_train)
            nb_pred = nb.predict(X_test)
            nb_accuracy = accuracy_score(y_test, nb_pred)
            models['naive_bayes'] = nb
            accuracies['naive_bayes'] = nb_accuracy
            self.logger.info(f"naive_bayes accuracy: {nb_accuracy:.4f}")
            
            # Create voting ensemble with balanced voting
            estimators = [(name, model) for name, model in models.items()]
            self.ensemble = VotingClassifier(estimators=estimators, voting='soft')
            self.ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            ensemble_pred = self.ensemble.predict(X_test)
            ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
            accuracies['ensemble'] = ensemble_accuracy
            self.logger.info(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
            
            # Detailed evaluation
            ensemble_report = classification_report(y_test, ensemble_pred, 
                                                  target_names=self.label_encoder.classes_,
                                                  output_dict=True, zero_division=0)
            self.logger.info(f"Ensemble classification report: {ensemble_report}")
            
            # Calculate ensemble improvement
            best_individual = max(rf_accuracy, lr_accuracy, nb_accuracy)
            ensemble_improvement = ensemble_accuracy - best_individual
            
            # Calculate ensemble weights
            weights = {
                'random_forest': 1/3,
                'logistic_regression': 1/3,
                'naive_bayes': 1/3,
                'ensemble_improvement': ensemble_improvement
            }
            
            self.ensemble_weights = weights
            self.logger.info(f"Ensemble weights: {weights}")
            
            # Store training metadata
            self.training_metadata = {
                'accuracies': accuracies,
                'ensemble_weights': weights,
                'feature_count': X.shape[1],
                'training_samples': len(texts),
                'classes': self.label_encoder.classes_.tolist(),
                'classification_report': ensemble_report
            }
            
            # Mark as trained
            self.is_trained = True
            
            # Save the model
            self.save_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.is_trained = False
            return False
    
    def predict_with_explanation(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Get ensemble predictions
        ensemble_pred = self.ensemble_classifier.predict(X)
        ensemble_proba = self.ensemble_classifier.predict_proba(X)
        
        # Get individual predictions
        individual_predictions = {}
        individual_probabilities = {}
        
        for name, classifier in self.individual_classifiers.items():
            individual_predictions[name] = classifier.predict(X)
            individual_probabilities[name] = classifier.predict_proba(X)
        
        return {
            'ensemble_prediction': ensemble_pred,
            'ensemble_probabilities': ensemble_proba,
            'individual_predictions': individual_predictions,
            'individual_probabilities': individual_probabilities,
            'ensemble_weights': self.ensemble_weights
        }
    
    def get_model_performance(self):
        if not self.training_history:
            return {"error": "No training history available"}
        
        latest_training = self.training_history[-1]
        
        return {
            'latest_training': latest_training,
            'training_history_count': len(self.training_history),
            'ensemble_weights': self.ensemble_weights,
            'is_trained': self.is_trained,
            'model_type': 'VotingClassifier (RF + LR + NB)'
        }
    
    def save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'ensemble': self.ensemble,
                'ensemble_weights': self.ensemble_weights,
                'training_metadata': self.training_metadata
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Enhanced model saved to {self.model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self):
        """Load a pre-trained model"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.vectorizer = model_data['vectorizer']
                self.label_encoder = model_data['label_encoder']
                self.ensemble = model_data['ensemble']
                self.ensemble_weights = model_data.get('ensemble_weights', {})
                self.training_metadata = model_data.get('training_metadata', {})
                
                # Verify vectorizer is properly fitted
                if hasattr(self.vectorizer, 'vocabulary_') and self.vectorizer.vocabulary_:
                    self.is_trained = True
                    self.logger.info(f"Enhanced model loaded from {self.model_path}")
                    return True
                else:
                    self.logger.error("Loaded vectorizer is not properly fitted")
                    self.is_trained = False
                    return False
            else:
                self.logger.warning(f"Model file not found: {self.model_path}")
                self.is_trained = False
                return False
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.is_trained = False
            return False
    
    def predict(self, texts):
        """Make predictions on text samples"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Check if vectorizer is properly fitted
            if not hasattr(self.vectorizer, 'vocabulary_') or self.vectorizer.vocabulary_ is None:
                self.logger.error("Vectorizer not fitted properly")
                # Try to reload the model
                if self.load_model():
                    self.logger.info("Model reloaded successfully")
                else:
                    return [{'prediction': 'benign', 'confidence': 0.3} for _ in texts]
            
            # Transform texts using fitted vectorizer
            X = self.vectorizer.transform(texts)
            
            # Ensure ensemble is available
            if not hasattr(self, 'ensemble') or self.ensemble is None:
                self.logger.error("Ensemble model not available")
                return [{'prediction': 'benign', 'confidence': 0.3} for _ in texts]
            
            # Get ensemble prediction
            predictions = self.ensemble.predict(X)
            probabilities = self.ensemble.predict_proba(X)
            
            results = []
            for i, (pred_encoded, proba) in enumerate(zip(predictions, probabilities)):
                # Decode prediction
                pred = self.label_encoder.inverse_transform([pred_encoded])[0]
                confidence = float(np.max(proba))
                
                # Create probability dict
                prob_dict = {}
                for j, class_name in enumerate(self.label_encoder.classes_):
                    prob_dict[class_name] = float(proba[j])
                
                results.append({
                    'prediction': pred,
                    'confidence': confidence,
                    'probabilities': prob_dict
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return [{'prediction': 'benign', 'confidence': 0.3} for _ in texts] 