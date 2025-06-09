import numpy as np
import pandas as pd
from typing import Dict, List, Any
import lime
from lime.lime_text import LimeTextExplainer
from ..utils.logger import setup_logger

class SecurityExplainer:
    def __init__(self, classifier, vectorizer, preprocessor):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.preprocessor = preprocessor
        self.logger = setup_logger('SecurityExplainer')
        
        try:
            self.explainer = LimeTextExplainer(
                class_names=['benign', 'xss', 'sqli', 'csrf']
            )
            self.logger.info("LIME explainer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LIME explainer: {e}")
            self.explainer = None
    
    def explain_prediction(self, request_data: Dict[str, Any], num_features: int = 10) -> Dict[str, Any]:
        try:
            preprocessed_text = self.preprocessor.preprocess(request_data)
            
            def predict_proba_func(texts):
                try:
                    X = self.vectorizer.transform(texts)
                    return self.classifier.predict_proba(X)
                except:
                    return np.array([[0.25, 0.25, 0.25, 0.25]] * len(texts))
            
            explanation = self.explainer.explain_instance(
                preprocessed_text,
                predict_proba_func,
                num_features=num_features,
                top_labels=4
            )
            
            feature_importance = {}
            for label in explanation.available_labels():
                features = explanation.as_list(label=label)
                feature_importance[f'class_{label}'] = [
                    {'feature': feat, 'importance': imp} for feat, imp in features
                ]
            
            prediction_proba = predict_proba_func([preprocessed_text])[0]
            
            return {
                'explanation_type': 'LIME',
                'feature_importance': feature_importance,
                'prediction_probabilities': {
                    'benign': float(prediction_proba[0]),
                    'csrf': float(prediction_proba[1]), 
                    'sqli': float(prediction_proba[2]),
                    'xss': float(prediction_proba[3])
                },
                'top_contributing_features': self._get_top_features(feature_importance),
                'preprocessed_input': preprocessed_text
            }
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            return {'error': f'Explanation failed: {str(e)}'}
    
    def _get_top_features(self, feature_importance: Dict) -> List[Dict]:
        all_features = []
        
        for class_name, features in feature_importance.items():
            for feat_info in features:
                all_features.append({
                    'feature': feat_info['feature'],
                    'importance': abs(feat_info['importance']),
                    'class': class_name,
                    'direction': 'positive' if feat_info['importance'] > 0 else 'negative'
                })
        
        all_features.sort(key=lambda x: x['importance'], reverse=True)
        return all_features[:10]
    
    def get_feature_statistics(self, dataset: List[Dict]) -> Dict[str, Any]:
        feature_stats = {}
        
        try:
            if hasattr(self.classifier, 'feature_importances_'):
                feature_names = self.vectorizer.vectorizer.get_feature_names_out()
                importances = self.classifier.feature_importances_
                
                top_indices = np.argsort(importances)[::-1][:20]
                
                feature_stats = {
                    'top_features': [
                        {
                            'feature': feature_names[i],
                            'importance': float(importances[i]),
                            'rank': rank + 1
                        }
                        for rank, i in enumerate(top_indices)
                    ],
                    'total_features': len(feature_names),
                    'feature_distribution': {
                        'high_importance': len([x for x in importances if x > 0.01]),
                        'medium_importance': len([x for x in importances if 0.001 < x <= 0.01]),
                        'low_importance': len([x for x in importances if x <= 0.001])
                    }
                }
        
        except Exception as e:
            self.logger.error(f"Feature statistics generation failed: {e}")
            feature_stats['error'] = str(e)
        
        return feature_stats 