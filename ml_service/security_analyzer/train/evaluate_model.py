import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score
import time
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime

from ..model.classifier import SecurityAnalyzer
from ..data.synthetic_generator import SyntheticDataGenerator
from ..utils.schema import SecurityAnalysisRequest
from ..utils.logger import setup_logger
from .evaluation import ModelEvaluator
from .visualization import SecurityVisualization

class SecurityModelEvaluator:
    def __init__(self):
        self.logger = setup_logger('SecurityModelEvaluator')
        self.analyzer = SecurityAnalyzer()
        self.generator = SyntheticDataGenerator()
        self.evaluator = ModelEvaluator()
        self.visualizer = SecurityVisualization()
    
    def evaluate_model_robustness(self, test_size: int = 2000) -> Dict[str, Any]:
        
        self.logger.info("Starting robustness evaluation...")
        
        test_scenarios = self._generate_robustness_scenarios(test_size)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_scenarios': len(test_scenarios),
            'scenario_results': {},
            'overall_robustness': {}
        }
        
        for scenario_name, scenario_data in test_scenarios.items():
            self.logger.info(f"Evaluating scenario: {scenario_name}")
            
            scenario_results = self._evaluate_scenario(scenario_data)
            results['scenario_results'][scenario_name] = scenario_results
        
        results['overall_robustness'] = self._calculate_robustness_metrics(
            results['scenario_results']
        )
        
        return results
    
    def _generate_robustness_scenarios(self, total_size: int) -> Dict[str, List[Dict]]:
        
        scenarios = {}
        scenario_size = total_size // 6
        
        scenarios['standard'] = self.generator.generate_evaluation_set(scenario_size)
        
        scenarios['obfuscated'] = self._generate_obfuscated_attacks(scenario_size)
        
        scenarios['edge_cases'] = self._generate_edge_cases(scenario_size)
        
        scenarios['benign_heavy'] = self._generate_benign_heavy_dataset(scenario_size)
        
        scenarios['mixed_complexity'] = self._generate_mixed_complexity(scenario_size)
        
        scenarios['realistic'] = self._generate_realistic_scenarios(scenario_size)
        
        return scenarios
    
    def _generate_obfuscated_attacks(self, size: int) -> List[Dict]:
        obfuscated_data = []
        
        url_encoded_payloads = [
            "%3Cscript%3Ealert%28%27XSS%27%29%3C%2Fscript%3E",  # <script>alert('XSS')</script>
            "%27%20OR%20%271%27%3D%271",  # ' OR '1'='1
            "%22%3E%3Cimg%20src%3Dx%20onerror%3Dalert%281%29%3E"  # "><img src=x onerror=alert(1)>
        ]
        
        base64_payloads = [
            "PHNjcmlwdD5hbGVydCgnWFNTJyk8L3NjcmlwdD4=",  # <script>alert('XSS')</script>
            "JyBPUiAnMSc9JzE="  # ' OR '1'='1
        ]
        
        unicode_payloads = [
            "\\u003cscript\\u003ealert('XSS')\\u003c/script\\u003e",
            "\\u0027 OR \\u00271\\u0027=\\u00271"
        ]
        
        attack_types = ['xss', 'sqli', 'csrf']
        all_payloads = url_encoded_payloads + base64_payloads + unicode_payloads
        
        for i in range(size):
            attack_type = attack_types[i % len(attack_types)]
            payload = all_payloads[i % len(all_payloads)]
            
            sample = {
                'method': 'POST',
                'path': f'/api/search',
                'query_params': f'q={payload}',
                'body': {'comment': payload},
                'headers': {'User-Agent': 'Mozilla/5.0'},
                'ip_address': self.generator.fake.ipv4(),
                'attack_type': attack_type
            }
            
            obfuscated_data.append(sample)
        
        return obfuscated_data
    
    def _generate_edge_cases(self, size: int) -> List[Dict]:
        edge_cases = []
        
        edge_patterns = [
            ('', 'benign'),  # Empty payload
            ('a' * 10000, 'benign'),  # Very long benign payload
            ('SELECT * FROM users WHERE id = 1', 'benign'),  # Benign SQL
            ('<b>Bold text</b>', 'benign'),  # Benign HTML
            ('javascript:void(0)', 'xss'),  # Edge XSS
            ("'; SELECT 1; --", 'sqli'),  # Minimal SQLi
        ]
        
        for i in range(size):
            pattern, attack_type = edge_patterns[i % len(edge_patterns)]
            
            sample = {
                'method': 'GET' if i % 2 == 0 else 'POST',
                'path': '/api/test',
                'query_params': pattern,
                'body': {'data': pattern},
                'headers': {},
                'ip_address': self.generator.fake.ipv4(),
                'attack_type': attack_type
            }
            
            edge_cases.append(sample)
        
        return edge_cases
    
    def _generate_benign_heavy_dataset(self, size: int) -> List[Dict]:
        benign_heavy = []
        
        benign_count = int(size * 0.9)
        attack_count = size - benign_count
        
        for i in range(benign_count):
            sample = self.generator.generate_request('benign')
            benign_heavy.append(sample)
        
        attack_types = ['xss', 'sqli', 'csrf']
        for i in range(attack_count):
            attack_type = attack_types[i % len(attack_types)]
            sample = self.generator.generate_request(attack_type)
            benign_heavy.append(sample)
        
        return benign_heavy
    
    def _generate_mixed_complexity(self, size: int) -> List[Dict]:
        mixed_data = []
        
        simple_count = int(size * 0.3)
        for i in range(simple_count):
            sample = self.generator.generate_request('xss')
            sample['query_params'] = '<script>alert(1)</script>'
            mixed_data.append(sample)
        
        medium_count = int(size * 0.4)
        for i in range(medium_count):
            sample = self.generator.generate_request('sqli')
            mixed_data.append(sample)
        
        complex_count = size - simple_count - medium_count
        for i in range(complex_count):
            sample = self.generator.generate_request('csrf')
            sample['headers']['X-Forwarded-For'] = self.generator.fake.ipv4()
            sample['headers']['X-Real-IP'] = self.generator.fake.ipv4()
            mixed_data.append(sample)
        
        return mixed_data
    
    def _generate_realistic_scenarios(self, size: int) -> List[Dict]:
        realistic_data = []
        
        realistic_paths = [
            '/login', '/register', '/profile', '/search', '/comment',
            '/admin/users', '/api/v1/data', '/upload', '/contact'
        ]
        
        for i in range(size):
            if i % 10 < 7:
                attack_type = 'benign'
            elif i % 10 < 8:
                attack_type = 'xss'
            elif i % 10 < 9:
                attack_type = 'sqli'
            else:
                attack_type = 'csrf'
            
            sample = self.generator.generate_request(attack_type)
            sample['path'] = realistic_paths[i % len(realistic_paths)]
            
            sample['user_agent'] = self.generator.fake.user_agent()
            
            realistic_data.append(sample)
        
        return realistic_data
    
    def _evaluate_scenario(self, scenario_data: List[Dict]) -> Dict[str, Any]:
        y_true = []
        y_pred = []
        response_times = []
        confidence_scores = []
        
        for sample in scenario_data:
            request = SecurityAnalysisRequest(**sample)
            
            start_time = time.time()
            result = self.analyzer.analyze(request)
            end_time = time.time()
            
            y_true.append(sample['attack_type'])
            y_pred.append(result.attack_type)
            response_times.append(end_time - start_time)
            confidence_scores.append(result.confidence)
        
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        
        true_attacks = sum(1 for t in y_true if t != 'benign')
        detected_attacks = sum(1 for t, p in zip(y_true, y_pred) if t != 'benign' and p != 'benign')
        
        attack_recall = detected_attacks / max(1, true_attacks)
        
        true_benign = sum(1 for t in y_true if t == 'benign')
        false_positives = sum(1 for t, p in zip(y_true, y_pred) if t == 'benign' and p != 'benign')
        
        fpr = false_positives / max(1, true_benign)
        
        return {
            'accuracy': accuracy,
            'attack_recall': attack_recall,
            'false_positive_rate': fpr,
            'avg_response_time': np.mean(response_times),
            'avg_confidence': np.mean(confidence_scores),
            'samples_evaluated': len(scenario_data)
        }
    
    def _calculate_robustness_metrics(self, scenario_results: Dict[str, Dict]) -> Dict[str, Any]:
        
        accuracies = [result['accuracy'] for result in scenario_results.values()]
        attack_recalls = [result['attack_recall'] for result in scenario_results.values()]
        fprs = [result['false_positive_rate'] for result in scenario_results.values()]
        
        return {
            'avg_accuracy': np.mean(accuracies),
            'min_accuracy': min(accuracies),
            'accuracy_std': np.std(accuracies),
            'avg_attack_recall': np.mean(attack_recalls),
            'min_attack_recall': min(attack_recalls),
            'avg_false_positive_rate': np.mean(fprs),
            'max_false_positive_rate': max(fprs),
            'robustness_score': self._calculate_robustness_score(scenario_results)
        }
    
    def _calculate_robustness_score(self, scenario_results: Dict[str, Dict]) -> float:
        weights = {
            'standard': 0.3,
            'obfuscated': 0.2,
            'edge_cases': 0.15,
            'benign_heavy': 0.1,
            'mixed_complexity': 0.15,
            'realistic': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for scenario, results in scenario_results.items():
            if scenario in weights:
                weight = weights[scenario]
                scenario_score = (results['accuracy'] + results['attack_recall']) / 2
                scenario_score -= results['false_positive_rate'] * 0.5
                
                weighted_score += weight * max(0, scenario_score)
                total_weight += weight
        
        return weighted_score / max(total_weight, 1.0)
    
    def cross_validate_model(self, cv_folds: int = 5, sample_size: int = 5000) -> Dict[str, Any]:
        self.logger.info(f"Starting {cv_folds}-fold cross-validation...")
        
        dataset = self.generator.generate_dataset(sample_size)
        
        texts = []
        labels = []
        
        for sample in dataset:
            text = self.analyzer.preprocessor.preprocess(sample)
            texts.append(text)
            labels.append(sample['attack_type'])
        
        X = self.analyzer.vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        if hasattr(self.analyzer.classifier, 'ensemble_classifier'):
            model = self.analyzer.classifier.ensemble_classifier
        else:
            model = self.analyzer.simple_classifier
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_scores': cv_scores.tolist(),
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'cv_folds': cv_folds,
            'sample_size': sample_size
        }
    
    def benchmark_against_baselines(self) -> Dict[str, Any]:
        self.logger.info("Benchmarking against baseline models...")
        
        test_data = self.generator.generate_evaluation_set(1000)
        
        our_results = []
        for sample in test_data:
            request = SecurityAnalysisRequest(**sample)
            result = self.analyzer.analyze(request)
            our_results.append({
                'true': sample['attack_type'],
                'pred': result.attack_type,
                'confidence': result.confidence
            })
        
        our_accuracy = sum(1 for r in our_results if r['true'] == r['pred']) / len(our_results)
        
        baselines = {
            'random': 0.25,
            'always_benign': sum(1 for sample in test_data if sample['attack_type'] == 'benign') / len(test_data),
            'pattern_matching_only': self._evaluate_pattern_matching_only(test_data)
        }
        
        return {
            'our_model_accuracy': our_accuracy,
            'baselines': baselines,
            'improvement_over_random': our_accuracy - baselines['random'],
            'improvement_over_always_benign': our_accuracy - baselines['always_benign'],
            'improvement_over_patterns_only': our_accuracy - baselines['pattern_matching_only']
        }
    
    def _evaluate_pattern_matching_only(self, test_data: List[Dict]) -> float:
        correct = 0
        total = len(test_data)
        
        for sample in test_data:
            text = self.analyzer.preprocessor.preprocess(sample)
            pattern_type, _ = self.analyzer.pattern_matcher.analyze(text, sample)
            
            if pattern_type == sample['attack_type']:
                correct += 1
        
        return correct / total

    def run_cross_validation(self, texts, labels, cv_folds=3):
        """Run cross-validation on the model"""
        try:
            self.logger.info(f"Starting {cv_folds}-fold cross-validation...")
            
            # Ensure we have a trained classifier
            if not self.analyzer.enhanced_classifier or not self.analyzer.enhanced_classifier.is_trained:
                self.logger.error("Enhanced classifier not available for cross-validation")
                return {}
            
            # Get the ensemble model for cross-validation
            ensemble = self.analyzer.enhanced_classifier.ensemble
            vectorizer = self.analyzer.enhanced_classifier.vectorizer
            
            if ensemble is None or vectorizer is None:
                self.logger.error("Model components not available for cross-validation")
                return {}
            
            # Transform texts
            X = vectorizer.transform(texts)
            y = self.analyzer.enhanced_classifier.label_encoder.transform(labels)
            
            # Run cross-validation
            cv_scores = cross_val_score(ensemble, X, y, cv=cv_folds, scoring='accuracy')
            
            return {
                'cv_mean': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores)),
                'cv_scores': cv_scores.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            return {} 