import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime

from ..utils.logger import setup_logger
from ..train.visualization import SecurityVisualization

class ModelEvaluator:
    def __init__(self):
        self.logger = setup_logger('ModelEvaluator')
        self.visualizer = SecurityVisualization()
        self.evaluation_history = []
    
    def comprehensive_evaluation(self, y_true: List[str], y_pred: List[str], 
                               y_prob: np.ndarray, labels: List[str],
                               dataset_info: Dict[str, Any] = None) -> Dict[str, Any]:
        
        evaluation_start = datetime.now()
        
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        
        class_report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        y_true_bin = label_binarize(y_true, classes=labels)
        if y_true_bin.shape[1] > 1:
            roc_auc = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
        else:
            roc_auc = 0.0
        
        per_class_metrics = self._calculate_per_class_metrics(y_true, y_pred, y_prob, labels)
        
        attack_analysis = self._analyze_attack_detection(y_true, y_pred, labels)
        
        error_analysis = self._analyze_errors(y_true, y_pred, labels)
        
        complexity_analysis = self._analyze_by_complexity(y_true, y_pred)
        
        evaluation_result = {
            'timestamp': evaluation_start.isoformat(),
            'dataset_info': dataset_info or {},
            'overall_metrics': {
                'accuracy': accuracy,
                'macro_precision': class_report['macro avg']['precision'],
                'macro_recall': class_report['macro avg']['recall'],
                'macro_f1': class_report['macro avg']['f1-score'],
                'weighted_precision': class_report['weighted avg']['precision'],
                'weighted_recall': class_report['weighted avg']['recall'],
                'weighted_f1': class_report['weighted avg']['f1-score'],
                'roc_auc_macro': roc_auc
            },
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_metrics,
            'attack_analysis': attack_analysis,
            'error_analysis': error_analysis,
            'complexity_analysis': complexity_analysis
        }
        
        self.evaluation_history.append(evaluation_result)
        self.logger.info(f"Comprehensive evaluation completed. Accuracy: {accuracy:.4f}")
        
        return evaluation_result
    
    def _calculate_per_class_metrics(self, y_true: List[str], y_pred: List[str],
                                   y_prob: np.ndarray, labels: List[str]) -> Dict[str, Any]:
        per_class = {}
        
        for i, label in enumerate(labels):
            true_binary = [1 if t == label else 0 for t in y_true]
            pred_binary = [1 if p == label else 0 for p in y_pred]
            
            if i < y_prob.shape[1]:
                prob_binary = y_prob[:, i]
                
                avg_precision = average_precision_score(true_binary, prob_binary)
                
                precision, recall, _ = precision_recall_curve(true_binary, prob_binary)
            else:
                avg_precision = 0.0
                precision, recall = [], []
            
            tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)
            tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
            fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)
            fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
            
            per_class[label] = {
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': tp / max(1, tp + fp),
                'recall': tp / max(1, tp + fn),
                'specificity': tn / max(1, tn + fp),
                'average_precision': avg_precision,
                'support': sum(true_binary)
            }
        
        return per_class
    
    def _analyze_attack_detection(self, y_true: List[str], y_pred: List[str], 
                                labels: List[str]) -> Dict[str, Any]:
        
        true_binary = [0 if t == 'benign' else 1 for t in y_true]
        pred_binary = [0 if p == 'benign' else 1 for p in y_pred]
        
        attack_tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
        attack_tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)
        attack_fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
        attack_fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)
        
        attack_precision = attack_tp / max(1, attack_tp + attack_fp)
        attack_recall = attack_tp / max(1, attack_tp + attack_fn)
        
        attack_labels = [l for l in labels if l != 'benign']
        attack_confusion = {}
        
        for true_attack in attack_labels:
            attack_confusion[true_attack] = {}
            for pred_attack in attack_labels:
                count = sum(1 for t, p in zip(y_true, y_pred) 
                           if t == true_attack and p == pred_attack)
                attack_confusion[true_attack][pred_attack] = count
        
        return {
            'overall_attack_detection': {
                'precision': attack_precision,
                'recall': attack_recall,
                'f1_score': 2 * (attack_precision * attack_recall) / max(0.001, attack_precision + attack_recall),
                'false_positive_rate': attack_fp / max(1, attack_fp + attack_tn),
                'false_negative_rate': attack_fn / max(1, attack_fn + attack_tp)
            },
            'attack_type_confusion': attack_confusion
        }
    
    def _analyze_errors(self, y_true: List[str], y_pred: List[str], 
                       labels: List[str]) -> Dict[str, Any]:
        
        errors = []
        for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            if true_label != pred_label:
                errors.append({
                    'index': i,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'error_type': self._classify_error_type(true_label, pred_label)
                })
        
        error_types = {}
        for error in errors:
            error_type = error['error_type']
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
        
        misclassifications = {}
        for error in errors:
            key = f"{error['true_label']} -> {error['predicted_label']}"
            if key not in misclassifications:
                misclassifications[key] = 0
            misclassifications[key] += 1
        
        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(y_true),
            'error_type_distribution': error_types,
            'common_misclassifications': dict(sorted(misclassifications.items(), 
                                                   key=lambda x: x[1], reverse=True)[:10])
        }
    
    def _classify_error_type(self, true_label: str, pred_label: str) -> str:
        if true_label == 'benign' and pred_label != 'benign':
            return 'false_positive'
        elif true_label != 'benign' and pred_label == 'benign':
            return 'false_negative'
        elif true_label != 'benign' and pred_label != 'benign':
            return 'attack_misclassification'
        else:
            return 'unknown'
    
    def _analyze_by_complexity(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        return {
            'note': 'Complexity analysis placeholder - could be enhanced with payload analysis',
            'simple_attacks': {'accuracy': 0.0, 'count': 0},
            'complex_attacks': {'accuracy': 0.0, 'count': 0}
        }
    
    def generate_evaluation_report(self, evaluation_result: Dict[str, Any], 
                                 save_path: str = None) -> str:
        
        report_lines = [
            "=" * 80,
            "SECURITY ANALYZER - COMPREHENSIVE EVALUATION REPORT",
            "=" * 80,
            f"Evaluation Date: {evaluation_result['timestamp']}",
            f"Dataset: {evaluation_result['dataset_info']}",
            "",
            "OVERALL PERFORMANCE:",
            "-" * 40,
            f"Accuracy: {evaluation_result['overall_metrics']['accuracy']:.4f}",
            f"Macro F1-Score: {evaluation_result['overall_metrics']['macro_f1']:.4f}",
            f"Weighted F1-Score: {evaluation_result['overall_metrics']['weighted_f1']:.4f}",
            f"ROC-AUC (Macro): {evaluation_result['overall_metrics']['roc_auc_macro']:.4f}",
            "",
            "ATTACK DETECTION PERFORMANCE:",
            "-" * 40
        ]
        
        attack_metrics = evaluation_result['attack_analysis']['overall_attack_detection']
        report_lines.extend([
            f"Attack Detection Precision: {attack_metrics['precision']:.4f}",
            f"Attack Detection Recall: {attack_metrics['recall']:.4f}",
            f"Attack Detection F1-Score: {attack_metrics['f1_score']:.4f}",
            f"False Positive Rate: {attack_metrics['false_positive_rate']:.4f}",
            f"False Negative Rate: {attack_metrics['false_negative_rate']:.4f}",
            "",
            "PER-CLASS PERFORMANCE:",
            "-" * 40
        ])
        
        for class_name, metrics in evaluation_result['per_class_metrics'].items():
            report_lines.extend([
                f"{class_name.upper()}:",
                f"  Precision: {metrics['precision']:.4f}",
                f"  Recall: {metrics['recall']:.4f}",
                f"  Support: {metrics['support']}",
                ""
            ])
        
        report_lines.extend([
            "ERROR ANALYSIS:",
            "-" * 40,
            f"Total Errors: {evaluation_result['error_analysis']['total_errors']}",
            f"Error Rate: {evaluation_result['error_analysis']['error_rate']:.4f}",
            "",
            "Common Misclassifications:"
        ])
        
        for misclass, count in evaluation_result['error_analysis']['common_misclassifications'].items():
            report_lines.append(f"  {misclass}: {count}")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Evaluation report saved to {save_path}")
        
        return report_text
    
    def save_evaluation_history(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2, default=str)
        self.logger.info(f"Evaluation history saved to {filename}") 