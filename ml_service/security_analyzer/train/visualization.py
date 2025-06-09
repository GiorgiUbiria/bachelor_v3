import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd

class SecurityVisualization:
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, y_true, y_pred, labels, save_path=None):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Security Attack Detection - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curves(self, y_true, y_prob, labels, save_path=None):
        y_true_bin = label_binarize(y_true, classes=labels)
        n_classes = len(labels)
        
        plt.figure(figsize=(12, 8))
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            if i < y_true_bin.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, linewidth=2,
                        label=f'{label} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multi-class Attack Detection')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_attack_distribution(self, dataset, save_path=None):
        attack_counts = pd.Series([item['attack_type'] for item in dataset]).value_counts()
        
        plt.figure(figsize=(10, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = plt.bar(attack_counts.index, attack_counts.values, color=colors)
        
        plt.title('Distribution of Attack Types in Training Dataset')
        plt.xlabel('Attack Type')
        plt.ylabel('Number of Samples')
        
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return attack_counts.to_dict()
    
    def plot_feature_importance(self, classifier, vectorizer, top_n=20, save_path=None):
        if hasattr(classifier, 'feature_importances_'):
            feature_names = vectorizer.vectorizer.get_feature_names_out()
            importances = classifier.feature_importances_
            
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Top {top_n} Feature Importances - Security Classifier')
            plt.bar(range(top_n), importances[indices])
            plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            return [(feature_names[i], importances[i]) for i in indices]
        else:
            print("Classifier doesn't have feature_importances_ attribute")
            return [] 