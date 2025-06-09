import os
import json
import time
from sklearn.metrics import classification_report, confusion_matrix
from ..data.synthetic_generator import SyntheticDataGenerator
from ..model.classifier import SecurityAnalyzer
from ..utils.logger import setup_logger

class ModelTrainer:
    def __init__(self):
        self.logger = setup_logger('ModelTrainer')
        self.generator = SyntheticDataGenerator()
        self.analyzer = SecurityAnalyzer()
    
    def generate_training_data(self, samples: int = 50000):
        self.logger.info(f"Generating {samples} synthetic training samples...")
        start_time = time.time()
        
        dataset = self.generator.generate_dataset(samples)
        
        os.makedirs('security_analyzer/data/datasets', exist_ok=True)
        filename = f'security_analyzer/data/datasets/training_data_{samples}.json'
        self.generator.save_dataset(dataset, filename)
        
        generation_time = time.time() - start_time
        self.logger.info(f"Generated {samples} samples in {generation_time:.2f} seconds")
        
        return dataset, filename
    
    def train_model(self, dataset=None, dataset_file=None):
        if dataset is None and dataset_file is None:
            self.logger.info("No dataset provided, generating new training data...")
            dataset, _ = self.generate_training_data()
        elif dataset_file:
            self.logger.info(f"Loading dataset from {dataset_file}")
            dataset = self.generator.load_dataset(dataset_file)
        
        self.logger.info(f"Training model with {len(dataset)} samples...")
        start_time = time.time()
        
        self.analyzer.train(dataset)
        
        training_time = time.time() - start_time
        self.logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        return True
    
    def evaluate_model(self, eval_samples: int = 4000):
        self.logger.info(f"Generating {eval_samples} evaluation samples...")
        eval_dataset = self.generator.generate_evaluation_set(eval_samples // 4)
        
        predictions = []
        true_labels = []
        
        self.logger.info("Running evaluation...")
        start_time = time.time()
        
        for sample in eval_dataset:
            from ..utils.schema import SecurityAnalysisRequest
            
            request = SecurityAnalysisRequest(**sample)
            result = self.analyzer.analyze(request)
            
            predictions.append(result.attack_type)
            true_labels.append(sample['attack_type'])
        
        eval_time = time.time() - start_time
        
        report = classification_report(true_labels, predictions, output_dict=True)
        cm = confusion_matrix(true_labels, predictions)
        
        self.logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        self.logger.info(f"Overall Accuracy: {report['accuracy']:.4f}")
        self.logger.info("\nDetailed Classification Report:")
        self.logger.info(classification_report(true_labels, predictions))
        
        return {
            'accuracy': report['accuracy'],
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'evaluation_time': eval_time,
            'samples_evaluated': len(eval_dataset)
        } 