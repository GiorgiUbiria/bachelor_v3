import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ComprehensiveTestRunner:
    def __init__(self):
        self.logger = setup_logger('ComprehensiveTestRunner')
        self.results = {}
        self.profiler = SystemProfiler()
        
        # Set testing mode
        os.environ['TESTING_MODE'] = 'true'
        os.environ['ENABLE_DB_LOGGING'] = 'false'
        
    def run_model_training(self):
        """Train and validate model"""
        try:
            self.logger.info("Training/validating model...")
            trainer = ModelTrainer()
            
            # Generate training data
            training_data = trainer.generate_training_data()
            
            # Train model with error handling
            training_result = trainer.train_model(training_data)
            
            if training_result and 'error' not in training_result:
                self.results['model_training'] = {
                    'status': 'success',
                    'details': training_result
                }
                return True
            else:
                self.logger.error(f"Model training failed: {training_result.get('error', 'Unknown error')}")
                self.results['model_training'] = {
                    'status': 'failed',
                    'error': training_result.get('error', 'Unknown error')
                }
                return False
                
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            self.results['model_training'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False 