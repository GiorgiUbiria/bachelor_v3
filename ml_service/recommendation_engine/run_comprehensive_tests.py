import logging
import asyncio
from .train.train_hybrid import HybridTrainer
from .data.demo_data import DemoDataGenerator
from .train.evaluation import ModelEvaluator
from .utils.metrics import RecommendationMetrics

logger = logging.getLogger(__name__)

class ComprehensiveTestSuite:
    def __init__(self):
        self.trainer = HybridTrainer()
        self.demo_data = DemoDataGenerator()
        self.evaluator = ModelEvaluator()
        
    async def run_full_evaluation(self):
        logger.info("Starting comprehensive test suite...")
        
        try:
            interactions_df, items_df, users_df = self.demo_data.generate_demo_data(
                n_users=1000, n_items=2000, n_interactions=10000
            )
            
            success = self.trainer.train_all_models(interactions_df, items_df, users_df)
            if not success:
                logger.error("Model training failed")
                return False
                
            model_results = {}
            for model_name, model in self.trainer.models.items():
                if model_name != 'pricing':
                    logger.info(f"Evaluating {model_name} model...")
                    results = self.evaluator.evaluate_model(model, interactions_df)
                    model_results[model_name] = results
                    
            comparison = self.evaluator.compare_models(
                {k: v for k, v in self.trainer.models.items() if k != 'pricing'}, 
                interactions_df
            )
            
            cold_start_results = {}
            for model_name, model in self.trainer.models.items():
                if model_name != 'pricing':
                    cold_start = self.evaluator.evaluate_cold_start(model, interactions_df)
                    cold_start_results[model_name] = cold_start
                    
            return {
                'individual_results': model_results,
                'model_comparison': comparison,
                'cold_start_analysis': cold_start_results,
                'test_data_stats': {
                    'users': len(users_df),
                    'items': len(items_df), 
                    'interactions': len(interactions_df)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")
            return None
            
    def generate_performance_report(self, results):
        if not results:
            return "Evaluation failed - no results available"
            
        report = ["="*50, "RECOMMENDATION ENGINE PERFORMANCE REPORT", "="*50, ""]
        
        report.append("MODEL COMPARISON SUMMARY:")
        report.append("-" * 30)
        
        comparison = results.get('model_comparison', {})
        for model_name, metrics in comparison.items():
            report.append(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                report.append(f"  {metric}: {value:.4f}")
                
        report.append(f"\n\nCOLD START ANALYSIS:")
        report.append("-" * 30)
        
        cold_start = results.get('cold_start_analysis', {})
        for model_name, analysis in cold_start.items():
            if 'cold_start' in analysis and 'warm_users' in analysis:
                cold_metrics = analysis['cold_start'].get('k_10', {})
                warm_metrics = analysis['warm_users'].get('k_10', {})
                
                report.append(f"\n{model_name}:")
                report.append(f"  Cold Start Precision@10: {cold_metrics.get('precision_at_k', 0):.4f}")
                report.append(f"  Warm Users Precision@10: {warm_metrics.get('precision_at_k', 0):.4f}")
                
        stats = results.get('test_data_stats', {})
        report.append(f"\n\nTEST DATA STATISTICS:")
        report.append("-" * 30)
        report.append(f"Users: {stats.get('users', 0)}")
        report.append(f"Items: {stats.get('items', 0)}")
        report.append(f"Interactions: {stats.get('interactions', 0)}")
        
        return "\n".join(report)

async def main():
    test_suite = ComprehensiveTestSuite()
    results = await test_suite.run_full_evaluation()
    
    if results:
        report = test_suite.generate_performance_report(results)
        print(report)
        
        with open('recommendation_engine_report.txt', 'w') as f:
            f.write(report)
            
        logger.info("Comprehensive evaluation completed successfully")
    else:
        logger.error("Comprehensive evaluation failed")

if __name__ == "__main__":
    asyncio.run(main()) 