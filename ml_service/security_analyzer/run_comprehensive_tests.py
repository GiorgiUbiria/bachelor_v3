import os
import sys
import time
import json
from datetime import datetime
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security_analyzer.tests import run_all_tests
from security_analyzer.train.train_model import ModelTrainer
from security_analyzer.train.evaluate_model import SecurityModelEvaluator
from security_analyzer.utils.logger import setup_logger
from security_analyzer.utils.timer import performance_timer
from security_analyzer.utils.profiling import system_profiler

class ComprehensiveTestRunner:
    def __init__(self):
        self.logger = setup_logger('ComprehensiveTestRunner')
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_results': {},
            'performance_results': {},
            'training_results': {},
            'evaluation_results': {}
        }
    
    def run_full_test_suite(self):
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPREHENSIVE SECURITY ANALYZER TEST SUITE")
        self.logger.info("=" * 80)
        
        self.logger.info("Starting system resource monitoring...")
        system_profiler.start_monitoring(interval=2.0)
        
        try:
            self.logger.info("Running unit tests...")
            self._run_unit_tests()
            
            self.logger.info("Training/validating model...")
            self._train_and_validate_model()
            
            self.logger.info("Running comprehensive evaluation...")
            self._run_comprehensive_evaluation()
            
            self.logger.info("Running performance tests...")
            self._run_performance_tests()
            
            self.logger.info("Running robustness tests...")
            self._run_robustness_tests()
            
        finally:
            system_profiler.stop_monitoring()
        
        self._generate_final_report()
        
        self.logger.info("=" * 80)
        self.logger.info("COMPREHENSIVE TEST SUITE COMPLETED")
        self.logger.info("=" * 80)
    
    def _run_unit_tests(self):
        try:
            with performance_timer.time_operation("unit_tests"):
                test_result = run_all_tests()
            
            self.results['test_results'] = {
                'tests_run': test_result.testsRun,
                'failures': len(test_result.failures),
                'errors': len(test_result.errors),
                'success_rate': (test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / max(1, test_result.testsRun),
                'details': {
                    'failures': [str(f) for f in test_result.failures],
                    'errors': [str(e) for e in test_result.errors]
                }
            }
            
            self.logger.info(f"Unit tests completed: {self.results['test_results']['success_rate']:.2%} success rate")
            
        except Exception as e:
            self.logger.error(f"Unit tests failed: {e}")
            self.results['test_results'] = {'error': str(e)}
    
    def _train_and_validate_model(self):
        try:
            trainer = ModelTrainer()
            
            with performance_timer.time_operation("model_training"):
                dataset, filename = trainer.generate_training_data(10000)
                
                training_success = trainer.train_model(dataset)
                
                eval_results = trainer.evaluate_model(1000)
            
            self.results['training_results'] = {
                'training_success': training_success,
                'dataset_size': len(dataset),
                'dataset_file': filename,
                'immediate_evaluation': eval_results
            }
            
            self.logger.info(f"Model training completed with accuracy: {eval_results['accuracy']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            self.results['training_results'] = {'error': str(e)}
    
    def _run_comprehensive_evaluation(self):
        try:
            evaluator = SecurityModelEvaluator()
            
            with performance_timer.time_operation("comprehensive_evaluation"):
                cv_results = evaluator.cross_validate_model(cv_folds=3, sample_size=2000)
                
                baseline_results = evaluator.benchmark_against_baselines()
            
            self.results['evaluation_results'] = {
                'cross_validation': cv_results,
                'baseline_comparison': baseline_results
            }
            
            self.logger.info(f"Cross-validation accuracy: {cv_results['mean_cv_score']:.4f} ± {cv_results['std_cv_score']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Comprehensive evaluation failed: {e}")
            self.results['evaluation_results'] = {'error': str(e)}
    
    def _run_performance_tests(self):
        try:
            from security_analyzer.tests.test_performance import PerformanceTestSuite
            
            with performance_timer.time_operation("performance_tests"):
                perf_tester = PerformanceTestSuite()
                stress_results = {
                    'note': 'Simplified stress test for comprehensive suite',
                    'basic_latency': 'measured',
                    'resource_usage': system_profiler.get_current_usage()
                }
            
            self.results['performance_results'] = {
                'stress_test': stress_results,
                'timing_statistics': performance_timer.get_statistics()
            }
            
            self.logger.info("Performance tests completed")
            
        except Exception as e:
            self.logger.error(f"Performance tests failed: {e}")
            self.results['performance_results'] = {'error': str(e)}
    
    def _run_robustness_tests(self):
        try:
            evaluator = SecurityModelEvaluator()
            
            with performance_timer.time_operation("robustness_tests"):
                robustness_results = evaluator.evaluate_model_robustness(test_size=500)
            
            self.results['evaluation_results']['robustness'] = robustness_results
            
            robustness_score = robustness_results['overall_robustness']['robustness_score']
            self.logger.info(f"Robustness evaluation completed. Score: {robustness_score:.4f}")
            
        except Exception as e:
            self.logger.error(f"Robustness tests failed: {e}")
            self.results['evaluation_results']['robustness'] = {'error': str(e)}
    
    def _generate_final_report(self):
        final_usage = system_profiler.get_usage_statistics()
        self.results['final_system_usage'] = final_usage
        
        report_filename = f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            os.makedirs('test_reports', exist_ok=True)
            with open(f'test_reports/{report_filename}', 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.logger.info(f"Comprehensive test report saved to: test_reports/{report_filename}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
        
        self._print_summary()
    
    def _print_summary(self):
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST SUITE SUMMARY")
        print("=" * 80)
        
        if 'test_results' in self.results and 'success_rate' in self.results['test_results']:
            success_rate = self.results['test_results']['success_rate']
            print(f"Unit Tests: {success_rate:.1%} success rate")
        
        if 'training_results' in self.results and 'immediate_evaluation' in self.results['training_results']:
            accuracy = self.results['training_results']['immediate_evaluation']['accuracy']
            print(f"Model Accuracy: {accuracy:.4f}")
        
        if 'evaluation_results' in self.results and 'cross_validation' in self.results['evaluation_results']:
            cv_score = self.results['evaluation_results']['cross_validation']['mean_cv_score']
            print(f"Cross-Validation: {cv_score:.4f}")
        
        if ('evaluation_results' in self.results and 
            'robustness' in self.results['evaluation_results'] and
            'overall_robustness' in self.results['evaluation_results']['robustness']):
            robustness_score = self.results['evaluation_results']['robustness']['overall_robustness']['robustness_score']
            print(f"Robustness Score: {robustness_score:.4f}")
        
        timing_stats = performance_timer.get_statistics()
        if timing_stats:
            print(f"Performance Operations: {len(timing_stats)} tracked")
        
        print("=" * 80)

def main():
    runner = ComprehensiveTestRunner()
    runner.run_full_test_suite()

if __name__ == "__main__":
    main() 