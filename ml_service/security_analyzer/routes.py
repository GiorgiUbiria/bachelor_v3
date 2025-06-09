from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import time

from .model.classifier import SecurityAnalyzer
from .train.train_model import ModelTrainer
from .data.synthetic_generator import SyntheticDataGenerator
from .utils.schema import SecurityAnalysisRequest, SecurityAnalysisResponse, SystemStatus
from .database.logger import SecurityDatabaseLogger
from .utils.performance_timer import PerformanceTimer
from .utils.mitigation_advisor import mitigation_advisor
from .utils.feedback_system import feedback_system
from .utils.advanced_viz import security_viz_engine
from .utils.ablation_framework import ablation_framework, SecurityAblationTests
from .benchmarks.dataset_benchmark import RealWorldBenchmark
from .analysis.error_analysis import AdvancedErrorAnalyzer
from .utils.profiling import system_profiler

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/security", tags=["Security Analysis"])

# Global instances
security_analyzer = SecurityAnalyzer()
model_trainer = ModelTrainer()
data_generator = SyntheticDataGenerator()
performance_timer = PerformanceTimer()
ablation_framework = ablation_framework

# Training status tracking
training_status = {
    "is_training": False,
    "progress": 0,
    "message": "Ready to train",
    "last_trained": None
}

# Initialize additional components
benchmark_runner = RealWorldBenchmark()
error_analyzer = AdvancedErrorAnalyzer()

# Global performance timer
perf_timer = PerformanceTimer()

@router.post("/analyze", response_model=SecurityAnalysisResponse)
async def analyze_request(request: SecurityAnalysisRequest):
    """Analyze HTTP request for security threats"""
    start_time = time.time()
    
    try:
        is_malicious = False
        threats_detected = []
        confidence = 0.1
        
        # Basic pattern matching for demonstration
        suspicious_patterns = ['<script>', 'SELECT * FROM', 'DROP TABLE', '../', 'exec(', 'eval(']
        
        request_content = f"{request.path} {request.method} {request.user_agent} {request.body or ''}"
        
        for pattern in suspicious_patterns:
            if pattern.lower() in request_content.lower():
                is_malicious = True
                confidence = 0.8
                
                # Determine attack type based on pattern
                if pattern in ['<script>']:
                    attack_type = "XSS"
                elif pattern in ['SELECT * FROM', 'DROP TABLE']:
                    attack_type = "SQL_INJECTION"
                elif pattern == '../':
                    attack_type = "PATH_TRAVERSAL"
                elif pattern in ['exec(', 'eval(']:
                    attack_type = "COMMAND_INJECTION"
                else:
                    attack_type = "UNKNOWN"
                
                threats_detected.append({
                    "attack_type": attack_type,
                    "confidence": confidence,
                    "evidence": [pattern],
                    "severity": "high"
                })
                break
        
        processing_time = time.time() - start_time
        
        # Record metrics
        system_profiler.record_request(processing_time, is_malicious)
        
        # Get mitigation advice if threats detected
        recommendations = []
        if threats_detected:
            for threat in threats_detected:
                advice = mitigation_advisor.get_mitigation_advice(
                    threat["attack_type"], 
                    threat["severity"]
                )
                recommendations.extend(advice["immediate_actions"][:2])  # Top 2 recommendations
        
        return SecurityAnalysisResponse(
            is_malicious=is_malicious,
            confidence=confidence,
            threats_detected=threats_detected,
            analysis_method="pattern_matching",
            processing_time=processing_time,
            timestamp=datetime.now(),
            recommendations=list(set(recommendations)),  # Remove duplicates
            metadata={
                "request_size": len(request_content),
                "patterns_checked": len(suspicious_patterns)
            }
        )
        
    except Exception as e:
        logger.error(f"Error analyzing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get security analyzer system status"""
    try:
        stats = system_profiler.get_stats()
        
        return SystemStatus(
            status="operational",
            models_loaded=False,  # Will be True when models are implemented
            patterns_loaded=True,
            last_training=None,
            requests_processed=stats["requests"]["total_processed"],
            threats_detected=stats["requests"]["threats_detected"]
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/demo-attacks")
async def get_demo_attacks():
    """Get demo attack scenarios for testing"""
    return {
        "attack_scenarios": [
            {
                "name": "XSS Attack",
                "description": "Cross-site scripting attempt",
                "example_request": {
                    "path": "/search?q=<script>alert('XSS')</script>",
                    "method": "GET",
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "ip_address": "192.168.1.100"
                },
                "expected_threat": "XSS"
            },
            {
                "name": "SQL Injection",
                "description": "SQL injection attempt",
                "example_request": {
                    "path": "/login",
                    "method": "POST",
                    "body": "username=admin&password=' OR '1'='1",
                    "user_agent": "curl/7.68.0",
                    "ip_address": "10.0.0.50"
                },
                "expected_threat": "SQL_INJECTION"
            },
            {
                "name": "Path Traversal",
                "description": "Directory traversal attempt",
                "example_request": {
                    "path": "/files?file=../../../etc/passwd",
                    "method": "GET",
                    "user_agent": "Python-requests/2.25.1",
                    "ip_address": "172.16.0.25"
                },
                "expected_threat": "PATH_TRAVERSAL"
            }
        ]
    }

@router.get("/mitigation-advice/{attack_type}")
async def get_mitigation_advice(attack_type: str):
    """Get detailed mitigation advice for specific attack type"""
    try:
        advice = mitigation_advisor.get_mitigation_advice(attack_type.upper())
        return advice
        
    except Exception as e:
        logger.error(f"Error getting mitigation advice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance-stats")
async def get_performance_stats():
    """Get security analyzer performance statistics"""
    try:
        stats = system_profiler.get_stats()
        timing_summary = perf_timer.get_summary()
        
        return {
            "system_performance": stats,
            "timing_details": perf_timer.get_timings(),
            "summary": timing_summary
        }
        
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
async def train_security_model(background_tasks: BackgroundTasks, samples: int = 50000):
    """Train the security model with synthetic data"""
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    def train_task():
        global training_status
        try:
            training_status.update({
                "is_training": True,
                "progress": 10,
                "message": "Generating synthetic data..."
            })
            
            dataset, _ = model_trainer.generate_training_data(samples)
            
            training_status.update({
                "progress": 50,
                "message": "Training model..."
            })
            
            model_trainer.train_model(dataset)
            
            training_status.update({
                "progress": 90,
                "message": "Evaluating model..."
            })
            
            eval_results = model_trainer.evaluate_model()
            
            training_status.update({
                "is_training": False,
                "progress": 100,
                "message": f"Training completed! Accuracy: {eval_results['accuracy']:.3f}",
                "last_trained": datetime.now().isoformat(),
                "evaluation_results": eval_results
            })
            
        except Exception as e:
            training_status.update({
                "is_training": False,
                "progress": 0,
                "message": f"Training failed: {str(e)}"
            })
    
    background_tasks.add_task(train_task)
    return {"message": "Training started", "samples": samples}

@router.get("/training-status")
async def get_training_status():
    """Get current training status"""
    return training_status

@router.post("/evaluate")
async def evaluate_security_model(samples: int = 4000):
    """Evaluate the security model performance"""
    if not security_analyzer.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    try:
        results = model_trainer.evaluate_model(samples)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@router.post("/batch-analyze")
async def batch_analyze_security(requests: List[SecurityAnalysisRequest]):
    """Analyze multiple HTTP requests for security threats"""
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 requests per batch")
    
    results = []
    for request in requests:
        try:
            result = security_analyzer.analyze(request)
            results.append(result)
        except Exception as e:
            results.append({
                "error": f"Analysis failed: {str(e)}",
                "request": request.dict()
            })
    
    return {"results": results, "total_analyzed": len(results)}

@router.post("/generate-data")
async def generate_synthetic_data(samples: int = 10000):
    """Generate synthetic security data for training"""
    if samples > 100000:
        raise HTTPException(status_code=400, detail="Maximum 100,000 samples allowed")
    
    try:
        dataset = data_generator.generate_dataset(samples)
        filename = f"security_analyzer/data/datasets/synthetic_data_{samples}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        data_generator.save_dataset(dataset, filename)
        
        distribution = {}
        for item in dataset:
            attack_type = item['attack_type']
            distribution[attack_type] = distribution.get(attack_type, 0) + 1
        
        return {
            "message": f"Generated {samples} synthetic samples",
            "filename": filename,
            "distribution": distribution
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")

@router.get("/attack-stats")
async def get_attack_statistics():
    """Get statistics about detected attacks"""
    return {
        "supported_attacks": ["xss", "sqli", "csrf"],
        "pattern_detectors": {
            "xss": len(data_generator.xss_payloads),
            "sqli": len(data_generator.sqli_payloads),
            "csrf": len(data_generator.csrf_scenarios)
        },
        "model_info": security_analyzer.get_model_info()
    }

@router.post("/stress-test")
async def run_stress_test():
    """Run performance stress test on the security analyzer"""
    try:
        from .tests.test_performance import PerformanceTestSuite
        performance_tester = PerformanceTestSuite()
        results = await performance_tester.run_stress_test()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stress test failed: {str(e)}")

@router.post("/analyze-detailed", response_model=SecurityAnalysisResponse)
async def analyze_security_detailed(request: SecurityAnalysisRequest, include_explanation: bool = False):
    """Analyze HTTP request with detailed explanations"""
    try:
        result = security_analyzer.analyze(request, include_explanation=include_explanation)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detailed analysis failed: {str(e)}")

@router.post("/explain")
async def explain_prediction(request: SecurityAnalysisRequest):
    """Get detailed explanation for a security prediction"""
    try:
        explanation = security_analyzer.get_explanation(request)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@router.get("/dashboard")
async def get_security_dashboard(days: int = 7):
    """Get security dashboard data for the last N days"""
    try:
        dashboard_data = security_analyzer.get_dashboard_data(days)
        return dashboard_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard data retrieval failed: {str(e)}")

@router.post("/update-metrics")
async def update_security_metrics():
    """Update daily security metrics"""
    try:
        db_logger = SecurityDatabaseLogger()
        db_logger.update_security_metrics()
        return {"message": "Security metrics updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics update failed: {str(e)}")

@router.get("/model-performance")
async def get_model_performance():
    """Get detailed model performance metrics"""
    try:
        if hasattr(security_analyzer.classifier, 'get_model_performance'):
            performance = security_analyzer.classifier.get_model_performance()
            return performance
        else:
            return {"message": "Enhanced performance metrics not available for this model"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance retrieval failed: {str(e)}")

@router.post("/analyze-timed", response_model=SecurityAnalysisResponse)
async def analyze_security_with_timing(request: SecurityAnalysisRequest):
    """Analyze HTTP request with performance timing"""
    try:
        with performance_timer.time_operation("security_analysis"):
            result = security_analyzer.analyze(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Timed analysis failed: {str(e)}")

@router.get("/feature-statistics")
async def get_feature_statistics():
    """Get feature importance statistics"""
    try:
        if security_analyzer.explainer:
            # Generate sample data for feature analysis
            sample_data = data_generator.generate_dataset(100)
            stats = security_analyzer.explainer.get_feature_statistics(sample_data)
            return stats
        else:
            return {"error": "Explainer not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature statistics failed: {str(e)}")

@router.post("/reset-performance-stats")
async def reset_performance_statistics():
    """Reset performance timing statistics"""
    try:
        performance_timer.reset_measurements()
        return {"message": "Performance statistics reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset stats: {str(e)}")

@router.get("/throughput-report")
async def get_throughput_report(window_seconds: int = 60):
    """Get throughput report for recent operations"""
    try:
        report = performance_timer.get_throughput_report(window_seconds)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Throughput report failed: {str(e)}")

@router.post("/run-comprehensive-tests")
async def run_comprehensive_test_suite(background_tasks: BackgroundTasks):
    """Run the complete comprehensive test suite"""
    try:
        def run_tests():
            from .run_comprehensive_tests import ComprehensiveTestRunner
            runner = ComprehensiveTestRunner()
            runner.run_full_test_suite()
        
        background_tasks.add_task(run_tests)
        return {
            "message": "Comprehensive test suite started in background",
            "note": "Check logs and test_reports/ directory for results"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start comprehensive tests: {str(e)}")

@router.get("/test-reports")
async def list_test_reports():
    """List available test reports"""
    try:
        import os
        reports_dir = "test_reports"
        if os.path.exists(reports_dir):
            reports = [f for f in os.listdir(reports_dir) if f.endswith('.json')]
            return {"reports": reports, "reports_directory": reports_dir}
        else:
            return {"reports": [], "message": "No test reports directory found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")

@router.post("/feedback")
async def submit_feedback(request_id: str, user_feedback: str, 
                         actual_threat: bool, predicted_threat: bool, 
                         confidence: float):
    """Submit feedback for model improvement"""
    try:
        feedback_system.record_feedback(
            request_id=request_id,
            user_feedback=user_feedback,
            actual_threat=actual_threat,
            predicted_threat=predicted_threat,
            confidence=confidence
        )
        
        return {
            "message": "Feedback recorded successfully",
            "request_id": request_id,
            "current_accuracy": feedback_system.get_model_accuracy()
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feedback/summary")
async def get_feedback_summary(days: int = 30):
    """Get feedback summary for the last N days"""
    try:
        summary = feedback_system.get_feedback_summary(days)
        return summary
        
    except Exception as e:
        logger.error(f"Error getting feedback summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/accuracy")
async def get_model_accuracy():
    """Get current model accuracy metrics"""
    try:
        accuracy = feedback_system.get_model_accuracy()
        improvement_areas = feedback_system.identify_improvement_areas()
        training_recommendations = feedback_system.get_training_recommendations()
        
        return {
            "accuracy_metrics": accuracy,
            "improvement_areas": improvement_areas,
            "training_recommendations": training_recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting model accuracy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ablation/run")
async def run_ablation_study(components: List[str] = ["pattern_matching", "ml_classifier", "ensemble"]):
    """Run ablation study on security analyzer components"""
    try:
        logger.info(f"Starting ablation study for components: {components}")
        
        # Set baseline performance
        baseline_metrics = {
            'accuracy': 0.90,
            'precision': 0.88,
            'recall': 0.85,
            'f1_score': 0.86
        }
        ablation_framework.set_baseline_performance(baseline_metrics)
        
        # Map component names to test functions
        test_functions = {
            'pattern_matching': SecurityAblationTests.test_pattern_matching_component,
            'ml_classifier': SecurityAblationTests.test_ml_classifier_component,
            'ensemble': SecurityAblationTests.test_ensemble_component
        }
        
        # Sample test data (in practice, this would be real test data)
        test_data = {'sample_requests': 1000, 'attack_ratio': 0.2}
        
        results = {}
        for component in components:
            if component in test_functions:
                result = ablation_framework.run_component_ablation(
                    component, 
                    test_functions[component], 
                    test_data
                )
                results[component] = result
                
        # Generate comprehensive analysis
        comprehensive_results = ablation_framework.run_comprehensive_ablation(
            components, 
            SecurityAblationTests.test_pattern_matching_component,  # Default test function
            test_data
        )
        
        return {
            "ablation_study": "completed",
            "components_tested": components,
            "individual_results": results,
            "comprehensive_analysis": comprehensive_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running ablation study: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ablation/results")
async def get_ablation_results():
    """Get ablation study results"""
    try:
        results = ablation_framework.export_results()
        ranking = ablation_framework._rank_components_by_contribution()
        summary = ablation_framework._generate_ablation_summary()
        
        return {
            "results": results,
            "component_ranking": ranking,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error getting ablation results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/ablation/clear")
async def clear_ablation_results():
    """Clear ablation study results"""
    try:
        ablation_framework.clear_results()
        return {"message": "Ablation study results cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing ablation results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/visualizations/dashboard")
async def get_security_dashboard():
    """Get security analysis dashboard"""
    try:
        # Generate sample data for visualization
        sample_metrics = {
            'threat_stats': {
                'detected': 150,
                'blocked': 140,
                'allowed': 10
            },
            'response_times': [0.1, 0.15, 0.08, 0.12, 0.09, 0.11, 0.14],
            'false_positive_timeline': [0.05, 0.03, 0.02, 0.04, 0.01],
            'system_load': {
                'cpu': 25.5,
                'memory': 67.8,
                'network': 12.3
            }
        }
        
        dashboard_fig = security_viz_engine.create_security_dashboard(sample_metrics)
        
        return {
            "dashboard_available": True,
            "metrics_summary": sample_metrics,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating security dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/visualizations/threat-timeline")
async def get_threat_timeline():
    """Get threat detection timeline visualization"""
    try:
        # Generate sample threat data
        sample_threats = [
            {
                'timestamp': datetime.now(),
                'attack_type': 'XSS',
                'confidence': 0.85,
                'ip_address': '192.168.1.100'
            },
            {
                'timestamp': datetime.now(),
                'attack_type': 'SQL_INJECTION',
                'confidence': 0.92,
                'ip_address': '10.0.0.50'
            }
        ]
        
        timeline_fig = security_viz_engine.create_threat_timeline(sample_threats)
        
        return {
            "timeline_available": True,
            "threat_count": len(sample_threats),
            "unique_attack_types": len(set(t['attack_type'] for t in sample_threats))
        }
        
    except Exception as e:
        logger.error(f"Error creating threat timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/visualizations/model-performance")
async def get_model_performance_viz():
    """Get model performance visualization"""
    try:
        # Get current model metrics
        model_metrics = feedback_system.get_model_accuracy()
        
        performance_radar = security_viz_engine.create_model_performance_radar(model_metrics)
        
        return {
            "performance_viz_available": True,
            "current_metrics": model_metrics,
            "visualization_type": "radar_chart"
        }
        
    except Exception as e:
        logger.error(f"Error creating performance visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 