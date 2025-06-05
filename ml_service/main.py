import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import asyncio
import traceback
from contextlib import asynccontextmanager

# Import ML models
from security_analyzer import SecurityAnalyzer
from models.recommendation_engine import RecommendationEngine
from models.enhanced_recommendation_engine import EnhancedRecommendationEngine
from models.product_automation import ProductAutomation

# Import database connection
from database import db_connection, is_database_available

# Import persistence functionality
import pickle
import json
from pathlib import Path

# Create models directory for persistence
MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)

# Model persistence functions
def save_model(model_instance, model_name: str):
    """Save a trained model to disk"""
    try:
        model_path = MODELS_DIR / f"{model_name}.pkl"
        metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model_instance, f)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'saved_at': datetime.now().isoformat(),
            'is_trained': getattr(model_instance, 'is_trained', False),
            'training_source': getattr(model_instance, 'training_data_source', 'unknown'),
            'version': '1.0'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Model {model_name} saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving model {model_name}: {e}")
        return False

def load_model(model_name: str, default_class):
    """Load a trained model from disk or return new instance"""
    try:
        model_path = MODELS_DIR / f"{model_name}.pkl"
        metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
        
        if model_path.exists() and metadata_path.exists():
            # Load metadata first
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if model is recent (less than 7 days old)
            from datetime import datetime, timedelta
            saved_at = datetime.fromisoformat(metadata['saved_at'])
            if datetime.now() - saved_at > timedelta(days=7):
                logger.info(f"Model {model_name} is older than 7 days, will retrain")
                return default_class()
            
            # Load model
            with open(model_path, 'rb') as f:
                model_instance = pickle.load(f)
            
            logger.info(f"✅ Model {model_name} loaded from disk (trained: {metadata.get('is_trained', False)})")
            return model_instance
        else:
            logger.info(f"No saved model found for {model_name}, creating new instance")
            return default_class()
            
    except Exception as e:
        logger.error(f"❌ Error loading model {model_name}: {e}")
        logger.info(f"Creating new {model_name} instance")
        return default_class()

def get_model_status() -> Dict:
    """Get status of all saved models"""
    status = {}
    
    for model_name in ['security_analyzer', 'recommendation_engine', 'enhanced_recommendation_engine', 'product_automation']:
        model_path = MODELS_DIR / f"{model_name}.pkl"
        metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
        
        if model_path.exists() and metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                status[model_name] = {
                    'saved': True,
                    'metadata': metadata,
                    'file_size': model_path.stat().st_size
                }
            except:
                status[model_name] = {'saved': False, 'error': 'Metadata corrupted'}
        else:
            status[model_name] = {'saved': False}
    
    return status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced ML Service",
    description="Three-Phase Machine Learning System for E-Commerce Platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models with persistence
logger.info("Initializing ML models with persistence...")
security_analyzer = load_model('security_analyzer', SecurityAnalyzer)
recommendation_engine = load_model('recommendation_engine', RecommendationEngine)
enhanced_recommendation_engine = load_model('enhanced_recommendation_engine', EnhancedRecommendationEngine)
product_automation = load_model('product_automation', ProductAutomation)

# Pydantic models for request/response validation
class HttpRequestData(BaseModel):
    id: Optional[str] = "unknown"
    path: str
    method: str = "GET"
    user_agent: Optional[str] = ""
    ip_address: Optional[str] = ""
    headers: Optional[Dict[str, str]] = {}
    body: Optional[str] = ""

class SecurityAnalysisResponse(BaseModel):
    request_id: str
    attack_score: float
    suspected_attack_type: Optional[str]
    confidence: float
    details: Dict[str, Any]
    recommendations: List[str]

class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 10
    strategy: str = "hybrid"  # hybrid, demographic, collaborative, content, interaction

class ProductData(BaseModel):
    name: str
    description: str
    category: Optional[str] = None
    brand: Optional[str] = None

class TrainingStatus(BaseModel):
    security_analyzer: bool
    recommendation_engine: bool
    product_automation: bool

class SecurityAnalysisRequest(BaseModel):
    method: str = "GET"
    path: str = "/"
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    body: Optional[str] = None
    query_params: Optional[Dict[str, str]] = None

class ProductAnalysisRequest(BaseModel):
    name: str
    description: str
    category: Optional[str] = None
    price: Optional[float] = None
    tags: Optional[List[str]] = None

class ModelTrainingRequest(BaseModel):
    model_type: str  # security, recommendations, product_automation, all
    force_retrain: bool = False

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with system status"""
    try:
        db_status = "connected" if is_database_available() else "disconnected"
        
        # Check model status
        models_status = {
            "security_analyzer": security_analyzer.is_trained,
            "recommendation_engine": recommendation_engine.is_trained,
            "product_automation": product_automation.is_trained
        }
        
        all_models_trained = all(models_status.values())
        
        return {
            "status": "healthy" if all_models_trained else "partial",
            "timestamp": datetime.now().isoformat(),
            "database": db_status,
            "models": models_status,
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# =============================================================================
# SECURITY ANALYSIS ENDPOINTS
# =============================================================================

@app.post("/security/train")
async def train_security_models(background_tasks: BackgroundTasks):
    """Train security models with synthetic data"""
    def train_models():
        try:
            security_analyzer.train()
            # Save model after successful training
            save_model(security_analyzer, 'security_analyzer')
        except Exception as e:
            logger.error(f"Security training error: {e}")
    
    background_tasks.add_task(train_models)
    return {"message": "Security model training started", "status": "training"}

@app.post("/security/train-from-database")
async def train_security_models_from_database(background_tasks: BackgroundTasks):
    """Train security models from database data"""
    def train_models():
        try:
            security_analyzer.train_from_database()
            # Save model after successful training
            save_model(security_analyzer, 'security_analyzer')
        except Exception as e:
            logger.error(f"Security database training error: {e}")
    
    background_tasks.add_task(train_models)
    return {"message": "Security model training from database started", "status": "training"}

@app.post("/security/analyze", response_model=SecurityAnalysisResponse)
async def analyze_request_security(request_data: HttpRequestData, log_to_db: bool = True):
    """
    Analyze HTTP request for security threats
    
    This endpoint demonstrates multi-layered security analysis using:
    - Pattern matching for known attack signatures
    - TF-IDF + Naive Bayes for text analysis
    - Logistic Regression for binary classification
    - Isolation Forest for anomaly detection
    
    Args:
        request_data: HTTP request data to analyze
        log_to_db: Whether to log the analysis result to database for future training
    """
    try:
        analysis = security_analyzer.analyze_request(request_data.dict())
        
        # Log to database if requested (for continuous learning)
        if log_to_db:
            try:
                security_analyzer.log_request_to_database(request_data.dict(), analysis)
            except Exception as e:
                logger.warning(f"Failed to log request to database: {e}")
        
        return SecurityAnalysisResponse(**analysis)
    except Exception as e:
        logger.error(f"Security analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/security/demo-attacks")
async def get_demo_attack_scenarios():
    """Get demo attack scenarios for testing the security analyzer"""
    return {
        "demo_attacks": [
            {
                "name": "XSS - Script Injection",
                "description": "Basic script tag injection attempt",
                "request": {
                    "method": "GET",
                    "path": "/search?q=<script>alert('XSS')</script>",
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "ip_address": "192.168.1.100",
                    "headers": {"Content-Type": "text/html"},
                    "body": ""
                },
                "expected_attack_type": "xss",
                "danger_level": "HIGH"
            },
            {
                "name": "SQL Injection - Union Select",
                "description": "SQL injection attempt using UNION SELECT",
                "request": {
                    "method": "POST",
                    "path": "/api/v1/auth/login",
                    "user_agent": "curl/7.68.0",
                    "ip_address": "10.0.0.5",
                    "headers": {"Content-Type": "application/x-www-form-urlencoded"},
                    "body": "username=admin' UNION SELECT password FROM users --&password=anything"
                },
                "expected_attack_type": "sqli",
                "danger_level": "CRITICAL"
            },
            {
                "name": "SQL Injection - Authentication Bypass",
                "description": "Attempt to bypass authentication with OR 1=1",
                "request": {
                    "method": "POST",
                    "path": "/login",
                    "user_agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36",
                    "ip_address": "172.16.0.10",
                    "headers": {"Content-Type": "application/json"},
                    "body": "{\"username\": \"admin' OR '1'='1\", \"password\": \"anything\"}"
                },
                "expected_attack_type": "sqli",
                "danger_level": "HIGH"
            },
            {
                "name": "CSRF - Hidden Form Attack",
                "description": "Cross-site request forgery with hidden form",
                "request": {
                    "method": "GET",
                    "path": "/malicious.html",
                    "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
                    "ip_address": "203.0.113.50",
                    "headers": {"Referer": "http://evil-site.com"},
                    "body": "<form action=\"http://bank.com/transfer\" method=\"post\"><input name=\"to\" value=\"attacker\"><input name=\"amount\" value=\"10000\"></form>"
                },
                "expected_attack_type": "csrf",
                "danger_level": "CRITICAL"
            },
            {
                "name": "XSS - Image Onerror",
                "description": "XSS using image onerror event",
                "request": {
                    "method": "GET",
                    "path": "/profile?bio=<img src=x onerror=alert('XSS')>",
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
                    "ip_address": "198.51.100.25",
                    "headers": {"Content-Type": "text/html"},
                    "body": ""
                },
                "expected_attack_type": "xss",
                "danger_level": "MEDIUM"
            },
            {
                "name": "Normal Request - API Call",
                "description": "Legitimate API request for testing",
                "request": {
                    "method": "GET",
                    "path": "/api/v1/products?category=electronics&limit=10",
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "ip_address": "192.168.1.50",
                    "headers": {"Authorization": "Bearer valid-token", "Content-Type": "application/json"},
                    "body": ""
                },
                "expected_attack_type": "normal",
                "danger_level": "NONE"
            },
            {
                "name": "Normal Request - User Profile",
                "description": "Normal user profile access",
                "request": {
                    "method": "GET",
                    "path": "/user/profile/123",
                    "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15",
                    "ip_address": "192.168.1.75",
                    "headers": {"Accept": "text/html,application/xhtml+xml"},
                    "body": ""
                },
                "expected_attack_type": "normal",
                "danger_level": "NONE"
            }
        ],
        "test_instructions": [
            "Copy any request object and use it with the /security/analyze endpoint",
            "The analyzer will classify the request and assess danger level",
            "Compare the results with the expected_attack_type and danger_level",
            "Use these scenarios to test model accuracy and fine-tune detection"
        ],
        "analyzer_endpoints": {
            "analyze": "/security/analyze",
            "model_info": "/security/model-info",
            "training_progress": "/models/training-progress"
        }
    }

@app.get("/security/model-info")
async def get_security_model_info():
    """Get information about trained security models"""
    return security_analyzer.get_model_info()

@app.get("/security/performance-report")
async def get_security_performance_report():
    """Get comprehensive performance report with detailed metrics"""
    try:
        report = security_analyzer.get_comprehensive_performance_report()
        return report
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate performance report: {str(e)}")

@app.post("/security/generate-visualizations")
async def generate_security_visualizations(save_path: str = "performance_plots"):
    """Generate performance visualization plots"""
    try:
        plots = security_analyzer.generate_performance_visualizations(save_path)
        return {
            "message": "Visualization plots generated successfully",
            "plots_generated": plots,
            "save_location": save_path
        }
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate visualizations: {str(e)}")

@app.post("/security/export-report")
async def export_security_report(file_path: str = "security_performance_report.json"):
    """Export comprehensive performance report to JSON file"""
    try:
        success = security_analyzer.export_performance_report(file_path)
        if success:
            return {
                "message": "Performance report exported successfully",
                "file_path": file_path,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to export performance report")
    except Exception as e:
        logger.error(f"Error exporting report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export report: {str(e)}")

@app.get("/security/statistics")
async def get_security_statistics():
    """Get security analysis statistics from database"""
    try:
        if not is_database_available():
            return {"error": "Database not available", "statistics": {}}
        
        # Get recent attack statistics
        attack_stats_query = """
        SELECT 
            suspected_attack_type,
            COUNT(*) as count,
            AVG(attack_score) as avg_score,
            MAX(attack_score) as max_score,
            COUNT(DISTINCT ip_address) as unique_ips
        FROM http_request_logs 
        WHERE timestamp > NOW() - INTERVAL '24 hours'
            AND suspected_attack_type IS NOT NULL
        GROUP BY suspected_attack_type
        ORDER BY count DESC
        """
        
        attack_stats = db_connection.execute_query(attack_stats_query)
        
        # Get hourly request volume
        volume_query = """
        SELECT 
            DATE_TRUNC('hour', timestamp) as hour,
            COUNT(*) as total_requests,
            COUNT(CASE WHEN attack_score > 0.7 THEN 1 END) as high_risk_requests,
            COUNT(CASE WHEN attack_score > 0.3 AND attack_score <= 0.7 THEN 1 END) as medium_risk_requests,
            AVG(duration_ms) as avg_response_time
        FROM http_request_logs
        WHERE timestamp > NOW() - INTERVAL '24 hours'
        GROUP BY hour
        ORDER BY hour DESC
        LIMIT 24
        """
        
        volume_stats = db_connection.execute_query(volume_query)
        
        # Get top attacking IPs
        ip_stats_query = """
        SELECT 
            ip_address,
            COUNT(*) as attack_attempts,
            AVG(attack_score) as avg_attack_score,
            MAX(timestamp) as last_attack
        FROM http_request_logs 
        WHERE timestamp > NOW() - INTERVAL '24 hours'
            AND attack_score > 0.5
        GROUP BY ip_address
        ORDER BY attack_attempts DESC
        LIMIT 10
        """
        
        ip_stats = db_connection.execute_query(ip_stats_query)
        
        return {
            "attack_statistics": attack_stats.to_dict('records') if attack_stats is not None else [],
            "volume_statistics": volume_stats.to_dict('records') if volume_stats is not None else [],
            "top_attacking_ips": ip_stats.to_dict('records') if ip_stats is not None else [],
            "timestamp": datetime.now().isoformat(),
            "time_range": "24 hours"
        }
        
    except Exception as e:
        logger.error(f"Error getting security statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get security statistics: {str(e)}")

@app.get("/security/health-check")
async def security_health_check():
    """Comprehensive health check for security analyzer"""
    try:
        health_status = {
            "security_analyzer": {
                "is_trained": security_analyzer.is_trained,
                "training_source": security_analyzer.training_data_source,
                "training_timestamp": getattr(security_analyzer, 'training_timestamp', 'unknown')
            },
            "database": {
                "connected": is_database_available(),
                "stats": db_connection.get_database_stats() if is_database_available() else {}
            },
            "model_performance": security_analyzer.model_performance if security_analyzer.is_trained else {},
            "attack_patterns": {
                "total_patterns": sum(len(patterns) for patterns in security_analyzer.attack_patterns.values()),
                "coverage": {k: len(v) for k, v in security_analyzer.attack_patterns.items()}
            },
            "system_status": "healthy" if security_analyzer.is_trained else "needs_training",
            "timestamp": datetime.now().isoformat()
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Security health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# =============================================================================
# RECOMMENDATION SYSTEM ENDPOINTS
# =============================================================================

@app.post("/recommendations/train")
async def train_recommendation_models(background_tasks: BackgroundTasks):
    """Train recommendation models with synthetic data"""
    def train_models():
        try:
            recommendation_engine.train()
            # Save model after successful training
            save_model(recommendation_engine, 'recommendation_engine')
        except Exception as e:
            logger.error(f"Recommendation training error: {e}")
    
    background_tasks.add_task(train_models)
    return {"message": "Recommendation model training started", "status": "training"}

@app.post("/recommendations/train-from-database")
async def train_recommendation_models_from_database(background_tasks: BackgroundTasks):
    """Train recommendation models from database data"""
    def train_models():
        try:
            recommendation_engine.train_from_database()
            # Save model after successful training
            save_model(recommendation_engine, 'recommendation_engine')
        except Exception as e:
            logger.error(f"Recommendation database training error: {e}")
    
    background_tasks.add_task(train_models)
    return {"message": "Recommendation model training from database started", "status": "training"}

@app.post("/recommendations/get")
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized product recommendations using multiple strategies
    
    Supports demographic, collaborative, content-based, interaction-based, and hybrid strategies
    """
    try:
        logger.info(f"Recommendation request for user {request.user_id} using {request.strategy} strategy")
        
        recommendations = recommendation_engine.get_recommendations(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
            strategy=request.strategy
        )
        
        return recommendations
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

@app.get("/recommendations/strategies")
async def get_recommendation_strategies():
    """Get available recommendation strategies with descriptions"""
    return {
        "strategies": {
            "hybrid": {
                "name": "Hybrid Recommendations",
                "description": "Combines multiple strategies for optimal results",
                "weights": "35% demographic, 30% collaborative, 25% interaction, 10% content"
            },
            "demographic": {
                "name": "Demographic-Based",
                "description": "Based on user age group and region preferences"
            },
            "collaborative": {
                "name": "Collaborative Filtering",
                "description": "Based on similar users' preferences and behaviors"
            },
            "content": {
                "name": "Content-Based",
                "description": "Based on product features and user interaction history"
            },
            "interaction": {
                "name": "Interaction-Based",
                "description": "Based on user's specific interaction patterns"
            }
        },
        "supported_regions": list(recommendation_engine.regional_preferences.keys()),
        "supported_age_groups": list(recommendation_engine.age_preferences.keys())
    }

@app.get("/recommendations/user/{user_id}/profile")
async def get_user_recommendation_profile(user_id: str):
    """Get detailed user profile for recommendations"""
    try:
        if not recommendation_engine.is_trained:
            raise HTTPException(status_code=503, detail="Recommendation engine not trained")
        
        user_info = recommendation_engine._get_user_info(user_id)
        
        # Get user's interaction summary
        interaction_summary = {}
        if recommendation_engine.interactions_df is not None:
            user_interactions = recommendation_engine.interactions_df[
                recommendation_engine.interactions_df['user_id'] == user_id
            ]
            if not user_interactions.empty:
                interaction_summary = {
                    "total_interactions": len(user_interactions),
                    "unique_products": user_interactions['product_id'].nunique(),
                    "interaction_types": user_interactions['event_type'].value_counts().to_dict()
                }
        
        return {
            "user_id": user_id,
            "profile": user_info,
            "interaction_summary": interaction_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user profile: {str(e)}")

@app.get("/recommendations/demo-users")
async def get_demo_users():
    """
    Get demonstration user profiles for testing recommendations
    
    Returns sample users from different regions and age groups to demonstrate
    how the recommendation system adapts to different user segments
    """
    if not recommendation_engine.is_trained:
        recommendation_engine.train()
    
    # Get sample users from different segments
    demo_users = []
    if recommendation_engine.users_df is not None:
        # Sample users from different regions and age groups
        for region in ['EU', 'NA', 'Asia', 'Other']:
            for age_group in ['young', 'adult', 'middle', 'senior']:
                region_age_users = recommendation_engine.users_df[
                    (recommendation_engine.users_df['region'] == region) & 
                    (recommendation_engine.users_df['age_group'] == age_group)
                ]
                
                if len(region_age_users) > 0:
                    user = region_age_users.iloc[0]
                    demo_users.append({
                        'user_id': user['user_id'],
                        'region': user['region'],
                        'age_group': user['age_group'],
                        'age': user['age'],
                        'cluster': user.get('cluster', 'unknown'),
                        'description': f"{age_group.title()} user from {region}"
                    })
    
    return {
        "demo_users": demo_users[:12],  # Limit to 12 users for demo
        "usage": "Use these user IDs with the /recommendations/get endpoint to see different recommendation strategies",
        "regional_preferences": recommendation_engine.regional_preferences,
        "age_preferences": recommendation_engine.age_preferences
    }

@app.get("/recommendations/deals/{user_id}")
async def get_dynamic_deals(user_id: str, num_deals: int = 5):
    """
    Get personalized dynamic deals for a user
    
    Demonstrates how ML can be used to create personalized pricing and deals
    based on user preferences and behavior patterns
    """
    try:
        deals = recommendation_engine.get_dynamic_deals(user_id, num_deals)
        return deals
    except Exception as e:
        logger.error(f"Dynamic deals error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deal generation failed: {str(e)}")

@app.get("/recommendations/segments")
async def analyze_user_segments():
    """
    Analyze user segments and their characteristics
    
    Shows how clustering can be used to understand different user groups
    and their preferences for targeted marketing and recommendations
    """
    try:
        segments = recommendation_engine.analyze_user_segments()
        return segments
    except Exception as e:
        logger.error(f"Segment analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Segment analysis failed: {str(e)}")

@app.get("/recommendations/model-info")
async def get_recommendation_model_info():
    """Get information about trained recommendation models"""
    return recommendation_engine.get_model_info()

# =============================================================================
# ENHANCED RECOMMENDATION SYSTEM ENDPOINTS (Multi-Stage Flow)
# =============================================================================

@app.post("/recommendations/enhanced/train")
async def train_enhanced_recommendation_models(background_tasks: BackgroundTasks):
    """Train enhanced recommendation models with database data"""
    def train_models():
        try:
            enhanced_recommendation_engine.train_from_database()
            # Save model after successful training
            save_model(enhanced_recommendation_engine, 'enhanced_recommendation_engine')
        except Exception as e:
            logger.error(f"Enhanced recommendation training error: {e}")
    
    background_tasks.add_task(train_models)
    return {
        "message": "Enhanced recommendation model training started",
        "status": "training",
        "features": [
            "Real-time k-NN recommendations",
            "Batch matrix factorization",
            "Dynamic deal generation",
            "User segmentation"
        ]
    }

@app.post("/recommendations/enhanced/batch-process")
async def run_batch_processing(background_tasks: BackgroundTasks):
    """
    Run Stage 2: Batch processing for matrix factorization and user clustering
    
    This endpoint triggers the hourly batch processing mentioned in Description-3.md:
    - Matrix factorization updates user-product affinity scores
    - Cluster users into segments for targeted deals
    - Generate dynamic pricing based on demand prediction
    """
    background_tasks.add_task(enhanced_recommendation_engine.run_batch_processing)
    return {
        "message": "Batch processing started in background",
        "status": "processing",
        "stage": "Stage 2: Batch Processing",
        "processes": [
            "Matrix factorization update",
            "User clustering and segmentation",
            "Demand prediction updates",
            "Seasonal factor calculation"
        ]
    }

@app.post("/recommendations/enhanced/realtime")
async def get_realtime_recommendations(request: RecommendationRequest):
    """
    Stage 1: Real-time scoring with k-NN for immediate recommendations
    
    Provides lightweight k-NN recommendations for immediate API response.
    Checks recent user activity and provides instant suggestions.
    """
    try:
        recommendations = enhanced_recommendation_engine.get_realtime_recommendations(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations
        )
        
        recommendations["stage"] = "Stage 1: Real-time Scoring"
        recommendations["latency_target"] = "< 100ms"
        recommendations["method_details"] = {
            "algorithm": "k-Nearest Neighbors",
            "similarity_metric": "cosine",
            "neighbors_considered": 20,
            "fallback": "demographic_recommendations"
        }
        
        return recommendations
    except Exception as e:
        logger.error(f"Real-time recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Real-time recommendation failed: {str(e)}")

@app.post("/recommendations/enhanced/full")
async def get_enhanced_recommendations_with_deals(request: RecommendationRequest):
    """
    Stage 3: Hybrid enrichment with dynamic deals
    
    Implements the complete multi-stage recommendation flow:
    - Combines collaborative filtering with content-based scores
    - Applies business rules for deal eligibility
    - Stores results with deal metadata
    
    Includes personalized deal generation with:
    - High-affinity products with low recent sales
    - Products similar to frequently viewed but not purchased
    - Seasonal/regional promotions
    """
    try:
        result = enhanced_recommendation_engine.get_enhanced_recommendations_with_deals(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
            include_deals=request.include_deals
        )
        
        result["api_info"] = {
            "endpoint": "/recommendations/enhanced/full",
            "stage": "Stage 3: Hybrid Enrichment",
            "features": [
                "Multi-strategy combination",
                "Dynamic deal generation",
                "Personalized pricing",
                "Regional targeting",
                "Seasonal promotions"
            ]
        }
        
        return result
    except Exception as e:
        logger.error(f"Enhanced recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced recommendation failed: {str(e)}")

@app.get("/recommendations/enhanced/deals/{user_id}")
async def get_personalized_deals(user_id: str, num_deals: int = 5):
    """
    Get personalized dynamic deals with enhanced logic
    
    Demonstrates the deal generation logic from Description-3.md:
    - Special offers for high-affinity products with low recent sales
    - Products similar to frequently viewed but not purchased
    - Seasonal/regional promotions
    
    Deal attributes include:
    - discount_percentage
    - deal_expiry  
    - deal_reason ("Popular in your area", "Limited time offer", etc.)
    """
    try:
        if not enhanced_recommendation_engine.is_trained:
            enhanced_recommendation_engine.train()
        
        user_profile = enhanced_recommendation_engine._get_user_profile(user_id)
        if user_profile is None:
            user_profile = {'region': 'Other', 'age_group': 'adult', 'spending_tier': 'mid'}
        
        deals = enhanced_recommendation_engine._generate_dynamic_deals(
            user_id, user_profile, num_deals
        )
        
        return {
            "user_id": user_id,
            "deals": deals,
            "user_profile": user_profile,
            "deal_generation_info": {
                "algorithm": "Multi-factor Dynamic Pricing",
                "factors": [
                    "Regional preferences",
                    "Age group targeting", 
                    "Demand-based pricing",
                    "Seasonal adjustments",
                    "User price sensitivity"
                ],
                "deal_types": [
                    "High-affinity low-sales products",
                    "Similar to viewed items",
                    "Regional seasonal promotions",
                    "Personalized discounts"
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Personalized deals error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deal generation failed: {str(e)}")

@app.get("/recommendations/enhanced/segments")
async def get_enhanced_user_segments():
    """
    Get enhanced user segmentation analysis
    
    Shows detailed user clustering results from the enhanced system:
    - User segments with characteristics
    - Regional distribution
    - Age group patterns
    - Spending behavior
    - Price sensitivity analysis
    """
    try:
        if not enhanced_recommendation_engine.is_trained:
            enhanced_recommendation_engine.train()
        
        segments_info = {
            "user_segments": enhanced_recommendation_engine.user_segments or {},
            "clustering_info": {
                "n_clusters": enhanced_recommendation_engine.user_clustering.n_clusters,
                "algorithm": "K-Means",
                "features_used": [
                    "Demographics (age, region)",
                    "Activity levels",
                    "Spending tiers", 
                    "Interaction patterns",
                    "Price sensitivity"
                ]
            },
            "regional_preferences": enhanced_recommendation_engine.regional_preferences,
            "age_preferences": enhanced_recommendation_engine.age_preferences,
            "model_info": enhanced_recommendation_engine.get_model_info()
        }
        
        return segments_info
    except Exception as e:
        logger.error(f"Enhanced segments error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Segment analysis failed: {str(e)}")

@app.get("/recommendations/enhanced/demo-flow")
async def get_demo_recommendation_flow():
    """
    Demonstrate the complete multi-stage recommendation flow
    
    Shows how the system works according to Description-3.md:
    1. Data Collection (user interactions, sentiment analysis, metadata)
    2. Multi-Stage Recommendation Generation
    3. Deal Generation Logic
    4. API Delivery
    """
    try:
        if not enhanced_recommendation_engine.is_trained:
            enhanced_recommendation_engine.train()
        
        # Get demo users from different segments
        demo_users = []
        if enhanced_recommendation_engine.users_df is not None:
            sample_users = enhanced_recommendation_engine.users_df.sample(min(6, len(enhanced_recommendation_engine.users_df)))
            for _, user in sample_users.iterrows():
                demo_users.append({
                    'user_id': user['user_id'],
                    'region': user['region'],
                    'age_group': user['age_group'],
                    'spending_tier': user['spending_tier'],
                    'activity_level': user['activity_level']
                })
        
        return {
            "demo_workflow": {
                "stage_1": {
                    "name": "Real-time Scoring",
                    "description": "Lightweight k-NN for immediate recommendations",
                    "endpoint": "/recommendations/enhanced/realtime",
                    "target_latency": "< 100ms"
                },
                "stage_2": {
                    "name": "Batch Processing", 
                    "description": "Hourly matrix factorization and user clustering",
                    "endpoint": "/recommendations/enhanced/batch-process",
                    "frequency": "Hourly background job"
                },
                "stage_3": {
                    "name": "Hybrid Enrichment",
                    "description": "Combined strategies with deal generation",
                    "endpoint": "/recommendations/enhanced/full",
                    "features": ["Collaborative + Content + Demographic + Deals"]
                }
            },
            "demo_users": demo_users,
            "sample_api_calls": [
                {
                    "description": "Get real-time recommendations",
                    "curl": f"curl -X POST '{app.url_path_for('get_realtime_recommendations')}' -H 'Content-Type: application/json' -d '{{\"user_id\": \"user_0\", \"num_recommendations\": 5}}'"
                },
                {
                    "description": "Get full recommendations with deals",
                    "curl": f"curl -X POST '{app.url_path_for('get_enhanced_recommendations_with_deals')}' -H 'Content-Type: application/json' -d '{{\"user_id\": \"user_0\", \"num_recommendations\": 10, \"include_deals\": true}}'"
                },
                {
                    "description": "Get personalized deals only",
                    "curl": f"curl '{app.url_path_for('get_personalized_deals', user_id='user_0')}?num_deals=5'"
                }
            ],
            "data_flow": {
                "data_collection": [
                    "User interactions (views, favorites, purchases)",
                    "Sentiment analysis on comments", 
                    "User metadata (region, age, purchase history)",
                    "Product features and categories"
                ],
                "deal_generation_logic": [
                    "High-affinity products with low recent sales",
                    "Products similar to frequently viewed but not purchased", 
                    "Seasonal/regional promotions",
                    "Dynamic pricing based on demand prediction"
                ],
                "api_delivery": [
                    "Base recommendations (sorted by affinity score)",
                    "Personalized deals (with discount details)",
                    "Explanations ('Why recommended')"
                ]
            }
        }
    except Exception as e:
        logger.error(f"Demo flow error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Demo flow failed: {str(e)}")

@app.get("/recommendations/enhanced/model-info")
async def get_enhanced_model_info():
    """Get detailed information about the enhanced recommendation models"""
    return enhanced_recommendation_engine.get_model_info()

# =============================================================================
# PRODUCT AUTOMATION ENDPOINTS
# =============================================================================

@app.post("/products/train")
async def train_product_models(background_tasks: BackgroundTasks):
    """Train product automation models with synthetic data"""
    def train_models():
        try:
            product_automation.train()
            # Save model after successful training
            save_model(product_automation, 'product_automation')
        except Exception as e:
            logger.error(f"Product automation training error: {e}")
    
    background_tasks.add_task(train_models)
    return {"message": "Product automation model training started", "status": "training"}

@app.post("/products/train-from-database")
async def train_product_models_from_database(background_tasks: BackgroundTasks):
    """Train product automation models from database data"""
    def train_models():
        try:
            product_automation.train_from_database()
            # Save model after successful training
            save_model(product_automation, 'product_automation')
        except Exception as e:
            logger.error(f"Product automation database training error: {e}")
    
    background_tasks.add_task(train_models)
    return {"message": "Product automation model training from database started", "status": "training"}

@app.post("/product/analyze")
async def analyze_product(request: ProductAnalysisRequest):
    """
    Comprehensive product analysis including price estimation, 
    category classification, and tag suggestions
    
    Works for both new and existing products
    """
    try:
        logger.info(f"Product analysis request: {request.name}")
        
        product_data = {
            "name": request.name,
            "description": request.description,
            "category": request.category,
            "price": request.price,
            "tags": request.tags
        }
        
        analysis_result = product_automation.analyze_new_product(product_data)
        return analysis_result
    except Exception as e:
        logger.error(f"Product analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Product analysis failed: {str(e)}")

@app.get("/products/demo-products")
async def get_demo_products():
    """
    Get demonstration products for testing automation features
    
    Returns sample product descriptions that can be used to test
    the automated tagging and pricing capabilities
    """
    demo_products = [
        {
            "name": "Wireless Bluetooth Headphones",
            "description": "High-quality wireless headphones with noise cancellation, 30-hour battery life, and premium sound quality. Perfect for music lovers and professionals.",
            "category": None,  # Let the system predict
            "price": None      # Let the system estimate
        },
        {
            "name": "Organic Cotton T-Shirt",
            "description": "Comfortable, breathable organic cotton t-shirt with modern fit. Eco-friendly and sustainable fashion choice for casual wear.",
            "category": None,
            "price": None
        },
        {
            "name": "Smart Home Security Camera",
            "description": "4K HD smart security camera with night vision, motion detection, and mobile app control. Easy installation and cloud storage.",
            "category": None,
            "price": None
        },
        {
            "name": "Professional Chef Knife Set",
            "description": "High-carbon stainless steel knife set with ergonomic handles. Includes chef knife, paring knife, and utility knife with wooden block.",
            "category": None,
            "price": None
        },
        {
            "name": "Fitness Tracker Watch",
            "description": "Advanced fitness tracker with heart rate monitoring, GPS, sleep tracking, and 7-day battery life. Water-resistant design.",
            "category": None,
            "price": None
        }
    ]
    
    return {
        "demo_products": demo_products,
        "usage": "Use these products with the /product/analyze endpoint to see automated categorization, tagging, and pricing"
    }

@app.get("/product/categories")
async def get_product_categories():
    """Get available product categories with pricing information"""
    try:
        categories_info = {}
        
        if product_automation.is_trained and product_automation.price_statistics:
            for category, stats in product_automation.price_statistics.items():
                categories_info[category] = {
                    "products_count": stats.get("count", 0),
                    "price_range": {
                        "min": stats.get("min", 0),
                        "max": stats.get("max", 0),
                        "average": stats.get("mean", 0),
                        "popular_range": stats.get("popular_range", {})
                    }
                }
        else:
            # Fallback to default categories
            categories_info = {category: data for category, data in product_automation.category_price_ranges.items()}
        
        return {
            "categories": categories_info,
            "tag_vocabulary": product_automation.category_tags,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")

@app.get("/products/model-info")
async def get_product_model_info():
    """Get information about trained product automation models"""
    return product_automation.get_model_info()

# =============================================================================
# GENERAL ENDPOINTS
# =============================================================================

@app.get("/models/status")
async def get_models_status():
    """Get training status of all ML models"""
    return TrainingStatus(
        security_analyzer=security_analyzer.is_trained,
        recommendation_engine=recommendation_engine.is_trained,
        product_automation=product_automation.is_trained
    )

@app.post("/models/train-all")
async def train_all_models(background_tasks: BackgroundTasks, use_database: bool = True):
    """Train all models either from database or with synthetic data"""
    
    async def train_all():
        training_results = {
            "started_at": datetime.now().isoformat(),
            "models": [],
            "database_used": use_database and is_database_available()
        }
        
        models_to_train = [
            ("security_analyzer", security_analyzer),
            ("recommendation_engine", recommendation_engine),
            ("enhanced_recommendation_engine", enhanced_recommendation_engine),
            ("product_automation", product_automation)
        ]
        
        for model_name, model_instance in models_to_train:
            try:
                logger.info(f"Training {model_name}...")
                
                if use_database and is_database_available():
                    if hasattr(model_instance, 'train_from_database'):
                        model_instance.train_from_database()
                    else:
                        model_instance.train()
                else:
                    model_instance.train()
                
                # Save model after successful training
                save_success = save_model(model_instance, model_name)
                
                training_results["models"].append({
                    "name": model_name,
                    "status": "success",
                    "trained": getattr(model_instance, 'is_trained', False),
                    "source": getattr(model_instance, 'training_data_source', 'unknown'),
                    "saved": save_success
                })
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                training_results["models"].append({
                    "name": model_name,
                    "status": "error",
                    "error": str(e),
                    "saved": False
                })
        
        training_results["completed_at"] = datetime.now().isoformat()
        logger.info(f"All models training completed: {training_results}")
    
    background_tasks.add_task(train_all)
    return {
        "message": "Training all models started",
        "status": "training",
        "models": ["security_analyzer", "recommendation_engine", "enhanced_recommendation_engine", "product_automation"],
        "database_available": is_database_available(),
        "use_database": use_database
    }

@app.get("/demo/complete-workflow")
async def get_complete_demo_workflow():
    """
    Get a complete demonstration workflow showing all ML capabilities
    
    This endpoint provides a guided tour of all the ML features with
    sample data and expected results for research demonstration
    """
    return {
        "workflow": {
            "step_1": {
                "title": "Train All Models",
                "options": {
                    "mock_data": {
                        "endpoint": "POST /models/train-all",
                        "description": "Initialize all ML models with generated sample data",
                        "estimated_time": "2-5 minutes"
                    },
                    "database_data": {
                        "endpoint": "POST /models/train-all?use_database=true",
                        "description": "Train models with actual database data (fallback to mock if unavailable)",
                        "estimated_time": "3-7 minutes"
                    }
                }
            },
            "step_1b": {
                "title": "Check Training Status & Performance",
                "endpoints": [
                    "GET /models/status - Check if all models are trained",
                    "GET /models/performance - View detailed performance metrics"
                ]
            },
            "step_2": {
                "title": "Security Analysis Demo",
                "endpoint": "GET /security/demo-attacks",
                "description": "Get sample attack scenarios to test security detection",
                "follow_up": "POST /security/analyze with each attack scenario",
                "advanced": {
                    "database_training": "POST /security/train-from-database",
                    "continuous_learning": "Use log_to_db=true parameter in analyze endpoint"
                }
            },
            "step_3": {
                "title": "Recommendation System Demo",
                "endpoint": "GET /recommendations/demo-users",
                "description": "Get sample users from different demographics",
                "follow_up": "POST /recommendations/get with different strategies",
                "advanced": {
                    "database_training": "POST /recommendations/train-from-database",
                    "user_segmentation": "GET /recommendations/segments"
                }
            },
            "step_4": {
                "title": "Product Automation Demo",
                "endpoint": "GET /products/demo-products",
                "description": "Get sample products for automated analysis",
                "follow_up": "POST /product/analyze for each product",
                "advanced": {
                    "database_training": "POST /products/train-from-database",
                    "category_insights": "GET /product/categories"
                }
            },
            "step_5": {
                "title": "Advanced Analytics & Performance",
                "endpoints": [
                    "GET /recommendations/segments - User clustering analysis",
                    "GET /product/categories - Category insights",
                    "GET /recommendations/deals/{user_id} - Dynamic pricing",
                    "GET /models/performance - Detailed performance metrics"
                ]
            }
        },
        "research_scenarios": {
            "regional_recommendations": "Compare recommendations for users from EU vs NA vs Asia",
            "age_based_preferences": "Analyze how recommendations differ across age groups",
            "attack_detection_accuracy": "Test various attack patterns and measure detection rates",
            "price_estimation_accuracy": "Compare estimated vs actual prices across categories",
            "tag_suggestion_relevance": "Evaluate ML vs rule-based tag suggestions",
            "database_vs_mock_training": "Compare model performance when trained on database vs mock data",
            "continuous_learning": "Analyze how security models improve with logged request data"
        },
        "database_integration": {
            "description": "All models can be trained using actual database data",
            "benefits": [
                "Real-world data patterns",
                "Improved accuracy for your specific use case",
                "Continuous learning from new data",
                "Better understanding of actual user behavior"
            ],
            "fallback": "Models automatically fallback to mock data if database is unavailable",
            "tables_used": [
                "users - User demographics and registration data",
                "products - Product catalog with descriptions and categories", 
                "user_events - User interaction logs (views, purchases, etc.)",
                "favorites - User favorite products",
                "http_request_logs - HTTP request logs for security analysis"
            ]
        },
        "performance_metrics": {
            "security_analyzer": [
                "Classification accuracy per attack type",
                "Precision/Recall for each model (Naive Bayes, Logistic Regression, Isolation Forest)",
                "F1-scores for multi-class classification"
            ],
            "recommendation_engine": [
                "User clustering distribution",
                "Data coverage (users, products, interactions)",
                "Regional and demographic preference analysis"
            ],
            "product_automation": [
                "Category classification accuracy",
                "Tag suggestion relevance scores",
                "Price estimation accuracy",
                "Cosine similarity validation for product matching"
            ]
        }
    }

@app.get("/models/performance")
async def get_models_performance():
    """Get performance metrics for all trained models"""
    try:
        performance_data = {}
        
        if hasattr(security_analyzer, 'model_performance'):
            performance_data["security_analyzer"] = security_analyzer.model_performance
        
        if hasattr(recommendation_engine, 'model_performance'):
            performance_data["recommendation_engine"] = getattr(recommendation_engine, 'model_performance', {})
        
        if hasattr(product_automation, 'model_performance'):
            performance_data["product_automation"] = getattr(product_automation, 'model_performance', {})
        
        return {
            "performance_metrics": performance_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@app.get("/models/persistence")
async def get_model_persistence_status():
    """Get the persistence status of all saved models"""
    try:
        model_status = get_model_status()
        
        # Add current runtime status
        runtime_status = {
            'security_analyzer': {
                'loaded': security_analyzer is not None,
                'trained': getattr(security_analyzer, 'is_trained', False),
                'source': getattr(security_analyzer, 'training_data_source', 'unknown')
            },
            'recommendation_engine': {
                'loaded': recommendation_engine is not None,
                'trained': getattr(recommendation_engine, 'is_trained', False),
                'source': getattr(recommendation_engine, 'training_data_source', 'unknown')
            },
            'enhanced_recommendation_engine': {
                'loaded': enhanced_recommendation_engine is not None,
                'trained': getattr(enhanced_recommendation_engine, 'is_trained', False),
                'source': 'unknown'  # Enhanced engine may not have this attribute
            },
            'product_automation': {
                'loaded': product_automation is not None,
                'trained': getattr(product_automation, 'is_trained', False),
                'source': getattr(product_automation, 'training_data_source', 'unknown')
            }
        }
        
        return {
            "models_directory": str(MODELS_DIR),
            "saved_models": model_status,
            "runtime_status": runtime_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting persistence status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting persistence status: {str(e)}")

@app.post("/models/save-all")
async def save_all_models():
    """Manually save all current models to disk"""
    try:
        results = {}
        models = {
            'security_analyzer': security_analyzer,
            'recommendation_engine': recommendation_engine,
            'enhanced_recommendation_engine': enhanced_recommendation_engine,
            'product_automation': product_automation
        }
        
        for model_name, model_instance in models.items():
            try:
                success = save_model(model_instance, model_name)
                results[model_name] = {
                    'saved': success,
                    'trained': getattr(model_instance, 'is_trained', False),
                    'source': getattr(model_instance, 'training_data_source', 'unknown')
                }
            except Exception as e:
                results[model_name] = {
                    'saved': False,
                    'error': str(e)
                }
        
        return {
            "message": "Model saving completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving models: {str(e)}")

@app.delete("/models/clear-saved")
async def clear_saved_models():
    """Clear all saved models from disk"""
    try:
        import shutil
        
        if MODELS_DIR.exists():
            # Remove all files in the models directory
            for file_path in MODELS_DIR.glob("*"):
                file_path.unlink()
            
            return {
                "message": "All saved models cleared",
                "models_directory": str(MODELS_DIR),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "message": "No saved models directory found",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error clearing saved models: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing saved models: {str(e)}")

@app.get("/models/training-progress")
async def get_training_progress():
    """Get current training progress and status"""
    try:
        return {
            "models": {
                "security_analyzer": {
                    "is_trained": getattr(security_analyzer, 'is_trained', False),
                    "training_source": getattr(security_analyzer, 'training_data_source', 'none'),
                    "status": "trained" if getattr(security_analyzer, 'is_trained', False) else "not_trained"
                },
                "recommendation_engine": {
                    "is_trained": getattr(recommendation_engine, 'is_trained', False),
                    "training_source": getattr(recommendation_engine, 'training_data_source', 'none'),
                    "status": "trained" if getattr(recommendation_engine, 'is_trained', False) else "not_trained"
                },
                "enhanced_recommendation_engine": {
                    "is_trained": getattr(enhanced_recommendation_engine, 'is_trained', False),
                    "training_source": "unknown",
                    "status": "trained" if getattr(enhanced_recommendation_engine, 'is_trained', False) else "not_trained"
                },
                "product_automation": {
                    "is_trained": getattr(product_automation, 'is_trained', False),
                    "training_source": getattr(product_automation, 'training_data_source', 'none'),
                    "status": "trained" if getattr(product_automation, 'is_trained', False) else "not_trained"
                }
            },
            "database_available": is_database_available(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting training progress: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting training progress: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "message": "Check /docs for available endpoints",
        "available_endpoints": [
            "/health - Health check",
            "/docs - API documentation",
            "/demo/complete-workflow - Complete demo guide"
        ]
    }

# Startup event to train models
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup - only train if needed"""
    logger.info("🚀 ML Service starting up...")
    logger.info(f"Database available: {is_database_available()}")
    
    # Check which models need training
    models_to_train = []
    
    if not getattr(security_analyzer, 'is_trained', False):
        models_to_train.append(("security_analyzer", security_analyzer))
    
    if not getattr(recommendation_engine, 'is_trained', False):
        models_to_train.append(("recommendation_engine", recommendation_engine))
    
    if not getattr(enhanced_recommendation_engine, 'is_trained', False):
        models_to_train.append(("enhanced_recommendation_engine", enhanced_recommendation_engine))
    
    if not getattr(product_automation, 'is_trained', False):
        models_to_train.append(("product_automation", product_automation))
    
    if models_to_train:
        logger.info(f"Training {len(models_to_train)} models that need training...")
        
        async def train_models():
            # Run training in separate threads to avoid blocking
            for model_name, model_instance in models_to_train:
                try:
                    logger.info(f"Training {model_name}...")
                    
                    # Try database first, fallback to synthetic
                    if is_database_available() and hasattr(model_instance, 'train_from_database'):
                        model_instance.train_from_database()
                    else:
                        model_instance.train()
                    
                    # Save after training
                    save_model(model_instance, model_name)
                    logger.info(f"✅ {model_name} trained and saved")
                    
                except Exception as e:
                    logger.error(f"❌ Error training {model_name}: {e}")
        
        # Start training in background
        asyncio.create_task(train_models())
    else:
        logger.info("✅ All models loaded from disk and ready!")
    
    logger.info("🎯 ML Service startup completed")

if __name__ == "__main__":
    logger.info("Starting ML Service API...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 