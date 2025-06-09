from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import pandas as pd

from .train.train_hybrid import HybridTrainer
from .train.evaluation import ModelEvaluator
from .utils.schema import RecommendationRequest, RecommendationResponse, PricingRequest, TrainingRequest
from .utils.metrics import RecommendationMetrics
from .constants import USER_SEGMENTS, REGIONAL_PREFERENCES, AGE_PREFERENCES
from .data.demo_data import DemoDataGenerator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommendations", tags=["recommendations"])

trainer = HybridTrainer()
demo_data = DemoDataGenerator()

@router.post("/get")
async def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    try:
        if not trainer.models:
            await _ensure_models_loaded()
            
        model = trainer.models.get(request.strategy, trainer.models.get('hybrid'))
        if not model:
            raise HTTPException(status_code=503, detail="Recommendation model not available")
            
        recommendations = model.recommend(
            user_id=request.user_id,
            n_recommendations=request.num_recommendations,
            exclude_seen=request.exclude_seen
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            strategy=request.strategy,
            timestamp=datetime.now(),
            metadata={
                "total_recommendations": len(recommendations),
                "model_used": request.strategy
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/demo-users")
async def get_demo_users():
    return {
        "users": [
            {
                "user_id": "user_eu_young",
                "region": "EU",
                "age_group": "young",
                "preferences": ["electronics", "gaming", "fashion"],
                "description": "Young European user interested in tech and fashion"
            },
            {
                "user_id": "user_na_middle",
                "region": "NA", 
                "age_group": "middle",
                "preferences": ["sports", "home", "books"],
                "description": "Middle-aged North American user interested in home and sports"
            },
            {
                "user_id": "user_asia_young",
                "region": "ASIA",
                "age_group": "young", 
                "preferences": ["fashion", "beauty", "electronics"],
                "description": "Young Asian user interested in fashion and beauty"
            }
        ]
    }

@router.get("/segments")
async def get_user_segments():
    try:
        if not trainer.models or 'clustering' not in trainer.models:
            raise HTTPException(status_code=503, detail="Clustering model not available")
            
        clustering_model = trainer.models['clustering']
        
        segments_info = {}
        for cluster_id, segment_name in USER_SEGMENTS.items():
            if cluster_id in clustering_model.cluster_popular_items:
                segments_info[segment_name] = {
                    "cluster_id": cluster_id,
                    "popular_items": clustering_model.cluster_popular_items[cluster_id][:10],
                    "user_count": sum(1 for cid in clustering_model.user_clusters.values() if cid == cluster_id)
                }
                
        return {
            "segments": segments_info,
            "total_clusters": len(USER_SEGMENTS)
        }
        
    except Exception as e:
        logger.error(f"Error getting segments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pricing/predict")
async def predict_price(request: PricingRequest):
    try:
        if not trainer.models or 'pricing' not in trainer.models:
            raise HTTPException(status_code=503, detail="Pricing model not available")
            
        pricing_model = trainer.models['pricing']
        
        pricing_result = pricing_model.predict_price(
            item_id=request.item_id,
            user_interest_score=request.user_interest_score,
            stock_level=request.stock_level,
            competitor_price=request.competitor_price,
            demand_score=request.demand_score
        )
        
        if not pricing_result:
            raise HTTPException(status_code=404, detail="Item not found or pricing unavailable")
            
        return pricing_result
        
    except Exception as e:
        logger.error(f"Error predicting price: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/deals/{user_id}")
async def get_personalized_deals(user_id: str):
    try:
        if not trainer.models:
            await _ensure_models_loaded()
            
        recommendations = trainer.models['hybrid'].recommend(user_id, 20)
        
        deals = trainer.models['pricing'].get_personalized_deals(recommendations, max_deals=10)
        
        return {
            "user_id": user_id,
            "deals": deals,
            "total_deals": len(deals)
        }
        
    except Exception as e:
        logger.error(f"Error getting deals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(_train_models_background, request)
        return {"message": "Training started", "model_type": request.model_type}
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-info")
async def get_model_info():
    model_info = {}
    
    for name, model in trainer.models.items():
        if hasattr(model, '__class__'):
            model_info[name] = {
                "type": model.__class__.__name__,
                "available": True
            }
            
            if name == 'collaborative' and hasattr(model, 'user_ids'):
                model_info[name].update({
                    "users": len(model.user_ids),
                    "items": len(model.item_ids),
                    "algorithm": model.algorithm
                })
            elif name == 'clustering' and hasattr(model, 'n_clusters'):
                model_info[name].update({
                    "clusters": model.n_clusters,
                    "users_clustered": len(model.user_clusters)
                })
                
    return {"models": model_info}

@router.get("/visualizations/dashboard")
async def get_visualization_dashboard():
    try:
        if not trainer.models:
            await _ensure_models_loaded()
            
        dashboard_data = {
            "available_visualizations": [
                {
                    "name": "User Clusters",
                    "endpoint": "/recommendations/visualizations/clusters",
                    "description": "t-SNE/UMAP visualization of user segments"
                },
                {
                    "name": "Recommendation Overlap",
                    "endpoint": "/recommendations/visualizations/overlap",
                    "description": "Analysis of recommendation method overlaps"
                },
                {
                    "name": "Pricing Analysis", 
                    "endpoint": "/recommendations/visualizations/pricing",
                    "description": "Dynamic pricing distribution and factors"
                },
                {
                    "name": "Model Performance",
                    "endpoint": "/recommendations/visualizations/performance", 
                    "description": "Precision@K, Recall@K, NDCG comparison"
                },
                {
                    "name": "User Behavior",
                    "endpoint": "/recommendations/visualizations/behavior",
                    "description": "User interaction patterns and engagement"
                }
            ],
            "data_summary": {
                "trained_models": len(trainer.models)
            }
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting visualization dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/visualizations/performance")
async def get_performance_visualization():
    try:
        if not trainer.models:
            await _ensure_models_loaded()
            
        demo_interactions, demo_items, demo_users = demo_data.generate_demo_data()
        evaluator = ModelEvaluator()
        
        model_results = {}
        for model_name, model in trainer.models.items():
            if model_name != 'pricing':
                results = evaluator.evaluate_model(model, demo_interactions)
                model_results[model_name] = results
        
        comparison_results = {
            model_name: results.get('k_10', {}) 
            for model_name, results in model_results.items()
        }
        
        return {
            "visualization_type": "model_performance",
            "data": comparison_results,
            "metrics_available": ["precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k"],
            "k_values_tested": [5, 10, 20]
        }
        
    except Exception as e:
        logger.error(f"Error generating performance visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _ensure_models_loaded():
    if not trainer.models:
        success = trainer.load_models()
        if not success:
            logger.info("No trained models found, training with demo data...")
            demo_interactions, demo_items, demo_users = demo_data.generate_demo_data()
            trainer.train_all_models(demo_interactions, demo_items, demo_users)

async def _train_models_background(request: TrainingRequest):
    try:
        logger.info(f"Starting background training for {request.model_type}")
        
        demo_interactions, demo_items, demo_users = demo_data.generate_demo_data()
        
        if request.model_type == "all":
            trainer.train_all_models(demo_interactions, demo_items, demo_users)
        else:
            if request.model_type == "collaborative":
                trainer._train_collaborative_model(demo_interactions)
            elif request.model_type == "content_based":
                trainer._train_content_based_model(demo_items, demo_interactions)
            
        logger.info(f"Background training completed for {request.model_type}")
        
    except Exception as e:
        logger.error(f"Error in background training: {e}") 