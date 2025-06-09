from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = Field(default=10, ge=1, le=50)
    strategy: str = Field(default="hybrid", pattern="^(collaborative|content_based|clustering|hybrid)$")
    exclude_seen: bool = True
    context: Optional[Dict[str, Any]] = None

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    strategy: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class ItemAnalysisRequest(BaseModel):
    item_id: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class PricingRequest(BaseModel):
    item_id: str
    user_id: Optional[str] = None
    user_interest_score: float = Field(default=0.5, ge=0.0, le=1.0)
    stock_level: int = Field(default=50, ge=0)
    competitor_price: Optional[float] = None
    demand_score: float = Field(default=0.5, ge=0.0, le=1.0)

class TrainingRequest(BaseModel):
    model_type: str = Field(pattern="^(collaborative|content_based|clustering|hybrid|pricing|all)$")
    force_retrain: bool = False
    parameters: Optional[Dict[str, Any]] = None 