from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union, List
import json

# Try to import field_validator for Pydantic v2, fallback to validator for v1
try:
    from pydantic import field_validator
    PYDANTIC_V2 = True
except ImportError:
    from pydantic import validator as field_validator
    PYDANTIC_V2 = False

class SecurityAnalysisRequest(BaseModel):
    method: str = "GET"
    path: str = "/"
    query_params: Union[Dict[str, Any], str] = {}
    headers: Optional[Dict[str, str]] = None
    body: Union[Dict[str, Any], str] = ""
    cookies: Optional[Dict[str, str]] = None
    ip_address: Optional[str] = "127.0.0.1"
    user_agent: Optional[str] = ""
    
    def __init__(self, **data):
        # Pre-process data before Pydantic validation
        if 'query_params' in data and isinstance(data['query_params'], str):
            # Keep as string for preprocessing, don't convert to dict yet
            pass  # Let preprocessing handle the string
        elif 'query_params' in data and isinstance(data['query_params'], dict):
            # Convert dict back to query string for consistent preprocessing
            if data['query_params']:
                try:
                    from urllib.parse import urlencode
                    data['query_params'] = urlencode(data['query_params'])
                except:
                    data['query_params'] = str(data['query_params'])
            else:
                data['query_params'] = ""
        elif 'query_params' not in data:
            data['query_params'] = ""
        
        if 'body' in data and isinstance(data['body'], dict):
            data['body'] = json.dumps(data['body']) if data['body'] else ""
        
        if 'headers' in data and data['headers'] is None:
            data['headers'] = {}
        
        if 'cookies' in data and data['cookies'] is None:
            data['cookies'] = {}
            
        super().__init__(**data)

class SecurityAnalysisResponse(BaseModel):
    is_malicious: bool
    attack_type: str
    attack_score: float
    confidence: float
    suspected_attack_type: Optional[str] = None
    details: Optional[Dict[str, Any]] = {}
    processing_time_ms: Optional[float] = None
    model_version: Optional[str] = "2.0"

class ThreatDetection(BaseModel):
    threat_type: str
    confidence: float
    severity: str
    description: Optional[str] = None
    recommended_action: Optional[str] = None

class ModelPerformance(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_samples: int
    model_version: str

class AttackPattern(BaseModel):
    pattern_id: str
    pattern_type: str
    regex_pattern: str
    description: str
    severity_level: int
    enabled: bool = True

class ExplainabilityResult(BaseModel):
    feature_importance: Dict[str, float]
    top_features: List[str]
    confidence_breakdown: Dict[str, float]
    explanation_text: Optional[str] = None

class ModelTrainingRequest(BaseModel):
    model_type: str = Field(default="ensemble", pattern="^(naive_bayes|svm|random_forest|ensemble)$")
    retrain: bool = Field(default=False)
    training_data_path: Optional[str] = Field(default=None)

class PatternRule(BaseModel):
    name: str
    pattern: str
    attack_type: str
    severity: str = Field(default="medium")
    description: Optional[str] = Field(default="")

class SystemStatus(BaseModel):
    status: str
    models_loaded: bool
    patterns_loaded: bool
    last_training: Optional[datetime] = Field(default=None)
    requests_processed: int = Field(default=0)
    threats_detected: int = Field(default=0) 