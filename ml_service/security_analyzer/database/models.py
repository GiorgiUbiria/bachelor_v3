from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, UUID, ForeignKey, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime

Base = declarative_base()

class HttpRequestLog(Base):
    __tablename__ = "http_request_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    ip_address = Column(String(45))  # Support IPv6
    user_agent = Column(Text)
    path = Column(Text)
    method = Column(String(10))
    query_params = Column(Text)
    headers = Column(JSON)
    body = Column(JSON)
    cookies = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    duration_ms = Column(Integer)
    status_code = Column(Integer)
    referrer = Column(Text)
    suspected_attack_type = Column(String(20))  # xss, csrf, sqli, benign
    attack_score = Column(Float)
    confidence_score = Column(Float)
    is_malicious = Column(Boolean, default=False)
    pattern_matches = Column(JSON)  # Store matched patterns
    ml_prediction = Column(String(20))
    ensemble_weights = Column(JSON)  # Store ensemble decision weights

class MLAnalysisLog(Base):
    __tablename__ = "ml_analysis_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_log_id = Column(UUID(as_uuid=True), ForeignKey("http_request_logs.id"))
    analysis_type = Column(String(50), nullable=False)  # security_analysis, pattern_matching, etc.
    model_version = Column(String(20), default="2.0")
    input_features = Column(JSON)  # Processed features used for analysis
    ml_probabilities = Column(JSON)  # Class probabilities from ML model
    pattern_results = Column(JSON)  # Pattern matching results
    ensemble_decision = Column(JSON)  # How ensemble made final decision
    feature_importance = Column(JSON)  # Top contributing features
    explanation_data = Column(JSON)  # LIME/SHAP explanations
    processing_time_ms = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    request_log = relationship("HttpRequestLog")

class SecurityMetrics(Base):
    __tablename__ = "security_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(DateTime, default=datetime.utcnow)
    total_requests = Column(Integer, default=0)
    malicious_requests = Column(Integer, default=0)
    attack_type_counts = Column(JSON)  # {"xss": 10, "sqli": 5, ...}
    false_positive_rate = Column(Float)
    false_negative_rate = Column(Float)
    average_confidence = Column(Float)
    average_processing_time = Column(Float)
    model_accuracy = Column(Float)
    top_attack_sources = Column(JSON)  # Top IP addresses
    top_attack_paths = Column(JSON)  # Most targeted endpoints 

class SecurityFeedback(Base):
    __tablename__ = "security_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_log_id = Column(UUID(as_uuid=True), ForeignKey("http_request_logs.id", ondelete="CASCADE"))
    feedback_type = Column(String(20), CheckConstraint("feedback_type IN ('false_positive', 'false_negative', 'correct', 'unknown')"))
    original_prediction = Column(String(20))
    corrected_label = Column(String(20))
    feedback_source = Column(String(50))
    feedback_reason = Column(Text)
    confidence_in_feedback = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey("User.id", ondelete="SET NULL"))

class AttackMitigation(Base):
    __tablename__ = "attack_mitigation"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    attack_type = Column(String(20), nullable=False)
    attack_pattern = Column(Text)
    mitigation_strategy = Column(Text, nullable=False)
    sanitization_code = Column(Text)
    prevention_tips = Column(Text)
    severity_level = Column(String(10), CheckConstraint("severity_level IN ('low', 'medium', 'high', 'critical')"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class AblationStudyResults(Base):
    __tablename__ = "ablation_study_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_name = Column(String(100))
    component_removed = Column(String(50))
    baseline_accuracy = Column(Float)
    reduced_accuracy = Column(Float)
    performance_impact = Column(Float)
    component_importance = Column(Float)
    test_samples = Column(Integer)
    study_timestamp = Column(DateTime, default=datetime.utcnow)

class VisualizationData(Base):
    __tablename__ = "visualization_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    visualization_type = Column(String(20), CheckConstraint("visualization_type IN ('tsne', 'umap', 'pca')"))
    data_points = Column(JSON)
    parameters = Column(JSON)
    dataset_info = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow) 