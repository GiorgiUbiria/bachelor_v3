"""
Unified Recommendation Engine Module
===================================

This module provides comprehensive personalized recommendations using multiple strategies:
1. Demographic-based filtering (age, region)
2. Collaborative filtering (user-based and item-based)
3. Content-based filtering using product features
4. Interaction-based filtering (comments, favorites, events)
5. Dynamic deal generation with user segmentation
6. Hybrid approach combining multiple methods

The module follows a clean architecture with:
- Core engine for recommendation logic
- Models for data handling
- Evaluation tools for research metrics
- Visualization components for analysis
- Comprehensive testing suite
"""

from .core.engine import UnifiedRecommendationEngine
from .core.deal_generator import DynamicDealGenerator
from .core.user_segmentation import UserSegmentationEngine
from .evaluation.metrics import RecommendationMetrics
from .evaluation.research_evaluation import ResearchEvaluator

__version__ = "2.0.0"
__author__ = "Bachelor Thesis Project"

__all__ = [
    "UnifiedRecommendationEngine",
    "DynamicDealGenerator", 
    "UserSegmentationEngine",
    "RecommendationMetrics",
    "ResearchEvaluator"
] 