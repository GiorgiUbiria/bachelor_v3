"""
Core recommendation engine components
"""

from .engine import UnifiedRecommendationEngine
from .deal_generator import DynamicDealGenerator
from .user_segmentation import UserSegmentationEngine
from .data_processor import DataProcessor

__all__ = ["UnifiedRecommendationEngine", "DynamicDealGenerator", "UserSegmentationEngine", "DataProcessor"] 