import logging

logger = logging.getLogger(__name__)

try:
    from .routes import router
    ROUTES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Security analyzer routes not available: {e}")
    ROUTES_AVAILABLE = False
    router = None

__version__ = "1.0.0"
__all__ = ['router'] if ROUTES_AVAILABLE else [] 