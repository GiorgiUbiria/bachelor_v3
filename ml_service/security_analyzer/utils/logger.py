import logging
import sys
from datetime import datetime

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def log_security_event(event_type: str, details: dict, severity: str = "info"):
    """Log security events with structured format"""
    security_logger = logging.getLogger("security_events")
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "severity": severity,
        "details": details
    }
    
    if severity.lower() == "critical":
        security_logger.critical(log_entry)
    elif severity.lower() == "high":
        security_logger.error(log_entry)
    elif severity.lower() == "medium":
        security_logger.warning(log_entry)
    else:
        security_logger.info(log_entry) 