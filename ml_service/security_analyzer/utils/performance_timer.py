import time
import logging
from typing import Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PerformanceTimer:
    def __init__(self):
        self.timings = {}
        self.start_times = {}
        
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        if operation not in self.start_times:
            logger.warning(f"Timer for '{operation}' was not started")
            return 0.0
            
        duration = time.time() - self.start_times[operation]
        self.timings[operation] = duration
        del self.start_times[operation]
        return duration
        
    @contextmanager
    def timer(self, operation: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.timings[operation] = duration
            logger.debug(f"{operation} took {duration:.4f}s")
            
    def get_timings(self) -> Dict[str, float]:
        """Get all recorded timings"""
        return self.timings.copy()
        
    def get_timing(self, operation: str) -> Optional[float]:
        """Get timing for specific operation"""
        return self.timings.get(operation)
        
    def reset(self):
        """Reset all timings"""
        self.timings.clear()
        self.start_times.clear()
        
    def get_summary(self) -> str:
        """Get formatted summary of all timings"""
        if not self.timings:
            return "No timings recorded"
            
        summary = ["Performance Summary:", "=" * 20]
        for operation, duration in sorted(self.timings.items()):
            summary.append(f"{operation}: {duration:.4f}s")
            
        return "\n".join(summary)

# Global performance timer instance
performance_timer = PerformanceTimer() 