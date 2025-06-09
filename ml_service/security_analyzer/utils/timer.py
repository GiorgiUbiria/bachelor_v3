import time
import statistics
from contextlib import contextmanager
from typing import List, Dict, Any
from ..utils.logger import setup_logger

class PerformanceTimer:
    def __init__(self):
        self.logger = setup_logger('PerformanceTimer')
        self.measurements = {}
        self.active_timers = {}
    
    @contextmanager
    def time_operation(self, operation_name: str):
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.record_measurement(operation_name, duration)
    
    def start_timer(self, timer_name: str):
        self.active_timers[timer_name] = time.time()
    
    def stop_timer(self, timer_name: str) -> float:
        if timer_name not in self.active_timers:
            raise ValueError(f"Timer '{timer_name}' was not started")
        
        start_time = self.active_timers.pop(timer_name)
        duration = time.time() - start_time
        self.record_measurement(timer_name, duration)
        return duration
    
    def record_measurement(self, operation: str, duration: float):
        if operation not in self.measurements:
            self.measurements[operation] = []
        
        self.measurements[operation].append(duration)
        
        if len(self.measurements[operation]) > 1000:
            self.measurements[operation] = self.measurements[operation][-1000:]
    
    def get_statistics(self, operation: str = None) -> Dict[str, Any]:
        if operation:
            if operation not in self.measurements:
                return {"error": f"No measurements for operation '{operation}'"}
            
            times = self.measurements[operation]
            return self._calculate_stats(operation, times)
        
        all_stats = {}
        for op_name, times in self.measurements.items():
            all_stats[op_name] = self._calculate_stats(op_name, times)
        
        return all_stats
    
    def _calculate_stats(self, operation: str, times: List[float]) -> Dict[str, Any]:
        if not times:
            return {"count": 0}
        
        sorted_times = sorted(times)
        count = len(times)
        
        return {
            "operation": operation,
            "count": count,
            "total_time": sum(times),
            "average": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "std_dev": statistics.stdev(times) if count > 1 else 0,
            "p95": sorted_times[int(0.95 * count)],
            "p99": sorted_times[int(0.99 * count)] if count > 1 else sorted_times[0],
            "throughput_per_second": count / sum(times) if sum(times) > 0 else 0
        }
    
    def reset_measurements(self, operation: str = None):
        if operation:
            if operation in self.measurements:
                del self.measurements[operation]
        else:
            self.measurements.clear()
            self.active_timers.clear()
    
    def get_throughput_report(self, window_seconds: int = 60) -> Dict[str, Any]:
        current_time = time.time()
        recent_operations = {}
        
        for operation, times in self.measurements.items():
            recent_count = sum(1 for t in times if current_time - t <= window_seconds)
            recent_operations[operation] = {
                "operations_count": recent_count,
                "throughput_per_second": recent_count / window_seconds,
                "window_seconds": window_seconds
            }
        
        return recent_operations

performance_timer = PerformanceTimer() 