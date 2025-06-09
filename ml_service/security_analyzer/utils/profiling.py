import psutil
import time
import threading
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from ..utils.logger import setup_logger
from .performance_timer import performance_timer

logger = logging.getLogger(__name__)

@dataclass
class ResourceSnapshot:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: int
    network_bytes_recv: int

class SystemProfiler:
    def __init__(self):
        self.logger = setup_logger('SystemProfiler')
        self.monitoring = False
        self.monitor_thread = None
        self.snapshots: List[ResourceSnapshot] = []
        self.monitoring_interval = 5.0  # seconds
        
        self.baseline_disk_io = psutil.disk_io_counters()
        self.baseline_network = psutil.net_io_counters()
        
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'request_times': [],
            'threat_detections': 0,
            'total_requests': 0
        }
    
    def start_monitoring(self, interval: float = 5.0):
        """Start system monitoring"""
        if self.monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.monitoring_interval = interval
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"Started system monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("Stopped system monitoring")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                self.metrics['cpu_usage'].append(cpu_percent)
                self.metrics['memory_usage'].append(memory_percent)
                
                # Keep only last 100 measurements
                if len(self.metrics['cpu_usage']) > 100:
                    self.metrics['cpu_usage'] = self.metrics['cpu_usage'][-100:]
                    self.metrics['memory_usage'] = self.metrics['memory_usage'][-100:]
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in profiling loop: {e}")
                time.sleep(interval)
    
    def _take_snapshot(self) -> ResourceSnapshot:
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024 * 1024)
        
        disk_io = psutil.disk_io_counters()
        disk_read_mb = (disk_io.read_bytes - self.baseline_disk_io.read_bytes) / (1024 * 1024)
        disk_write_mb = (disk_io.write_bytes - self.baseline_disk_io.write_bytes) / (1024 * 1024)
        
        network_io = psutil.net_io_counters()
        
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory_used_mb,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_bytes_sent=network_io.bytes_sent - self.baseline_network.bytes_sent,
            network_bytes_recv=network_io.bytes_recv - self.baseline_network.bytes_recv
        )
    
    def get_current_usage(self) -> Dict[str, Any]:
        snapshot = self._take_snapshot()
        
        return {
            "cpu_percent": snapshot.cpu_percent,
            "memory_percent": snapshot.memory_percent,
            "memory_used_mb": snapshot.memory_used_mb,
            "disk_io_read_mb": snapshot.disk_io_read_mb,
            "disk_io_write_mb": snapshot.disk_io_write_mb,
            "network_bytes_sent": snapshot.network_bytes_sent,
            "network_bytes_recv": snapshot.network_bytes_recv,
            "timestamp": snapshot.timestamp
        }
    
    def get_usage_statistics(self, last_minutes: int = 5) -> Dict[str, Any]:
        if not self.snapshots:
            return {"error": "No monitoring data available"}
        
        cutoff_time = time.time() - (last_minutes * 60)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        
        if not recent_snapshots:
            return {"error": f"No data available for the last {last_minutes} minutes"}
        
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_percent for s in recent_snapshots]
        memory_mb_values = [s.memory_used_mb for s in recent_snapshots]
        
        return {
            "time_window_minutes": last_minutes,
            "sample_count": len(recent_snapshots),
            "cpu": {
                "average": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory": {
                "average_percent": sum(memory_values) / len(memory_values),
                "max_percent": max(memory_values),
                "min_percent": min(memory_values),
                "average_mb": sum(memory_mb_values) / len(memory_mb_values),
                "max_mb": max(memory_mb_values),
                "min_mb": min(memory_mb_values)
            },
            "disk_io": {
                "total_read_mb": recent_snapshots[-1].disk_io_read_mb,
                "total_write_mb": recent_snapshots[-1].disk_io_write_mb
            },
            "network": {
                "total_sent": recent_snapshots[-1].network_bytes_sent,
                "total_received": recent_snapshots[-1].network_bytes_recv
            }
        }
    
    def get_process_info(self, pid: Optional[int] = None) -> Dict[str, Any]:
        try:
            if pid is None:
                process = psutil.Process()
            else:
                process = psutil.Process(pid)
            
            with process.oneshot():
                return {
                    "pid": process.pid,
                    "name": process.name(),
                    "status": process.status(),
                    "cpu_percent": process.cpu_percent(),
                    "memory_info": {
                        "rss_mb": process.memory_info().rss / (1024 * 1024),
                        "vms_mb": process.memory_info().vms / (1024 * 1024),
                        "percent": process.memory_percent()
                    },
                    "threads": process.num_threads(),
                    "open_files": len(process.open_files()),
                    "connections": len(process.connections()),
                    "create_time": process.create_time()
                }
        except psutil.NoSuchProcess:
            return {"error": f"Process with PID {pid} not found"}
        except Exception as e:
            return {"error": f"Error getting process info: {str(e)}"}
    
    def record_request(self, processing_time: float, threat_detected: bool = False):
        """Record request metrics"""
        self.metrics['total_requests'] += 1
        self.metrics['request_times'].append(processing_time)
        
        if threat_detected:
            self.metrics['threat_detections'] += 1
            
        # Keep only last 1000 request times
        if len(self.metrics['request_times']) > 1000:
            self.metrics['request_times'] = self.metrics['request_times'][-1000:]
            
    def get_stats(self) -> Dict:
        """Get current performance statistics"""
        stats = {
            'system': {
                'avg_cpu': sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
                'avg_memory': sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
                'current_cpu': psutil.cpu_percent(),
                'current_memory': psutil.virtual_memory().percent
            },
            'requests': {
                'total_processed': self.metrics['total_requests'],
                'threats_detected': self.metrics['threat_detections'],
                'avg_processing_time': sum(self.metrics['request_times']) / len(self.metrics['request_times']) if self.metrics['request_times'] else 0,
                'threat_detection_rate': self.metrics['threat_detections'] / max(1, self.metrics['total_requests'])
            }
        }
        
        return stats

# Global profiler instance
system_profiler = SystemProfiler() 