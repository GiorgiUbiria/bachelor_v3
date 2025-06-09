import psutil
import time
import threading
import logging
from typing import Dict, List, Optional
import plotly.graph_objects as go
from datetime import datetime

logger = logging.getLogger(__name__)

class SystemProfiler:
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_percent': [],
            'memory_used_mb': [],
            'disk_io_read_mb': [],
            'disk_io_write_mb': []
        }
        
    def start_monitoring(self, interval: float = 1.0):
        """Start system monitoring in background thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("System monitoring started")
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("System monitoring stopped")
        
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        last_disk_io = psutil.disk_io_counters()
        
        while self.monitoring:
            try:
                # Get current metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                current_disk_io = psutil.disk_io_counters()
                disk_read_mb = (current_disk_io.read_bytes - last_disk_io.read_bytes) / (1024 * 1024)
                disk_write_mb = (current_disk_io.write_bytes - last_disk_io.write_bytes) / (1024 * 1024)
                
                # Store metrics
                self.metrics_history['timestamps'].append(datetime.now())
                self.metrics_history['cpu_percent'].append(cpu_percent)
                self.metrics_history['memory_percent'].append(memory.percent)
                self.metrics_history['memory_used_mb'].append(memory.used / (1024 * 1024))
                self.metrics_history['disk_io_read_mb'].append(disk_read_mb)
                self.metrics_history['disk_io_write_mb'].append(disk_write_mb)
                
                # Keep only last 1000 measurements
                if len(self.metrics_history['timestamps']) > 1000:
                    for key in self.metrics_history:
                        self.metrics_history[key] = self.metrics_history[key][-1000:]
                
                last_disk_io = current_disk_io
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
                
    def get_current_stats(self) -> Dict:
        """Get current system statistics"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3)
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
            
    def create_monitoring_dashboard(self, save_path: Optional[str] = None) -> go.Figure:
        """Create system monitoring dashboard"""
        try:
            if not self.metrics_history['timestamps']:
                logger.warning("No monitoring data available")
                return go.Figure()
                
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['CPU Usage (%)', 'Memory Usage (%)',
                              'Memory Usage (MB)', 'Disk I/O (MB/s)']
            )
            
            timestamps = self.metrics_history['timestamps']
            
            # CPU usage
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=self.metrics_history['cpu_percent'],
                    mode='lines',
                    name='CPU %',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Memory percentage
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=self.metrics_history['memory_percent'],
                    mode='lines',
                    name='Memory %',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
            
            # Memory usage in MB
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=self.metrics_history['memory_used_mb'],
                    mode='lines',
                    name='Memory MB',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            # Disk I/O
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=self.metrics_history['disk_io_read_mb'],
                    mode='lines',
                    name='Disk Read',
                    line=dict(color='purple')
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=self.metrics_history['disk_io_write_mb'],
                    mode='lines',
                    name='Disk Write',
                    line=dict(color='orange')
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='System Performance Monitoring',
                title_x=0.5,
                height=600, width=1000,
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating monitoring dashboard: {e}")
            return go.Figure()

# Global profiler instance
system_profiler = SystemProfiler() 