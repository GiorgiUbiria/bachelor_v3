import asyncio
import aiohttp
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import psutil
import os
from ..utils.logger import setup_logger

class PerformanceTestSuite:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.logger = setup_logger('PerformanceTests')
    
    async def measure_latency(self, session, endpoint, data, num_requests=100):
        latencies = []
        
        for _ in range(num_requests):
            start_time = time.time()
            try:
                async with session.post(f"{self.base_url}{endpoint}", json=data) as response:
                    await response.json()
                    latency = time.time() - start_time
                    latencies.append(latency)
            except Exception as e:
                self.logger.error(f"Request failed: {e}")
        
        return latencies
    
    async def test_concurrent_requests(self, endpoint, data, concurrent_users=50, requests_per_user=20):
        self.logger.info(f"Testing {concurrent_users} concurrent users, {requests_per_user} requests each")
        
        async def user_session(session, user_id):
            user_latencies = []
            for _ in range(requests_per_user):
                start_time = time.time()
                try:
                    async with session.post(f"{self.base_url}{endpoint}", json=data) as response:
                        await response.json()
                        latency = time.time() - start_time
                        user_latencies.append(latency)
                except Exception as e:
                    self.logger.error(f"User {user_id} request failed: {e}")
            return user_latencies
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = [user_session(session, i) for i in range(concurrent_users)]
            all_latencies = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        flat_latencies = [lat for user_lats in all_latencies for lat in user_lats]
        
        total_requests = len(flat_latencies)
        throughput = total_requests / total_time
        
        return {
            'total_requests': total_requests,
            'total_time': total_time,
            'throughput': throughput,
            'avg_latency': statistics.mean(flat_latencies),
            'median_latency': statistics.median(flat_latencies),
            'p95_latency': sorted(flat_latencies)[int(0.95 * len(flat_latencies))],
            'p99_latency': sorted(flat_latencies)[int(0.99 * len(flat_latencies))],
            'min_latency': min(flat_latencies),
            'max_latency': max(flat_latencies)
        }
    
    def monitor_resources(self, duration=60):
        self.logger.info(f"Monitoring system resources for {duration} seconds")
        
        cpu_percentages = []
        memory_usage = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            cpu_percentages.append(psutil.cpu_percent(interval=1))
            memory_info = psutil.virtual_memory()
            memory_usage.append(memory_info.percent)
        
        return {
            'avg_cpu_percent': statistics.mean(cpu_percentages),
            'max_cpu_percent': max(cpu_percentages),
            'avg_memory_percent': statistics.mean(memory_usage),
            'max_memory_percent': max(memory_usage),
            'monitoring_duration': duration
        }
    
    async def run_stress_test(self):
        test_data = {
            "method": "POST",
            "path": "/api/search?q=<script>alert('xss')</script>",
            "headers": {"Content-Type": "application/json"},
            "body": {"comment": "' OR 1=1 --"},
            "ip_address": "192.168.1.100"
        }
        
        self.logger.info("Starting comprehensive stress test...")
        
        async with aiohttp.ClientSession() as session:
            latencies = await self.measure_latency(session, "/security/analyze", test_data, 100)
        
        basic_stats = {
            'avg_latency': statistics.mean(latencies),
            'median_latency': statistics.median(latencies),
            'p95_latency': sorted(latencies)[95],
            'min_latency': min(latencies),
            'max_latency': max(latencies)
        }
        
        concurrent_stats = await self.test_concurrent_requests(
            "/security/analyze", test_data, concurrent_users=100, requests_per_user=10
        )
        
        resource_stats = self.monitor_resources(30)
        
        return {
            'basic_performance': basic_stats,
            'concurrent_performance': concurrent_stats,
            'resource_usage': resource_stats
        } 