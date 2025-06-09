import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, List, Any, Optional
import requests
import zipfile
import os
from urllib.parse import unquote
import re
from datetime import datetime
import time

from ..model.classifier import SecurityAnalyzer
from ..utils.schema import SecurityAnalysisRequest
from ..utils.logger import setup_logger

class RealWorldBenchmark:
    def __init__(self):
        self.logger = setup_logger('RealWorldBenchmark')
        self.benchmark_datasets = {
            'csic_2010': {
                'url': 'http://www.isi.csic.es/dataset/csic2010http.zip',
                'description': 'CSIC 2010 HTTP Dataset',
                'format': 'http_requests'
            },
            'custom_payloads': {
                'description': 'Custom security payload collection',
                'format': 'json'
            }
        }
    
    def download_csic_dataset(self, dataset_path: str = "datasets/csic2010") -> bool:
        try:
            os.makedirs(dataset_path, exist_ok=True)
            
            dataset_url = self.benchmark_datasets['csic_2010']['url']
            zip_path = os.path.join(dataset_path, "csic2010.zip")
            
            if not os.path.exists(zip_path):
                self.logger.info("Downloading CSIC 2010 dataset...")
                response = requests.get(dataset_url, stream=True)
                response.raise_for_status()
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                self.logger.info("Dataset downloaded successfully")
            
            extracted_path = os.path.join(dataset_path, "extracted")
            if not os.path.exists(extracted_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_path)
                self.logger.info("Dataset extracted successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download CSIC dataset: {e}")
            return False
    
    def parse_csic_dataset(self, dataset_path: str = "datasets/csic2010/extracted") -> List[Dict[str, Any]]:
        
        parsed_requests = []
        
        try:
            normal_files = []
            anomalous_files = []
            
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if 'normal' in file.lower() and file.endswith('.txt'):
                        normal_files.append(os.path.join(root, file))
                    elif 'anomal' in file.lower() and file.endswith('.txt'):
                        anomalous_files.append(os.path.join(root, file))
            
            for file_path in normal_files:
                requests = self._parse_http_file(file_path, label='benign')
                parsed_requests.extend(requests)
                self.logger.info(f"Parsed {len(requests)} normal requests from {file_path}")
            
            for file_path in anomalous_files:
                requests = self._parse_http_file(file_path, label='malicious')
                parsed_requests.extend(requests)
                self.logger.info(f"Parsed {len(requests)} anomalous requests from {file_path}")
            
            self.logger.info(f"Total parsed requests: {len(parsed_requests)}")
            return parsed_requests
            
        except Exception as e:
            self.logger.error(f"Failed to parse CSIC dataset: {e}")
            return []
    
    def _parse_http_file(self, file_path: str, label: str) -> List[Dict[str, Any]]:
        
        requests = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            http_requests = re.split(r'\n\n(?=GET|POST|PUT|DELETE)', content)
            
            for http_request in http_requests:
                if not http_request.strip():
                    continue
                
                parsed_request = self._parse_single_http_request(http_request, label)
                if parsed_request:
                    requests.append(parsed_request)
        
        except Exception as e:
            self.logger.error(f"Error parsing file {file_path}: {e}")
        
        return requests
    
    def _parse_single_http_request(self, http_request: str, label: str) -> Optional[Dict[str, Any]]:
        
        try:
            lines = http_request.strip().split('\n')
            if not lines:
                return None
            
            request_line = lines[0]
            method, path, protocol = request_line.split(' ', 2)
            
            headers = {}
            body_start = len(lines)
            
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '':
                    body_start = i + 1
                    break
                
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip()] = value.strip()
            
            if '?' in path:
                path_part, query_params = path.split('?', 1)
                query_params = unquote(query_params)
            else:
                path_part = path
                query_params = ""
            
            body = {}
            if body_start < len(lines):
                body_content = '\n'.join(lines[body_start:])
                if body_content.strip():
                    body = {'raw_body': body_content}
            
            attack_type = label
            if label == 'malicious':
                attack_type = self._classify_attack_type(path, query_params, body)
            
            return {
                'method': method,
                'path': path_part,
                'query_params': query_params,
                'headers': headers,
                'body': body,
                'user_agent': headers.get('User-Agent', ''),
                'ip_address': '127.0.0.1',  # Placeholder
                'attack_type': attack_type,
                'dataset_source': 'csic_2010'
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing HTTP request: {e}")
            return None
    
    def _classify_attack_type(self, path: str, query_params: str, body: Dict[str, Any]) -> str:
        
        full_content = f"{path} {query_params} {body.get('raw_body', '')}".lower()
        
        if any(pattern in full_content for pattern in [
            '<script', 'javascript:', 'onerror=', 'onload=', 'alert(', 'document.cookie'
        ]):
            return 'xss'
        
        if any(pattern in full_content for pattern in [
            'union select', "' or '", '1=1', 'drop table', 'insert into', '--'
        ]):
            return 'sqli'
        
        if 'csrf' in full_content or 'cross-site' in full_content:
            return 'csrf'
        
        return 'unknown'
    
    def benchmark_against_csic(self, analyzer: SecurityAnalyzer, 
                              max_samples: int = 5000) -> Dict[str, Any]:
        
        self.logger.info("Starting CSIC 2010 benchmark")
        
        if not self.download_csic_dataset():
            return {"error": "Failed to download CSIC dataset"}
        
        dataset = self.parse_csic_dataset()
        if not dataset:
            return {"error": "Failed to parse CSIC dataset"}
        
        if len(dataset) > max_samples:
            dataset = dataset[:max_samples]
        
        y_true = []
        y_pred = []
        processing_times = []
        
        self.logger.info(f"Benchmarking on {len(dataset)} samples")
        
        for i, sample in enumerate(dataset):
            if i % 100 == 0:
                self.logger.info(f"Processed {i}/{len(dataset)} samples")
            
            try:
                request = SecurityAnalysisRequest(**sample)
                
                start_time = time.time()
                result = analyzer.analyze(request)
                processing_time = time.time() - start_time
                
                processing_times.append(processing_time)
                
                true_label = 'malicious' if sample['attack_type'] != 'benign' else 'benign'
                pred_label = 'malicious' if result.is_malicious else 'benign'
                
                y_true.append(true_label)
                y_pred.append(pred_label)
                
            except Exception as e:
                self.logger.error(f"Error processing sample {i}: {e}")
                continue
        
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        attack_specific_metrics = self._calculate_attack_specific_metrics(dataset, analyzer)
        
        benchmark_results = {
            'dataset': 'CSIC 2010',
            'samples_tested': len(y_true),
            'overall_accuracy': accuracy,
            'classification_report': report,
            'attack_specific_metrics': attack_specific_metrics,
            'performance_metrics': {
                'avg_processing_time': np.mean(processing_times),
                'max_processing_time': max(processing_times),
                'min_processing_time': min(processing_times),
                'total_processing_time': sum(processing_times)
            },
            'benchmark_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"CSIC benchmark completed. Accuracy: {accuracy:.4f}")
        return benchmark_results
    
    def _calculate_attack_specific_metrics(self, dataset: List[Dict], 
                                         analyzer: SecurityAnalyzer) -> Dict[str, Any]:
        
        attack_metrics = {}
        attack_types = ['xss', 'sqli', 'csrf', 'unknown']
        
        for attack_type in attack_types:
            attack_samples = [s for s in dataset if s['attack_type'] == attack_type]
            
            if not attack_samples:
                continue
            
            correct = 0
            total = len(attack_samples)
            
            for sample in attack_samples:
                try:
                    request = SecurityAnalysisRequest(**sample)
                    result = analyzer.analyze(request)
                    
                    if result.attack_type == attack_type or (result.is_malicious and attack_type != 'benign'):
                        correct += 1
                except:
                    continue
            
            attack_metrics[attack_type] = {
                'samples': total,
                'accuracy': correct / total if total > 0 else 0,
                'correct_detections': correct
            }
        
        return attack_metrics
    
    def create_custom_payload_benchmark(self) -> List[Dict[str, Any]]:
        
        payloads = [
            {
                'payload': '"><img src=x onerror=eval(String.fromCharCode(97,108,101,114,116,40,49,41))>',
                'attack_type': 'xss',
                'description': 'Encoded XSS payload'
            },
            {
                'payload': '<svg/onload=alert`1`>',
                'attack_type': 'xss',
                'description': 'SVG-based XSS'
            },
            {
                'payload': "1' AND (SELECT COUNT(*) FROM information_schema.tables)>0 AND '1'='1",
                'attack_type': 'sqli',
                'description': 'Information schema injection'
            },
            {
                'payload': "'; WAITFOR DELAY '00:00:05' --",
                'attack_type': 'sqli',
                'description': 'Time-based blind SQL injection'
            },
            {
                'payload': 'csrf_token=invalid&action=transfer&amount=1000',
                'attack_type': 'csrf',
                'description': 'Invalid CSRF token'
            },
            {
                'payload': 'search=select product from catalog where price < 100',
                'attack_type': 'benign',
                'description': 'Benign search with SQL keywords'
            },
            {
                'payload': 'comment=This script is really helpful for beginners',
                'attack_type': 'benign', 
                'description': 'Benign comment with "script" keyword'
            }
        ]
        
        benchmark_dataset = []
        
        for payload_data in payloads:
            sample = {
                'method': 'POST',
                'path': '/api/test',
                'query_params': payload_data['payload'],
                'body': {'data': payload_data['payload']},
                'headers': {'User-Agent': 'SecurityBenchmark/1.0'},
                'ip_address': '192.168.1.100',
                'attack_type': payload_data['attack_type'],
                'dataset_source': 'custom_payloads',
                'description': payload_data['description']
            }
            benchmark_dataset.append(sample)
        
        return benchmark_dataset 