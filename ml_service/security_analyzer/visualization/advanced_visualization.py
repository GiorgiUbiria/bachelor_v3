import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

from ..utils.logger import setup_logger
from ..database.logger import SecurityDatabaseLogger

class AdvancedSecurityVisualization:
    def __init__(self):
        self.logger = setup_logger('AdvancedSecurityVisualization')
        self.db_logger = SecurityDatabaseLogger()
    
    def generate_attack_clusters_2d(self, feature_vectors: np.ndarray, 
                                  labels: List[str], method: str = 'tsne',
                                  save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate 2D visualization of attack clusters using t-SNE or UMAP"""
        
        self.logger.info(f"Generating 2D visualization using {method}")
        
        # Reduce dimensionality
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_vectors)-1))
            embeddings_2d = reducer.fit_transform(feature_vectors)
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(feature_vectors)-1))
            embeddings_2d = reducer.fit_transform(feature_vectors)
        else:  # PCA fallback
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(feature_vectors)
        
        # Create interactive plot
        fig = px.scatter(
            x=embeddings_2d[:, 0], 
            y=embeddings_2d[:, 1],
            color=labels,
            title=f"Security Attack Clusters - {method.upper()}",
            labels={'x': f'{method.upper()} Component 1', 'y': f'{method.upper()} Component 2'},
            hover_data={'index': list(range(len(labels)))}
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Visualization saved to {save_path}")
        
        # Store in database
        visualization_data = {
            'coordinates': embeddings_2d.tolist(),
            'labels': labels,
            'method': method,
            'parameters': {
                'n_components': 2,
                'random_state': 42
            }
        }
        
        self._store_visualization_data(method, visualization_data)
        
        return {
            'embeddings_2d': embeddings_2d,
            'plot': fig,
            'method': method,
            'cluster_analysis': self._analyze_clusters(embeddings_2d, labels)
        }
    
    def _analyze_clusters(self, embeddings_2d: np.ndarray, labels: List[str]) -> Dict[str, Any]:
        """Analyze cluster separation and characteristics"""
        
        unique_labels = list(set(labels))
        cluster_stats = {}
        
        for label in unique_labels:
            mask = np.array(labels) == label
            cluster_points = embeddings_2d[mask]
            
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                spread = np.std(cluster_points, axis=0)
                
                cluster_stats[label] = {
                    'count': len(cluster_points),
                    'centroid': centroid.tolist(),
                    'spread': spread.tolist(),
                    'avg_distance_to_centroid': np.mean(
                        np.linalg.norm(cluster_points - centroid, axis=1)
                    )
                }
        
        # Calculate inter-cluster distances
        inter_cluster_distances = {}
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels[i+1:], i+1):
                if label1 in cluster_stats and label2 in cluster_stats:
                    dist = np.linalg.norm(
                        np.array(cluster_stats[label1]['centroid']) - 
                        np.array(cluster_stats[label2]['centroid'])
                    )
                    inter_cluster_distances[f"{label1}-{label2}"] = dist
        
        return {
            'cluster_statistics': cluster_stats,
            'inter_cluster_distances': inter_cluster_distances,
            'separation_quality': self._calculate_separation_quality(cluster_stats, inter_cluster_distances)
        }
    
    def _calculate_separation_quality(self, cluster_stats: Dict, inter_distances: Dict) -> float:
        """Calculate a quality score for cluster separation (0-1)"""
        if not inter_distances:
            return 0.0
        
        avg_inter_distance = np.mean(list(inter_distances.values()))
        avg_intra_distance = np.mean([
            stats['avg_distance_to_centroid'] 
            for stats in cluster_stats.values()
        ])
        
        # Higher is better (well-separated clusters)
        separation_ratio = avg_inter_distance / max(avg_intra_distance, 0.001)
        return min(separation_ratio / 10.0, 1.0)  # Normalize to 0-1
    
    def create_ablation_study_visualization(self, ablation_results: Dict[str, float],
                                          save_path: Optional[str] = None) -> go.Figure:
        """Create visualization for ablation study results"""
        
        components = list(ablation_results.keys())
        performance_impacts = list(ablation_results.values())
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=components,
                y=performance_impacts,
                text=[f"{impact:.3f}" for impact in performance_impacts],
                textposition='auto',
                marker_color=['red' if x > 0 else 'green' for x in performance_impacts]
            )
        ])
        
        fig.update_layout(
            title="Ablation Study: Component Importance",
            xaxis_title="Component Removed",
            yaxis_title="Performance Impact (Accuracy Drop)",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_attack_timeline_visualization(self, days: int = 30) -> go.Figure:
        """Create timeline visualization of attacks"""
        
        dashboard_data = self.db_logger.get_security_dashboard_data(days)
        
        if 'daily_stats' not in dashboard_data:
            return go.Figure().add_annotation(text="No data available")
        
        daily_stats = dashboard_data['daily_stats']
        dates = [stat['date'] for stat in daily_stats]
        requests = [stat['requests'] for stat in daily_stats]
        threats = [stat['threats'] for stat in daily_stats]
        threat_rates = [stat['threat_rate'] for stat in daily_stats]
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Request Volume', 'Threat Detection Rate'),
            vertical_spacing=0.1
        )
        
        # Requests and threats
        fig.add_trace(
            go.Scatter(x=dates, y=requests, name='Total Requests', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=threats, name='Threats Detected', line=dict(color='red')),
            row=1, col=1
        )
        
        # Threat rate
        fig.add_trace(
            go.Scatter(x=dates, y=threat_rates, name='Threat Rate', line=dict(color='orange')),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"Security Timeline - Last {days} Days",
            height=600
        )
        
        return fig
    
    def create_feature_importance_heatmap(self, feature_importance: Dict[str, float],
                                        top_n: int = 20) -> go.Figure:
        """Create heatmap of feature importance"""
        
        # Sort and get top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[importances],
            x=features,
            y=['Feature Importance'],
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Feature Importances",
            xaxis_title="Features",
            height=300
        )
        
        return fig
    
    def _store_visualization_data(self, viz_type: str, data: Dict[str, Any]):
        """Store visualization data in database"""
        if not self.db_logger.db_available:
            return
        
        try:
            session = self.db_logger.SessionLocal()
            from ..database.models import VisualizationData
            
            viz_data = VisualizationData(
                visualization_type=viz_type,
                data_points=data,
                parameters=data.get('parameters', {}),
                dataset_info={'timestamp': datetime.now().isoformat()}
            )
            
            session.add(viz_data)
            session.commit()
            session.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store visualization data: {e}") 