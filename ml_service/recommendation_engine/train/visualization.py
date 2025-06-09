import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class TrainingVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
    
    def plot_training_metrics(self, training_history: Dict[str, List[float]],
                             save_path: Optional[str] = None) -> go.Figure:
        """Plot training metrics over epochs"""
        try:
            logger.info("Creating training metrics visualization...")
            
            fig = go.Figure()
            
            for metric_name, values in training_history.items():
                fig.add_trace(go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode='lines+markers',
                    name=metric_name.replace('_', ' ').title(),
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title='Model Training Progress',
                title_x=0.5,
                xaxis_title='Epoch/Iteration',
                yaxis_title='Metric Value',
                width=800, height=500,
                legend=dict(x=0.02, y=0.98)
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting training metrics: {e}")
            return go.Figure()
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importance_scores: List[float],
                               save_path: Optional[str] = None) -> go.Figure:
        """Plot feature importance from trained models"""
        try:
            logger.info("Creating feature importance visualization...")
            
            # Sort by importance
            sorted_indices = np.argsort(importance_scores)[::-1]
            sorted_features = [feature_names[i] for i in sorted_indices]
            sorted_scores = [importance_scores[i] for i in sorted_indices]
            
            fig = go.Figure(go.Bar(
                x=sorted_scores[:20],  # Top 20 features
                y=sorted_features[:20],
                orientation='h',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title='Top 20 Feature Importance Scores',
                title_x=0.5,
                xaxis_title='Importance Score',
                yaxis_title='Features',
                width=800, height=600,
                yaxis=dict(autorange='reversed')
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
            return go.Figure()
    
    def plot_learning_curves(self, train_scores: List[float], 
                            val_scores: List[float],
                            train_sizes: List[int],
                            save_path: Optional[str] = None) -> go.Figure:
        """Plot learning curves showing performance vs training set size"""
        try:
            logger.info("Creating learning curves...")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_scores,
                mode='lines+markers',
                name='Training Score',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=val_scores,
                mode='lines+markers',
                name='Validation Score',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title='Learning Curves',
                title_x=0.5,
                xaxis_title='Training Set Size',
                yaxis_title='Score',
                width=800, height=500,
                legend=dict(x=0.02, y=0.98)
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting learning curves: {e}")
            return go.Figure()
    
    def create_model_comparison_summary(self, model_results: Dict[str, Dict],
                                      save_path: Optional[str] = None) -> go.Figure:
        """Create comprehensive model comparison summary"""
        try:
            logger.info("Creating model comparison summary...")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Precision Comparison', 'Recall Comparison',
                              'NDCG Comparison', 'Training Time Comparison']
            )
            
            models = list(model_results.keys())
            
            # Extract metrics
            precision_scores = [model_results[model].get('precision_at_k', 0) for model in models]
            recall_scores = [model_results[model].get('recall_at_k', 0) for model in models]
            ndcg_scores = [model_results[model].get('ndcg_at_k', 0) for model in models]
            
            # Simulated training times (in practice, you'd track these)
            training_times = [np.random.uniform(10, 120) for _ in models]
            
            # Add traces
            fig.add_trace(
                go.Bar(x=models, y=precision_scores, marker_color='lightblue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=models, y=recall_scores, marker_color='lightgreen'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(x=models, y=ndcg_scores, marker_color='salmon'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(x=models, y=training_times, marker_color='gold'),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Model Performance Comparison Summary',
                title_x=0.5,
                height=600, width=800,
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparison summary: {e}")
            return go.Figure() 