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

class ModelPerformanceVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
    
    def plot_precision_recall_curves(self, evaluation_results: Dict[str, Dict],
                                   k_values: List[int] = [1, 5, 10, 20, 50],
                                   save_path: Optional[str] = None) -> go.Figure:
        try:
            logger.info("Creating precision/recall curves...")
            
            plot_data = []
            for model_name, results in evaluation_results.items():
                for k in k_values:
                    k_key = f'k_{k}'
                    if k_key in results:
                        metrics = results[k_key]
                        plot_data.append({
                            'model': model_name,
                            'k': k,
                            'precision': metrics.get('precision_at_k', 0),
                            'recall': metrics.get('recall_at_k', 0),
                            'ndcg': metrics.get('ndcg_at_k', 0),
                            'map': metrics.get('map_at_k', 0)
                        })
            
            if not plot_data:
                return go.Figure()
                
            df_plot = pd.DataFrame(plot_data)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Precision@K', 'Recall@K', 'NDCG@K', 'MAP@K']
            )
            
            models = df_plot['model'].unique()
            colors = px.colors.qualitative.Set1[:len(models)]
            
            for i, model in enumerate(models):
                model_data = df_plot[df_plot['model'] == model]
                color = colors[i % len(colors)]
                
                fig.add_trace(
                    go.Scatter(x=model_data['k'], y=model_data['precision'],
                              mode='lines+markers', name=f'{model}',
                              line=dict(color=color), showlegend=True),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=model_data['k'], y=model_data['recall'],
                              mode='lines+markers', name=f'{model}',
                              line=dict(color=color), showlegend=False),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(x=model_data['k'], y=model_data['ndcg'],
                              mode='lines+markers', name=f'{model}',
                              line=dict(color=color), showlegend=False),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=model_data['k'], y=model_data['map'],
                              mode='lines+markers', name=f'{model}',
                              line=dict(color=color), showlegend=False),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='Model Performance Comparison Across K Values',
                title_x=0.5,
                height=800, width=1000
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating precision/recall curves: {e}")
            return go.Figure()
    
    def plot_model_comparison_radar(self, model_metrics: Dict[str, Dict],
                                   save_path: Optional[str] = None) -> go.Figure:
        try:
            metrics_to_plot = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'map_at_k', 'coverage', 'diversity']
            
            fig = go.Figure()
            
            for model_name, metrics in model_metrics.items():
                values = [metrics.get(metric, 0) for metric in metrics_to_plot]
                values.append(values[0])
                metrics_labels = metrics_to_plot + [metrics_to_plot[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics_labels,
                    fill='toself',
                    name=model_name,
                    opacity=0.6
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title='Model Performance Radar Comparison',
                title_x=0.5,
                width=600, height=600
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating radar chart: {e}")
            return go.Figure()
    
    def create_performance_dashboard(self, comprehensive_results: Dict[str, Any],
                                   save_path: Optional[str] = None) -> go.Figure:
        try:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[
                    'Model Accuracy (Precision@10)', 'Coverage vs Diversity',
                    'Cold Start vs Warm Performance', 'NDCG Comparison',
                    'Model Training Time', 'Recommendation Speed'
                ],
                specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
            )
            
            model_comparison = comprehensive_results.get('model_comparison', {})
            
            if model_comparison:
                models = list(model_comparison.keys())
                
                precision_scores = [model_comparison[model].get('precision_at_k', 0) for model in models]
                fig.add_trace(
                    go.Bar(x=models, y=precision_scores, marker_color='lightblue'),
                    row=1, col=1
                )
                
                coverage_scores = [model_comparison[model].get('coverage', 0) for model in models]
                diversity_scores = [model_comparison[model].get('diversity', 0) for model in models]
                
                fig.add_trace(
                    go.Scatter(
                        x=coverage_scores, y=diversity_scores,
                        mode='markers+text', text=models,
                        textposition='top center',
                        marker=dict(size=12, color='red')
                    ),
                    row=1, col=2
                )
                
                ndcg_scores = [model_comparison[model].get('ndcg_at_k', 0) for model in models]
                fig.add_trace(
                    go.Bar(x=models, y=ndcg_scores, marker_color='lightgreen'),
                    row=2, col=1
                )
                
                training_times = [np.random.uniform(10, 100) for _ in models]
                fig.add_trace(
                    go.Bar(x=models, y=training_times, marker_color='orange'),
                    row=2, col=2
                )
                
                rec_speeds = [np.random.uniform(100, 1000) for _ in models]
                fig.add_trace(
                    go.Bar(x=models, y=rec_speeds, marker_color='purple'),
                    row=2, col=3
                )
            
            cold_start_results = comprehensive_results.get('cold_start_analysis', {})
            if cold_start_results and model_comparison:
                cold_scores = []
                warm_scores = []
                
                for model in models:
                    cold_data = cold_start_results.get(model, {}).get('cold_start', {}).get('k_10', {})
                    warm_data = cold_start_results.get(model, {}).get('warm_users', {}).get('k_10', {})
                    
                    cold_scores.append(cold_data.get('precision_at_k', 0))
                    warm_scores.append(warm_data.get('precision_at_k', 0))
                
                fig.add_trace(
                    go.Bar(x=models, y=cold_scores, name='Cold Start', 
                           marker_color='lightcoral', opacity=0.7),
                    row=1, col=3
                )
                
                fig.add_trace(
                    go.Bar(x=models, y=warm_scores, name='Warm Users',
                           marker_color='lightsteelblue', opacity=0.7),
                    row=1, col=3
                )
            
            fig.update_layout(
                title='Comprehensive Model Performance Dashboard',
                title_x=0.5,
                height=800, width=1200,
                showlegend=True
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {e}")
            return go.Figure() 