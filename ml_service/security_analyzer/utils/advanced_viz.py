import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import io
import base64

logger = logging.getLogger(__name__)

# Conditional imports for plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available, falling back to matplotlib")
    PLOTLY_AVAILABLE = False
    px = None
    go = None
    make_subplots = None

class SecurityVisualizationEngine:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            logger.error(f"Error converting figure to base64: {e}")
            return ""
        
    def create_threat_timeline(self, threat_data: List[Dict], 
                              save_path: Optional[str] = None) -> Any:
        """Create timeline visualization of detected threats"""
        try:
            if not threat_data:
                logger.warning("No threat data available for timeline")
                return None
                
            df = pd.DataFrame(threat_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if PLOTLY_AVAILABLE:
                # Group by hour and threat type
                df['hour'] = df['timestamp'].dt.floor('H')
                threat_counts = df.groupby(['hour', 'attack_type']).size().reset_index(name='count')
                
                fig = px.line(
                    threat_counts, 
                    x='hour', 
                    y='count',
                    color='attack_type',
                    title='Threat Detection Timeline',
                    labels={'hour': 'Time', 'count': 'Number of Threats'}
                )
                
                fig.update_layout(
                    width=800, height=400,
                    showlegend=True,
                    title_x=0.5
                )
                
                if save_path:
                    fig.write_html(save_path)
                    
                return fig
            else:
                # Matplotlib fallback
                fig, ax = plt.subplots(figsize=(10, 6))
                
                df['hour'] = df['timestamp'].dt.floor('H')
                for attack_type in df['attack_type'].unique():
                    type_data = df[df['attack_type'] == attack_type]
                    threat_counts = type_data.groupby('hour').size()
                    ax.plot(threat_counts.index, threat_counts.values, 
                           marker='o', label=attack_type, linewidth=2)
                
                ax.set_title('Threat Detection Timeline', fontsize=16, fontweight='bold')
                ax.set_xlabel('Time', fontsize=12)
                ax.set_ylabel('Number of Threats', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path.replace('.html', '.png'))
                
                return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating threat timeline: {e}")
            return None
    
    def create_attack_distribution_pie(self, attack_data: List[Dict],
                                     save_path: Optional[str] = None) -> Any:
        """Create pie chart of attack type distribution"""
        try:
            if not attack_data:
                return None
                
            df = pd.DataFrame(attack_data)
            attack_counts = df['attack_type'].value_counts()
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure(data=[go.Pie(
                    labels=attack_counts.index,
                    values=attack_counts.values,
                    hole=0.3,
                    marker_colors=self.color_palette[:len(attack_counts)]
                )])
                
                fig.update_layout(
                    title='Attack Type Distribution',
                    title_x=0.5,
                    width=500, height=500
                )
                
                if save_path:
                    fig.write_html(save_path)
                    
                return fig
            else:
                # Matplotlib fallback
                fig, ax = plt.subplots(figsize=(8, 8))
                
                colors = self.color_palette[:len(attack_counts)]
                wedges, texts, autotexts = ax.pie(
                    attack_counts.values, 
                    labels=attack_counts.index,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90
                )
                
                ax.set_title('Attack Type Distribution', fontsize=16, fontweight='bold')
                
                if save_path:
                    plt.savefig(save_path.replace('.html', '.png'))
                
                return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating attack distribution: {e}")
            return None
    
    def create_confidence_heatmap(self, analysis_results: List[Dict],
                                save_path: Optional[str] = None) -> Any:
        """Create heatmap of confidence scores by attack type and time"""
        try:
            if not analysis_results:
                return None
                
            df = pd.DataFrame(analysis_results)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            
            # Create pivot table for heatmap
            if 'attack_type' in df.columns and 'confidence' in df.columns:
                pivot_data = df.pivot_table(
                    index='attack_type',
                    columns='hour',
                    values='confidence',
                    aggfunc='mean'
                ).fillna(0)
                
                if PLOTLY_AVAILABLE:
                    fig = go.Figure(data=go.Heatmap(
                        z=pivot_data.values,
                        x=pivot_data.columns,
                        y=pivot_data.index,
                        colorscale='RdYlBu_r',
                        showscale=True,
                        colorbar=dict(title="Avg Confidence")
                    ))
                    
                    fig.update_layout(
                        title='Confidence Scores Heatmap by Attack Type and Hour',
                        title_x=0.5,
                        xaxis_title='Hour of Day',
                        yaxis_title='Attack Type',
                        width=800, height=500
                    )
                    
                    if save_path:
                        fig.write_html(save_path)
                        
                    return fig
                else:
                    # Matplotlib fallback
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    sns.heatmap(
                        pivot_data, 
                        annot=True, 
                        cmap='RdYlBu_r', 
                        ax=ax,
                        cbar_kws={'label': 'Avg Confidence'}
                    )
                    
                    ax.set_title('Confidence Scores Heatmap by Attack Type and Hour', 
                               fontsize=16, fontweight='bold')
                    ax.set_xlabel('Hour of Day', fontsize=12)
                    ax.set_ylabel('Attack Type', fontsize=12)
                    
                    if save_path:
                        plt.savefig(save_path.replace('.html', '.png'))
                    
                    return self._fig_to_base64(fig)
            else:
                return None
            
        except Exception as e:
            logger.error(f"Error creating confidence heatmap: {e}")
            return None
    
    def create_security_dashboard(self, security_metrics: Dict[str, Any],
                                save_path: Optional[str] = None) -> Any:
        """Create comprehensive security dashboard"""
        try:
            if PLOTLY_AVAILABLE:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[
                        'Threats Detected vs Blocked',
                        'Response Time Distribution', 
                        'False Positive Rate',
                        'System Load Impact'
                    ],
                    specs=[[{"type": "bar"}, {"type": "histogram"}],
                           [{"type": "scatter"}, {"type": "bar"}]]
                )
                
                # Threats detected vs blocked
                threat_stats = security_metrics.get('threat_stats', {})
                fig.add_trace(
                    go.Bar(
                        x=['Detected', 'Blocked', 'Allowed'],
                        y=[
                            threat_stats.get('detected', 0),
                            threat_stats.get('blocked', 0), 
                            threat_stats.get('allowed', 0)
                        ],
                        marker_color=['red', 'green', 'orange']
                    ),
                    row=1, col=1
                )
                
                # Response time distribution
                response_times = security_metrics.get('response_times', [])
                if response_times:
                    fig.add_trace(
                        go.Histogram(x=response_times, nbinsx=20, marker_color='blue'),
                        row=1, col=2
                    )
                
                # False positive rate over time
                fp_data = security_metrics.get('false_positive_timeline', [])
                if fp_data:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(fp_data))),
                            y=fp_data,
                            mode='lines+markers',
                            line=dict(color='red', width=2),
                            name='FP Rate'
                        ),
                        row=2, col=1
                    )
                
                # System load impact
                load_data = security_metrics.get('system_load', {})
                fig.add_trace(
                    go.Bar(
                        x=['CPU', 'Memory', 'Network'],
                        y=[
                            load_data.get('cpu', 0),
                            load_data.get('memory', 0),
                            load_data.get('network', 0)
                        ],
                        marker_color=['lightblue', 'lightgreen', 'lightyellow']
                    ),
                    row=2, col=2
                )
                
                fig.update_layout(
                    title='Security Analysis Dashboard',
                    title_x=0.5,
                    height=800, width=1000,
                    showlegend=False
                )
                
                if save_path:
                    fig.write_html(save_path)
                    
                return fig
            else:
                # Matplotlib fallback
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Security Analysis Dashboard', fontsize=16, fontweight='bold')
                
                # Threats detected vs blocked
                threat_stats = security_metrics.get('threat_stats', {})
                categories = ['Detected', 'Blocked', 'Allowed']
                values = [
                    threat_stats.get('detected', 0),
                    threat_stats.get('blocked', 0), 
                    threat_stats.get('allowed', 0)
                ]
                colors = ['red', 'green', 'orange']
                axes[0, 0].bar(categories, values, color=colors)
                axes[0, 0].set_title('Threats Detected vs Blocked')
                axes[0, 0].set_ylabel('Count')
                
                # Response time distribution
                response_times = security_metrics.get('response_times', [])
                if response_times:
                    axes[0, 1].hist(response_times, bins=20, color='blue', alpha=0.7)
                    axes[0, 1].set_title('Response Time Distribution')
                    axes[0, 1].set_xlabel('Response Time (s)')
                    axes[0, 1].set_ylabel('Frequency')
                
                # False positive rate over time
                fp_data = security_metrics.get('false_positive_timeline', [])
                if fp_data:
                    axes[1, 0].plot(range(len(fp_data)), fp_data, 'ro-', color='red', linewidth=2)
                    axes[1, 0].set_title('False Positive Rate')
                    axes[1, 0].set_xlabel('Time Period')
                    axes[1, 0].set_ylabel('FP Rate')
                
                # System load impact
                load_data = security_metrics.get('system_load', {})
                load_categories = ['CPU', 'Memory', 'Network']
                load_values = [
                    load_data.get('cpu', 0),
                    load_data.get('memory', 0),
                    load_data.get('network', 0)
                ]
                load_colors = ['lightblue', 'lightgreen', 'lightyellow']
                axes[1, 1].bar(load_categories, load_values, color=load_colors)
                axes[1, 1].set_title('System Load Impact')
                axes[1, 1].set_ylabel('Usage %')
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path.replace('.html', '.png'))
                
                return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating security dashboard: {e}")
            return None
    
    def create_model_performance_radar(self, model_metrics: Dict[str, float],
                                     save_path: Optional[str] = None) -> Any:
        """Create radar chart for model performance metrics"""
        try:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
            values = [
                model_metrics.get('accuracy', 0),
                model_metrics.get('precision', 0),
                model_metrics.get('recall', 0),
                model_metrics.get('f1_score', 0),
                model_metrics.get('specificity', 0)
            ]
            
            if PLOTLY_AVAILABLE:
                # Close the radar chart
                values_closed = values + [values[0]]
                metrics_closed = metrics + [metrics[0]]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=metrics_closed,
                    fill='toself',
                    name='Model Performance',
                    line=dict(color='blue'),
                    fillcolor='rgba(0, 100, 255, 0.3)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title='Security Model Performance Metrics',
                    title_x=0.5,
                    width=500, height=500
                )
                
                if save_path:
                    fig.write_html(save_path)
                    
                return fig
            else:
                # Matplotlib fallback - simple bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                
                bars = ax.bar(metrics, values, color=self.color_palette[:len(metrics)])
                ax.set_title('Security Model Performance Metrics', fontsize=16, fontweight='bold')
                ax.set_ylabel('Score', fontsize=12)
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path.replace('.html', '.png'))
                
                return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating performance radar: {e}")
            return None
    
    def generate_security_report(self, comprehensive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive security visualization report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'visualizations': {},
                'summary_stats': {},
                'plotly_available': PLOTLY_AVAILABLE
            }
            
            # Create individual visualizations
            if 'threat_timeline' in comprehensive_data:
                timeline_viz = self.create_threat_timeline(comprehensive_data['threat_timeline'])
                if timeline_viz:
                    report['visualizations']['threat_timeline'] = timeline_viz
            
            if 'attack_distribution' in comprehensive_data:
                distribution_viz = self.create_attack_distribution_pie(comprehensive_data['attack_distribution'])
                if distribution_viz:
                    report['visualizations']['attack_distribution'] = distribution_viz
            
            if 'security_metrics' in comprehensive_data:
                dashboard_viz = self.create_security_dashboard(comprehensive_data['security_metrics'])
                if dashboard_viz:
                    report['visualizations']['security_dashboard'] = dashboard_viz
            
            if 'model_performance' in comprehensive_data:
                performance_viz = self.create_model_performance_radar(comprehensive_data['model_performance'])
                if performance_viz:
                    report['visualizations']['performance_radar'] = performance_viz
            
            # Calculate summary statistics
            if 'threat_timeline' in comprehensive_data:
                threats = comprehensive_data['threat_timeline']
                report['summary_stats'] = {
                    'total_threats': len(threats),
                    'unique_attack_types': len(set(t.get('attack_type', '') for t in threats)),
                    'avg_confidence': np.mean([t.get('confidence', 0) for t in threats]) if threats else 0,
                    'time_range': {
                        'start': min(t.get('timestamp', '') for t in threats) if threats else '',
                        'end': max(t.get('timestamp', '') for t in threats) if threats else ''
                    }
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating security report: {e}")
            return {'error': str(e)}

# Global visualization engine instance
security_viz_engine = SecurityVisualizationEngine() 