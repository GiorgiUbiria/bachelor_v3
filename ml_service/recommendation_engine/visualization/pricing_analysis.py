import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PricingAnalyzer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
    
    def plot_pricing_distribution(self, pricing_data: pd.DataFrame,
                                 save_path: Optional[str] = None) -> go.Figure:
        try:
            logger.info("Creating pricing distribution analysis...")
            
            if 'base_price' in pricing_data.columns and 'final_price' in pricing_data.columns:
                pricing_data = pricing_data.copy()
                pricing_data['discount_pct'] = (
                    (pricing_data['base_price'] - pricing_data['final_price']) / 
                    pricing_data['base_price'] * 100
                )
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Base Price Distribution', 'Final Price Distribution',
                                  'Discount Distribution', 'Price vs Discount'],
                    specs=[[{"type": "histogram"}, {"type": "histogram"}],
                           [{"type": "histogram"}, {"type": "scatter"}]]
                )
                
                fig.add_trace(
                    go.Histogram(x=pricing_data['base_price'], name='Base Price',
                               nbinsx=30, opacity=0.7, marker_color='lightblue'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Histogram(x=pricing_data['final_price'], name='Final Price',
                               nbinsx=30, opacity=0.7, marker_color='lightgreen'),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Histogram(x=pricing_data['discount_pct'], name='Discount %',
                               nbinsx=30, opacity=0.7, marker_color='salmon'),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pricing_data['base_price'],
                        y=pricing_data['discount_pct'],
                        mode='markers',
                        name='Price vs Discount',
                        marker=dict(color='purple', size=5, opacity=0.6)
                    ),
                    row=2, col=2
                )
                
                fig.update_layout(
                    title='Dynamic Pricing Analysis Dashboard',
                    title_x=0.5,
                    height=800, width=1000,
                    showlegend=False
                )
                
                if save_path:
                    fig.write_html(save_path)
                    
                return fig
            else:
                logger.warning("Required pricing columns not found")
                return go.Figure()
                
        except Exception as e:
            logger.error(f"Error creating pricing distribution: {e}")
            return go.Figure()
    
    def plot_pricing_by_factors(self, pricing_data: pd.DataFrame,
                               save_path: Optional[str] = None) -> go.Figure:
        try:
            logger.info("Analyzing pricing by factors...")
            
            if not all(col in pricing_data.columns for col in ['user_interest_score', 'stock_level', 'demand_score', 'final_price']):
                logger.warning("Required columns for factor analysis not found")
                return go.Figure()
                
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Price vs User Interest', 'Price vs Stock Level',
                              'Price vs Demand Score', 'Multi-factor Analysis'],
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter3d"}]]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pricing_data['user_interest_score'],
                    y=pricing_data['final_price'],
                    mode='markers',
                    name='User Interest',
                    marker=dict(color='blue', size=5, opacity=0.6)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pricing_data['stock_level'],
                    y=pricing_data['final_price'],
                    mode='markers',
                    name='Stock Level',
                    marker=dict(color='green', size=5, opacity=0.6)
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pricing_data['demand_score'],
                    y=pricing_data['final_price'],
                    mode='markers',
                    name='Demand Score',
                    marker=dict(color='red', size=5, opacity=0.6)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter3d(
                    x=pricing_data['user_interest_score'],
                    y=pricing_data['demand_score'],
                    z=pricing_data['final_price'],
                    mode='markers',
                    name='Multi-factor',
                    marker=dict(
                        color=pricing_data['stock_level'],
                        colorscale='Viridis',
                        size=4,
                        opacity=0.8,
                        colorbar=dict(title="Stock Level")
                    )
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Pricing Factors Analysis',
                title_x=0.5,
                height=800, width=1000,
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating factor analysis: {e}")
            return go.Figure()
    
    def plot_revenue_impact(self, pricing_history: pd.DataFrame,
                           save_path: Optional[str] = None) -> go.Figure:
        try:
            logger.info("Creating revenue impact analysis...")
            
            if 'timestamp' not in pricing_history.columns:
                logger.warning("Timestamp column required for revenue analysis")
                return go.Figure()
                
            pricing_history = pricing_history.copy()
            pricing_history['timestamp'] = pd.to_datetime(pricing_history['timestamp'])
            
            daily_metrics = pricing_history.groupby(pricing_history['timestamp'].dt.date).agg({
                'final_price': ['sum', 'mean', 'count'],
                'base_price': 'sum',
                'user_interest_score': 'mean',
                'demand_score': 'mean'
            }).reset_index()
            
            daily_metrics.columns = [
                'date', 'total_revenue', 'avg_price', 'transactions',
                'base_revenue', 'avg_interest', 'avg_demand'
            ]
            
            daily_metrics['revenue_uplift'] = (
                (daily_metrics['total_revenue'] - daily_metrics['base_revenue']) /
                daily_metrics['base_revenue'] * 100
            )
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Revenue Over Time', 'Average Price Trends',
                              'Transaction Volume', 'Revenue Uplift %'],
                specs=[[{"secondary_y": True}, {"secondary_y": True}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=daily_metrics['date'],
                    y=daily_metrics['total_revenue'],
                    mode='lines+markers',
                    name='Dynamic Revenue',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=daily_metrics['date'],
                    y=daily_metrics['base_revenue'],
                    mode='lines+markers',
                    name='Base Revenue',
                    line=dict(color='gray', width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=daily_metrics['date'],
                    y=daily_metrics['avg_price'],
                    mode='lines+markers',
                    name='Avg Price',
                    line=dict(color='green', width=2)
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=daily_metrics['date'],
                    y=daily_metrics['transactions'],
                    name='Transactions',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=daily_metrics['date'],
                    y=daily_metrics['revenue_uplift'],
                    mode='lines+markers',
                    name='Revenue Uplift %',
                    line=dict(color='red', width=2),
                    fill='tonexty'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Revenue Impact Analysis Over Time',
                title_x=0.5,
                height=800, width=1200,
                showlegend=True
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating revenue impact analysis: {e}")
            return go.Figure()
    
    def create_pricing_strategy_comparison(self, strategy_results: Dict[str, pd.DataFrame],
                                         save_path: Optional[str] = None) -> go.Figure:
        try:
            logger.info("Creating pricing strategy comparison...")
            
            comparison_data = []
            
            for strategy_name, results_df in strategy_results.items():
                if 'final_price' in results_df.columns and 'base_price' in results_df.columns:
                    avg_price = results_df['final_price'].mean()
                    avg_discount = ((results_df['base_price'] - results_df['final_price']) / 
                                  results_df['base_price'] * 100).mean()
                    total_revenue = results_df['final_price'].sum()
                    
                    comparison_data.append({
                        'strategy': strategy_name,
                        'avg_price': avg_price,
                        'avg_discount': avg_discount,
                        'total_revenue': total_revenue,
                        'num_transactions': len(results_df)
                    })
            
            if not comparison_data:
                logger.warning("No strategy comparison data available")
                return go.Figure()
                
            df_comparison = pd.DataFrame(comparison_data)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Average Price by Strategy', 'Average Discount by Strategy',
                              'Total Revenue by Strategy', 'Transactions by Strategy'],
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            fig.add_trace(
                go.Bar(x=df_comparison['strategy'], y=df_comparison['avg_price'],
                       name='Avg Price', marker_color='lightblue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=df_comparison['strategy'], y=df_comparison['avg_discount'],
                       name='Avg Discount %', marker_color='lightgreen'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(x=df_comparison['strategy'], y=df_comparison['total_revenue'],
                       name='Total Revenue', marker_color='salmon'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(x=df_comparison['strategy'], y=df_comparison['num_transactions'],
                       name='Transactions', marker_color='gold'),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Pricing Strategy Comparison',
                title_x=0.5,
                height=600, width=800,
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating strategy comparison: {e}")
            return go.Figure() 