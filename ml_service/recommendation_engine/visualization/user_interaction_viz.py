import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import logging
from operator import attrgetter

logger = logging.getLogger(__name__)

class UserInteractionVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
    
    def plot_user_behavior_patterns(self, interactions_df: pd.DataFrame,
                                   save_path: Optional[str] = None) -> go.Figure:
        try:
            logger.info("Creating user behavior patterns visualization...")
            
            interactions_df = interactions_df.copy()
            interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
            
            interactions_df['hour'] = interactions_df['timestamp'].dt.hour
            interactions_df['day_of_week'] = interactions_df['timestamp'].dt.day_name()
            interactions_df['month'] = interactions_df['timestamp'].dt.month_name()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Interactions by Hour', 'Interactions by Day of Week',
                              'Average Rating by Hour', 'Interaction Types Distribution'],
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "pie"}]]
            )
            
            hourly_counts = interactions_df['hour'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=hourly_counts.index, y=hourly_counts.values,
                       marker_color='lightblue'),
                row=1, col=1
            )
            
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_counts = interactions_df['day_of_week'].value_counts().reindex(day_order)
            fig.add_trace(
                go.Bar(x=daily_counts.index, y=daily_counts.values,
                       marker_color='lightgreen'),
                row=1, col=2
            )
            
            if 'rating' in interactions_df.columns:
                hourly_ratings = interactions_df.groupby('hour')['rating'].mean()
                fig.add_trace(
                    go.Scatter(x=hourly_ratings.index, y=hourly_ratings.values,
                              mode='lines+markers', line=dict(color='red', width=3)),
                    row=2, col=1
                )
            
            if 'interaction_type' in interactions_df.columns:
                type_counts = interactions_df['interaction_type'].value_counts()
                fig.add_trace(
                    go.Pie(labels=type_counts.index, values=type_counts.values),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='User Behavior Patterns Analysis',
                title_x=0.5,
                height=800, width=1000,
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating behavior patterns: {e}")
            return go.Figure()
    
    def plot_user_journey_funnel(self, session_data: pd.DataFrame,
                                save_path: Optional[str] = None) -> go.Figure:
        try:
            logger.info("Creating user journey funnel...")
            
            funnel_stages = []
            
            if 'page_views' in session_data.columns:
                total_sessions = len(session_data)
                funnel_stages.append(('Sessions Started', total_sessions))
                
                viewed_products = len(session_data[session_data['page_views'] > 0])
                funnel_stages.append(('Viewed Products', viewed_products))
                
            if 'recommendations_shown' in session_data.columns:
                saw_recommendations = len(session_data[session_data['recommendations_shown'] > 0])
                funnel_stages.append(('Saw Recommendations', saw_recommendations))
                
            if 'recommendations_clicked' in session_data.columns:
                clicked_recommendations = len(session_data[session_data['recommendations_clicked'] > 0])
                funnel_stages.append(('Clicked Recommendations', clicked_recommendations))
                
            if 'purchases' in session_data.columns:
                made_purchases = len(session_data[session_data['purchases'] > 0])
                funnel_stages.append(('Made Purchases', made_purchases))
            
            if not funnel_stages:
                logger.warning("No funnel data available")
                return go.Figure()
            
            stages, values = zip(*funnel_stages)
            
            fig = go.Figure(go.Funnel(
                y=stages,
                x=values,
                textinfo="value+percent initial",
                marker=dict(color=["deepskyblue", "lightsalmon", "tan", "teal", "silver"])
            ))
            
            fig.update_layout(
                title='User Journey Conversion Funnel',
                title_x=0.5,
                width=600, height=500
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating funnel: {e}")
            return go.Figure()
    
    def plot_user_engagement_metrics(self, user_stats: pd.DataFrame,
                                   save_path: Optional[str] = None) -> go.Figure:
        try:
            logger.info("Creating user engagement metrics...")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Session Duration Distribution', 'Pages per Session',
                              'Return User Analysis', 'Engagement Score Distribution']
            )
            
            if 'session_duration' in user_stats.columns:
                fig.add_trace(
                    go.Histogram(x=user_stats['session_duration'], nbinsx=30,
                               marker_color='lightblue', opacity=0.7),
                    row=1, col=1
                )
            
            if 'avg_pages_per_session' in user_stats.columns:
                fig.add_trace(
                    go.Histogram(x=user_stats['avg_pages_per_session'], nbinsx=20,
                               marker_color='lightgreen', opacity=0.7),
                    row=1, col=2
                )
            
            if 'total_sessions' in user_stats.columns:
                return_users = user_stats['total_sessions'] > 1
                return_counts = return_users.value_counts()
                
                fig.add_trace(
                    go.Bar(x=['New Users', 'Return Users'], 
                          y=[return_counts.get(False, 0), return_counts.get(True, 0)],
                          marker_color=['salmon', 'lightcoral']),
                    row=2, col=1
                )
            
            if all(col in user_stats.columns for col in ['total_interactions', 'avg_rating', 'total_sessions']):
                engagement_score = (
                    user_stats['total_interactions'] * 0.4 +
                    user_stats['avg_rating'] * 10 * 0.3 +
                    user_stats['total_sessions'] * 0.3
                )
                
                fig.add_trace(
                    go.Histogram(x=engagement_score, nbinsx=25,
                               marker_color='gold', opacity=0.7),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='User Engagement Metrics Dashboard',
                title_x=0.5,
                height=800, width=1000,
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating engagement metrics: {e}")
            return go.Figure()
    
    def plot_cohort_analysis(self, interactions_df: pd.DataFrame,
                           save_path: Optional[str] = None) -> go.Figure:
        try:
            logger.info("Creating cohort analysis...")
            
            interactions_df = interactions_df.copy()
            interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
            interactions_df['period'] = interactions_df['timestamp'].dt.to_period('M')
            
            user_first_month = interactions_df.groupby('user_id')['period'].min().reset_index()
            user_first_month.columns = ['user_id', 'cohort_group']
            
            df_cohort = interactions_df.merge(user_first_month, on='user_id')
            df_cohort['period_number'] = (df_cohort['period'] - df_cohort['cohort_group']).apply(attrgetter('n'))
            
            cohort_data = df_cohort.groupby(['cohort_group', 'period_number'])['user_id'].nunique().reset_index()
            cohort_counts = cohort_data.pivot(index='cohort_group', 
                                            columns='period_number', 
                                            values='user_id')
            
            cohort_sizes = user_first_month.groupby('cohort_group')['user_id'].nunique()
            retention_table = cohort_counts.divide(cohort_sizes, axis=0)
            
            fig = go.Figure(data=go.Heatmap(
                z=retention_table.values,
                x=[f'Period {i}' for i in retention_table.columns],
                y=[str(idx) for idx in retention_table.index],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Retention Rate")
            ))
            
            fig.update_layout(
                title='User Cohort Retention Analysis',
                title_x=0.5,
                xaxis_title='Period Number',
                yaxis_title='Cohort Month',
                width=800, height=500
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cohort analysis: {e}")
            return go.Figure() 