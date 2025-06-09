import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class ClusterVisualizer:
    def __init__(self):
        self.scaler = StandardScaler()
        plt.style.use('seaborn-v0_8')
        
    def visualize_user_clusters(self, user_features_df: pd.DataFrame, 
                               cluster_labels: np.ndarray,
                               method: str = 'tsne',
                               save_path: Optional[str] = None) -> go.Figure:
        try:
            logger.info(f"Creating {method.upper()} visualization of user clusters...")
            
            numeric_features = user_features_df.select_dtypes(include=[np.number])
            features_scaled = self.scaler.fit_transform(numeric_features)
            
            if method.lower() == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                embedding = reducer.fit_transform(features_scaled)
            elif method.lower() == 'umap':
                reducer = umap.UMAP(n_components=2, random_state=42)
                embedding = reducer.fit_transform(features_scaled)
            else:
                reducer = PCA(n_components=2, random_state=42)
                embedding = reducer.fit_transform(features_scaled)
                
            df_plot = pd.DataFrame({
                'x': embedding[:, 0],
                'y': embedding[:, 1],
                'cluster': cluster_labels,
                'user_id': user_features_df.index if hasattr(user_features_df, 'index') else range(len(user_features_df))
            })
            
            fig = px.scatter(
                df_plot, 
                x='x', y='y', 
                color='cluster',
                hover_data=['user_id'],
                title=f'User Clusters Visualization ({method.upper()})',
                labels={'x': f'{method.upper()}_1', 'y': f'{method.upper()}_2'},
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                width=800, height=600,
                showlegend=True,
                title_x=0.5
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cluster visualization: {e}")
            return go.Figure()
    
    def plot_cluster_characteristics(self, cluster_profiles: Dict,
                                   save_path: Optional[str] = None) -> go.Figure:
        try:
            cluster_data = []
            for cluster_id, profile in cluster_profiles.items():
                demographics = profile.get('demographics', {})
                cluster_data.append({
                    'cluster_id': cluster_id,
                    'size': profile.get('size', 0),
                    'avg_age': demographics.get('avg_age', 0),
                    'avg_interactions': demographics.get('avg_interactions', 0),
                    'avg_rating': demographics.get('avg_rating', 0)
                })
                
            df_clusters = pd.DataFrame(cluster_data)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Cluster Sizes', 'Average Age by Cluster', 
                              'Average Interactions', 'Average Rating'],
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            fig.add_trace(
                go.Bar(x=df_clusters['cluster_id'], y=df_clusters['size'], 
                       name='Size', marker_color='lightblue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=df_clusters['cluster_id'], y=df_clusters['avg_age'],
                       name='Avg Age', marker_color='lightgreen'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(x=df_clusters['cluster_id'], y=df_clusters['avg_interactions'],
                       name='Avg Interactions', marker_color='salmon'), 
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(x=df_clusters['cluster_id'], y=df_clusters['avg_rating'],
                       name='Avg Rating', marker_color='gold'),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800, 
                title_text="Cluster Characteristics Analysis",
                title_x=0.5,
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting cluster characteristics: {e}")
            return go.Figure()
    
    def plot_category_preferences_by_cluster(self, cluster_profiles: Dict,
                                           save_path: Optional[str] = None) -> go.Figure:
        try:
            categories_data = []
            
            for cluster_id, profile in cluster_profiles.items():
                top_categories = profile.get('top_categories', {})
                for category, count in top_categories.items():
                    categories_data.append({
                        'cluster_id': f'Cluster {cluster_id}',
                        'category': category,
                        'preference_score': count
                    })
                    
            if not categories_data:
                logger.warning("No category preference data available")
                return go.Figure()
                
            df_categories = pd.DataFrame(categories_data)
            
            pivot_data = df_categories.pivot(
                index='cluster_id', 
                columns='category', 
                values='preference_score'
            ).fillna(0)
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='Viridis',
                showscale=True
            ))
            
            fig.update_layout(
                title='Category Preferences by Cluster',
                title_x=0.5,
                xaxis_title='Product Categories',
                yaxis_title='User Clusters',
                width=800, height=500
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting category preferences: {e}")
            return go.Figure()
    
    def create_cluster_summary_dashboard(self, cluster_profiles: Dict,
                                       user_features_df: pd.DataFrame,
                                       cluster_labels: np.ndarray,
                                       save_path: Optional[str] = None) -> go.Figure:
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Cluster Distribution', 'Age Distribution by Cluster',
                              'Interaction Patterns', 'Rating Patterns'],
                specs=[[{"type": "pie"}, {"type": "box"}],
                       [{"type": "scatter"}, {"type": "violin"}]]
            )
            
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            fig.add_trace(
                go.Pie(labels=[f'Cluster {i}' for i in cluster_counts.index],
                       values=cluster_counts.values,
                       name="Distribution"),
                row=1, col=1
            )
            
            if 'age' in user_features_df.columns:
                for cluster_id in np.unique(cluster_labels):
                    cluster_ages = user_features_df[cluster_labels == cluster_id]['age']
                    fig.add_trace(
                        go.Box(y=cluster_ages, name=f'Cluster {cluster_id}',
                               showlegend=False),
                        row=1, col=2
                    )
            
            if 'total_interactions' in user_features_df.columns and 'avg_rating' in user_features_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=user_features_df['total_interactions'],
                        y=user_features_df['avg_rating'],
                        mode='markers',
                        marker=dict(color=cluster_labels, colorscale='viridis', size=5),
                        name='Users',
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            if 'avg_rating' in user_features_df.columns:
                for cluster_id in np.unique(cluster_labels):
                    cluster_ratings = user_features_df[cluster_labels == cluster_id]['avg_rating']
                    fig.add_trace(
                        go.Violin(y=cluster_ratings, name=f'Cluster {cluster_id}',
                                 showlegend=False),
                        row=2, col=2
                    )
            
            fig.update_layout(
                height=800,
                title_text="Cluster Analysis Dashboard",
                title_x=0.5
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cluster dashboard: {e}")
            return go.Figure() 