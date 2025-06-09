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

class RecommendationMapper:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        
    def create_user_item_heatmap(self, interactions_df: pd.DataFrame,
                                max_users: int = 50, max_items: int = 50,
                                save_path: Optional[str] = None) -> go.Figure:
        try:
            logger.info("Creating user-item interaction heatmap...")
            
            top_users = interactions_df['user_id'].value_counts().head(max_users).index
            top_items = interactions_df['item_id'].value_counts().head(max_items).index
            
            filtered_df = interactions_df[
                (interactions_df['user_id'].isin(top_users)) &
                (interactions_df['item_id'].isin(top_items))
            ]
            
            interaction_matrix = filtered_df.pivot_table(
                index='user_id',
                columns='item_id', 
                values='rating',
                fill_value=0
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=interaction_matrix.values,
                x=[f'Item_{i}' for i in range(len(interaction_matrix.columns))],
                y=[f'User_{i}' for i in range(len(interaction_matrix.index))],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Rating")
            ))
            
            fig.update_layout(
                title='User-Item Interaction Matrix',
                title_x=0.5,
                xaxis_title='Items',
                yaxis_title='Users',
                width=800, height=600
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating user-item heatmap: {e}")
            return go.Figure()
    
    def plot_recommendation_overlap(self, recommendation_results: Dict[str, List],
                                  save_path: Optional[str] = None) -> go.Figure:
        try:
            logger.info("Creating recommendation overlap analysis...")
            
            methods = list(recommendation_results.keys())
            n_methods = len(methods)
            
            overlap_matrix = np.zeros((n_methods, n_methods))
            
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i == j:
                        overlap_matrix[i, j] = 1.0
                    else:
                        recs1 = set(recommendation_results[method1])
                        recs2 = set(recommendation_results[method2])
                        
                        if len(recs1) > 0 and len(recs2) > 0:
                            overlap = len(recs1.intersection(recs2)) / len(recs1.union(recs2))
                            overlap_matrix[i, j] = overlap
            
            fig = go.Figure(data=go.Heatmap(
                z=overlap_matrix,
                x=methods,
                y=methods,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Jaccard Similarity"),
                text=np.round(overlap_matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 12}
            ))
            
            fig.update_layout(
                title='Recommendation Method Overlap Analysis',
                title_x=0.5,
                xaxis_title='Methods',
                yaxis_title='Methods',
                width=600, height=500
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating overlap analysis: {e}")
            return go.Figure()
    
    def plot_recommendation_diversity(self, recommendations_by_user: Dict[str, List],
                                    items_df: pd.DataFrame,
                                    save_path: Optional[str] = None) -> go.Figure:
        try:
            logger.info("Analyzing recommendation diversity...")
            
            diversity_data = []
            
            for user_id, recommendations in recommendations_by_user.items():
                if not recommendations:
                    continue
                    
                rec_items_df = items_df[items_df['item_id'].isin(recommendations)]
                
                if rec_items_df.empty:
                    continue
                    
                categories = rec_items_df['category'].tolist()
                unique_categories = len(set(categories))
                total_categories = len(categories)
                category_diversity = unique_categories / total_categories if total_categories > 0 else 0
                
                price_diversity = 0
                if 'price' in rec_items_df.columns:
                    prices = rec_items_df['price'].dropna()
                    if len(prices) > 1:
                        price_std = prices.std()
                        price_mean = prices.mean()
                        price_diversity = price_std / price_mean if price_mean > 0 else 0
                
                diversity_data.append({
                    'user_id': user_id,
                    'category_diversity': category_diversity,
                    'price_diversity': price_diversity,
                    'num_recommendations': len(recommendations)
                })
            
            if not diversity_data:
                logger.warning("No diversity data available")
                return go.Figure()
                
            df_diversity = pd.DataFrame(diversity_data)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Category Diversity Distribution', 'Price Diversity Distribution']
            )
            
            fig.add_trace(
                go.Histogram(x=df_diversity['category_diversity'], 
                           name='Category Diversity',
                           nbinsx=20, opacity=0.7),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=df_diversity['price_diversity'],
                           name='Price Diversity', 
                           nbinsx=20, opacity=0.7),
                row=1, col=2
            )
            
            fig.update_layout(
                title='Recommendation Diversity Analysis',
                title_x=0.5,
                height=400, width=800,
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating diversity analysis: {e}")
            return go.Figure()
    
    def create_recommendation_coverage_map(self, all_recommendations: List[List[str]],
                                         items_df: pd.DataFrame,
                                         save_path: Optional[str] = None) -> go.Figure:
        try:
            logger.info("Creating recommendation coverage map...")
            
            recommended_items = set()
            for recs in all_recommendations:
                recommended_items.update(recs)
                
            category_coverage = {}
            for category in items_df['category'].unique():
                category_items = set(items_df[items_df['category'] == category]['item_id'])
                recommended_in_category = recommended_items.intersection(category_items)
                
                coverage = len(recommended_in_category) / len(category_items) if category_items else 0
                category_coverage[category] = {
                    'coverage': coverage,
                    'total_items': len(category_items),
                    'recommended_items': len(recommended_in_category)
                }
            
            categories = list(category_coverage.keys())
            coverages = [category_coverage[cat]['coverage'] for cat in categories]
            total_items = [category_coverage[cat]['total_items'] for cat in categories]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=categories,
                y=coverages,
                name='Coverage %',
                marker_color='skyblue',
                text=[f'{c:.1%}' for c in coverages],
                textposition='outside'
            ))
            
            fig.add_trace(go.Scatter(
                x=categories,
                y=total_items,
                mode='markers+lines',
                name='Total Items',
                yaxis='y2',
                marker=dict(color='red', size=8),
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title='Recommendation Coverage by Category',
                title_x=0.5,
                xaxis_title='Product Categories',
                yaxis_title='Coverage Percentage',
                yaxis2=dict(
                    title='Total Items in Category',
                    overlaying='y',
                    side='right'
                ),
                height=500, width=800,
                legend=dict(x=0.02, y=0.98)
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating coverage map: {e}")
            return go.Figure() 