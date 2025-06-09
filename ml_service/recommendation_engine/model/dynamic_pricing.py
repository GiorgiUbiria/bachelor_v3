import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class DynamicPricingModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.base_prices = {}
        
    def fit(self, pricing_data):
        """Train dynamic pricing model"""
        try:
            # Prepare features
            features = self._extract_features(pricing_data)
            target = pricing_data['final_price'].values
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train model
            self.model.fit(features_scaled, target)
            
            # Store base prices for reference
            for _, row in pricing_data.iterrows():
                self.base_prices[row['item_id']] = row['base_price']
                
            logger.info(f"Dynamic pricing model trained on {len(pricing_data)} price points")
            return self
            
        except Exception as e:
            logger.error(f"Error training pricing model: {e}")
            raise
            
    def _extract_features(self, data):
        """Extract features for pricing model"""
        features = []
        feature_names = []
        
        # Product features
        if 'base_price' in data.columns:
            features.append(data['base_price'].values)
            feature_names.append('base_price')
            
        if 'category_popularity' in data.columns:
            features.append(data['category_popularity'].values)
            feature_names.append('category_popularity')
            
        if 'stock_level' in data.columns:
            features.append(data['stock_level'].values)
            feature_names.append('stock_level')
            
        # User interest features
        if 'user_interest_score' in data.columns:
            features.append(data['user_interest_score'].values)
            feature_names.append('user_interest_score')
            
        # Market features
        if 'competitor_price' in data.columns:
            features.append(data['competitor_price'].values)
            feature_names.append('competitor_price')
            
        if 'demand_score' in data.columns:
            features.append(data['demand_score'].values)
            feature_names.append('demand_score')
            
        # Time features
        if 'hour_of_day' in data.columns:
            features.append(data['hour_of_day'].values)
            feature_names.append('hour_of_day')
            
        if 'day_of_week' in data.columns:
            features.append(data['day_of_week'].values)
            feature_names.append('day_of_week')
            
        self.feature_names = feature_names
        return np.column_stack(features) if features else np.array([]).reshape(len(data), 0)
        
    def predict_price(self, item_id, user_interest_score=0.5, stock_level=50, 
                     competitor_price=None, demand_score=0.5, hour_of_day=12, day_of_week=1):
        """Predict optimal price for item given context"""
        try:
            if item_id not in self.base_prices:
                return None
                
            base_price = self.base_prices[item_id]
            
            # Create feature vector
            features = []
            if 'base_price' in self.feature_names:
                features.append(base_price)
            if 'category_popularity' in self.feature_names:
                features.append(0.5)  # Default category popularity
            if 'stock_level' in self.feature_names:
                features.append(stock_level)
            if 'user_interest_score' in self.feature_names:
                features.append(user_interest_score)
            if 'competitor_price' in self.feature_names:
                features.append(competitor_price or base_price)
            if 'demand_score' in self.feature_names:
                features.append(demand_score)
            if 'hour_of_day' in self.feature_names:
                features.append(hour_of_day)
            if 'day_of_week' in self.feature_names:
                features.append(day_of_week)
                
            if not features:
                return base_price
                
            # Scale and predict
            features_scaled = self.scaler.transform([features])
            predicted_price = self.model.predict(features_scaled)[0]
            
            # Apply reasonable bounds (50% to 150% of base price)
            min_price = base_price * 0.5
            max_price = base_price * 1.5
            final_price = max(min_price, min(max_price, predicted_price))
            
            return {
                'item_id': item_id,
                'base_price': base_price,
                'predicted_price': float(final_price),
                'discount_percentage': (1 - final_price / base_price) * 100,
                'factors': {
                    'user_interest': user_interest_score,
                    'stock_level': stock_level,
                    'demand_score': demand_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting price for item {item_id}: {e}")
            return None
            
    def get_personalized_deals(self, user_recommendations, max_deals=5):
        """Generate personalized deals based on recommendations"""
        deals = []
        
        for rec in user_recommendations[:max_deals]:
            item_id = rec['item_id']
            user_interest = rec['score']
            
            pricing = self.predict_price(
                item_id=item_id,
                user_interest_score=user_interest,
                demand_score=user_interest
            )
            
            if pricing and pricing['discount_percentage'] > 5:  # Only show deals with >5% discount
                deals.append({
                    'item_id': item_id,
                    'original_price': pricing['base_price'],
                    'deal_price': pricing['predicted_price'],
                    'discount_percentage': pricing['discount_percentage'],
                    'recommendation_score': user_interest
                })
                
        return sorted(deals, key=lambda x: x['discount_percentage'], reverse=True) 