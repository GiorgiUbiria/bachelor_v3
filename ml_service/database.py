import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Database connection manager for ML service - matches Go backend schema exactly"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._connect()
    
    def _connect(self):
        """Establish database connection with proper error handling"""
        try:
            # Get database configuration from environment variables (Docker setup)
            db_host = os.getenv('DB_HOST', 'postgres')  # Docker service name
            db_port = os.getenv('DB_PORT', '5432')
            db_user = os.getenv('DB_USER', 'postgres')
            db_password = os.getenv('DB_PASSWORD', 'postgres123')
            db_name = os.getenv('DB_NAME', 'bachelor_db')
            
            # Create database URL
            database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            
            logger.info(f"Attempting to connect to database: {db_host}:{db_port}/{db_name}")
            
            # Create engine with optimized settings
            self.engine = create_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) as total FROM users"))
                user_count = result.fetchone()[0]
                logger.info(f"Database connection successful! Found {user_count} users in database")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            logger.info("This is normal in development. Models will use fallback training data.")
            self.engine = None
            self.SessionLocal = None
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        if self.engine is None:
            return False
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except:
            return False
    
    def get_session(self):
        """Get database session"""
        if not self.is_connected():
            return None
        return self.SessionLocal()
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """Execute SQL query and return results as DataFrame"""
        if not self.is_connected():
            logger.warning("Database not connected, cannot execute query")
            return None
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                if result.returns_rows:
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    logger.info(f"Query executed successfully, returned {len(df)} rows")
                    return df
                return pd.DataFrame()
        except SQLAlchemyError as e:
            logger.error(f"Database query error: {e}")
            return None
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics for all tables"""
        if not self.is_connected():
            return {}
        
        stats = {}
        tables = [
            'users', 'products', 'categories', 'user_events', 
            'favorites', 'comments', 'recommendations', 
            'http_request_logs', 'product_suggestions'
        ]
        
        for table in tables:
            try:
                result = self.execute_query(f"SELECT COUNT(*) as count FROM {table}")
                if result is not None and len(result) > 0:
                    stats[table] = int(result.iloc[0]['count'])
                else:
                    stats[table] = 0
            except Exception as e:
                logger.warning(f"Could not get count for table {table}: {e}")
                stats[table] = 0
        
        return stats
    
    # ===== SECURITY ANALYSIS DATA =====
    def get_http_request_logs(self, limit: int = 5000) -> Optional[pd.DataFrame]:
        """Get HTTP request logs for security analysis training - matches Go HttpRequestLog model"""
        query = """
        SELECT 
            id::text as request_id,
            COALESCE(ip_address, 'unknown') as ip_address,
            COALESCE(user_agent, '') as user_agent,
            COALESCE(path, '/') as path,
            COALESCE(method, 'GET') as method,
            timestamp,
            COALESCE(suspected_attack_type, 'normal') as attack_type,
            COALESCE(attack_score, 0.0) as attack_score,
            COALESCE(status_code, 200) as status_code,
            COALESCE(duration_ms, 0) as duration_ms
        FROM http_request_logs 
        ORDER BY timestamp DESC 
        LIMIT :limit
        """
        return self.execute_query(query, {"limit": limit})
    
    # ===== RECOMMENDATION SYSTEM DATA =====
    def get_users_with_demographics(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Get users with demographics - matches Go User model exactly"""
        query = """
        SELECT 
            id::text as user_id,
            email,
            COALESCE(name, 'User') as name,
            COALESCE(region, 'Other') as region,
            COALESCE(birth_year, 1990) as birth_year,
            (EXTRACT(YEAR FROM CURRENT_DATE) - COALESCE(birth_year, 1990))::int as age,
            CASE 
                WHEN (EXTRACT(YEAR FROM CURRENT_DATE) - COALESCE(birth_year, 1990)) <= 25 THEN 'young'
                WHEN (EXTRACT(YEAR FROM CURRENT_DATE) - COALESCE(birth_year, 1990)) <= 40 THEN 'adult'
                WHEN (EXTRACT(YEAR FROM CURRENT_DATE) - COALESCE(birth_year, 1990)) <= 55 THEN 'middle'
                ELSE 'senior'
            END as age_group,
            is_admin,
            created_at
        FROM users
        ORDER BY created_at DESC
        LIMIT :limit
        """
        return self.execute_query(query, {"limit": limit})
    
    def get_products_with_categories(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Get products with categories - matches Go Product model exactly"""
        query = """
        SELECT 
            p.id::text as product_id,
            p.name,
            COALESCE(p.description, '') as description,
            COALESCE(p.price, 0.0) as price,
            COALESCE(p.tags, ARRAY[]::text[]) as tags,
            COALESCE(c.name, 'uncategorized') as category,
            p.created_at,
            COUNT(DISTINCT f.id) as favorite_count,
            COUNT(DISTINCT cm.id) as comment_count,
            AVG(CASE WHEN cm.sentiment_score IS NOT NULL THEN cm.sentiment_score ELSE 0.0 END) as avg_sentiment
        FROM products p
        LEFT JOIN categories c ON p.category_id = c.id
        LEFT JOIN favorites f ON p.id = f.product_id
        LEFT JOIN comments cm ON p.id = cm.product_id
        GROUP BY p.id, p.name, p.description, p.price, p.tags, c.name, p.created_at
        ORDER BY p.created_at DESC
        LIMIT :limit
        """
        return self.execute_query(query, {"limit": limit})
    
    def get_user_interactions(self, limit: int = 10000) -> Optional[pd.DataFrame]:
        """Get user interactions - matches Go UserEvent model exactly"""
        query = """
        SELECT 
            ue.id::text as interaction_id,
            ue.user_id::text as user_id,
            ue.product_id::text as product_id,
            ue.event_type,
            ue.timestamp,
            u.region,
            u.birth_year,
            (EXTRACT(YEAR FROM CURRENT_DATE) - COALESCE(u.birth_year, 1990))::int as user_age,
            p.name as product_name,
            COALESCE(p.price, 0.0) as product_price,
            COALESCE(c.name, 'uncategorized') as category
        FROM user_events ue
        JOIN users u ON ue.user_id = u.id
        LEFT JOIN products p ON ue.product_id = p.id
        LEFT JOIN categories c ON p.category_id = c.id
        WHERE ue.product_id IS NOT NULL
        ORDER BY ue.timestamp DESC
        LIMIT :limit
        """
        return self.execute_query(query, {"limit": limit})
    
    def get_favorites_data(self, limit: int = 5000) -> Optional[pd.DataFrame]:
        """Get favorites data - matches Go Favorite model exactly"""
        query = """
        SELECT 
            f.id::text as favorite_id,
            f.user_id::text as user_id,
            f.product_id::text as product_id,
            f.favorited_at as timestamp,
            u.region,
            u.birth_year,
            p.name as product_name,
            COALESCE(p.price, 0.0) as product_price,
            COALESCE(c.name, 'uncategorized') as category
        FROM favorites f
        JOIN users u ON f.user_id = u.id
        JOIN products p ON f.product_id = p.id
        LEFT JOIN categories c ON p.category_id = c.id
        ORDER BY f.favorited_at DESC
        LIMIT :limit
        """
        return self.execute_query(query, {"limit": limit})
    
    def get_comments_with_sentiment(self, limit: int = 5000) -> Optional[pd.DataFrame]:
        """Get comments with sentiment analysis - matches Go Comment model exactly"""
        query = """
        SELECT 
            c.id::text as comment_id,
            c.user_id::text as user_id,
            c.product_id::text as product_id,
            c.body,
            c.upvotes,
            c.downvotes,
            COALESCE(c.sentiment_label, 'neutral') as sentiment_label,
            COALESCE(c.sentiment_score, 0.0) as sentiment_score,
            c.created_at,
            u.region,
            p.name as product_name,
            COALESCE(cat.name, 'uncategorized') as category
        FROM comments c
        JOIN users u ON c.user_id = u.id
        JOIN products p ON c.product_id = p.id
        LEFT JOIN categories cat ON p.category_id = cat.id
        ORDER BY c.created_at DESC
        LIMIT :limit
        """
        return self.execute_query(query, {"limit": limit})
    
    # ===== PRODUCT AUTOMATION DATA =====
    def get_products_for_automation(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Get products for automation training - matches Go Product model exactly"""
        query = """
        SELECT 
            p.id::text as product_id,
            p.name,
            COALESCE(p.description, '') as description,
            COALESCE(p.price, 0.0) as price,
            COALESCE(p.tags, ARRAY[]::text[]) as tags,
            COALESCE(c.name, 'uncategorized') as category,
            p.created_at,
            COUNT(DISTINCT f.id) as favorite_count,
            COUNT(DISTINCT cm.id) as comment_count,
            COUNT(DISTINCT ue.id) as interaction_count,
            AVG(CASE WHEN cm.sentiment_score IS NOT NULL THEN cm.sentiment_score ELSE 0.0 END) as avg_sentiment
        FROM products p
        LEFT JOIN categories c ON p.category_id = c.id
        LEFT JOIN favorites f ON p.id = f.product_id
        LEFT JOIN comments cm ON p.id = cm.product_id
        LEFT JOIN user_events ue ON p.id = ue.product_id
        WHERE p.name IS NOT NULL AND p.description IS NOT NULL
        GROUP BY p.id, p.name, p.description, p.price, p.tags, c.name, p.created_at
        ORDER BY interaction_count DESC, p.created_at DESC
        LIMIT :limit
        """
        return self.execute_query(query, {"limit": limit})
    
    def get_product_categories(self) -> Optional[pd.DataFrame]:
        """Get all product categories - matches Go Category model"""
        query = """
        SELECT 
            c.id::text as category_id,
            c.name as category_name,
            COUNT(p.id) as product_count,
            AVG(COALESCE(p.price, 0.0)) as avg_price,
            MIN(COALESCE(p.price, 0.0)) as min_price,
            MAX(COALESCE(p.price, 0.0)) as max_price
        FROM categories c
        LEFT JOIN products p ON c.id = p.category_id
        GROUP BY c.id, c.name
        ORDER BY product_count DESC
        """
        return self.execute_query(query)
    
    # ===== COMBINED DATA FOR ML TRAINING =====
    def get_comprehensive_training_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Get comprehensive training data for all ML models"""
        logger.info("Loading comprehensive training data from database...")
        
        # Get all data needed for ML training
        users_df = self.get_users_with_demographics(limit=2000)
        products_df = self.get_products_with_categories(limit=2000)
        interactions_df = self.get_user_interactions(limit=20000)
        
        if users_df is not None:
            logger.info(f"Loaded {len(users_df)} users from database")
        if products_df is not None:
            logger.info(f"Loaded {len(products_df)} products from database") 
        if interactions_df is not None:
            logger.info(f"Loaded {len(interactions_df)} interactions from database")
        
        return users_df, products_df, interactions_df
    
    # ===== LOGGING METHODS =====
    def log_ml_analysis(self, analysis_type: str, input_data: Dict[str, Any], 
                       output_data: Dict[str, Any], model_version: str = "1.0"):
        """Log ML analysis results for monitoring and improvement"""
        if not self.is_connected():
            logger.warning("Database not connected, cannot log ML analysis")
            return False
        
        try:
            # Create table if it doesn't exist
            create_table_query = """
            CREATE TABLE IF NOT EXISTS ml_analysis_logs (
                id SERIAL PRIMARY KEY,
                analysis_type VARCHAR(100) NOT NULL,
                input_data TEXT,
                output_data TEXT,
                model_version VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(create_table_query))
                conn.commit()
            
            # Insert log entry
            query = """
            INSERT INTO ml_analysis_logs (analysis_type, input_data, output_data, model_version, timestamp)
            VALUES (:analysis_type, :input_data, :output_data, :model_version, :timestamp)
            """
            
            params = {
                "analysis_type": analysis_type,
                "input_data": str(input_data)[:1000],  # Limit size
                "output_data": str(output_data)[:1000],  # Limit size
                "model_version": model_version,
                "timestamp": datetime.now()
            }
            
            with self.engine.connect() as conn:
                conn.execute(text(query), params)
                conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Failed to log ML analysis: {e}")
            return False
    
    def log_security_analysis(self, request_data: Dict, analysis_result: Dict):
        """Log security analysis to http_request_logs table"""
        if not self.is_connected():
            return False
        
        try:
            query = """
            INSERT INTO http_request_logs 
            (ip_address, user_agent, path, method, suspected_attack_type, attack_score, timestamp)
            VALUES (:ip_address, :user_agent, :path, :method, :attack_type, :attack_score, :timestamp)
            """
            
            params = {
                "ip_address": request_data.get('ip_address', 'unknown'),
                "user_agent": request_data.get('user_agent', ''),
                "path": request_data.get('path', '/'),
                "method": request_data.get('method', 'GET'),
                "attack_type": analysis_result.get('suspected_attack_type'),
                "attack_score": analysis_result.get('attack_score', 0.0),
                "timestamp": datetime.now()
            }
            
            with self.engine.connect() as conn:
                conn.execute(text(query), params)
                conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Failed to log security analysis: {e}")
            return False

# Global database connection instance
db_connection = DatabaseConnection()

# Helper function to check database availability
def is_database_available() -> bool:
    """Check if database is available and has data"""
    if not db_connection.is_connected():
        return False
    
    try:
        stats = db_connection.get_database_stats()
        # Check if we have enough data for training
        return (stats.get('users', 0) > 10 and 
                stats.get('products', 0) > 10 and 
                stats.get('user_events', 0) > 50)
    except:
        return False

# Helper function to get training data
def get_training_data_from_database():
    """Get training data from database with proper error handling"""
    if not is_database_available():
        logger.warning("Database not available or insufficient data for training")
        return None, None, None
    
    try:
        return db_connection.get_comprehensive_training_data()
    except Exception as e:
        logger.error(f"Error loading training data from database: {e}")
        return None, None, None 