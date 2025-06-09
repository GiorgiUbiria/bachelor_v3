import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._connect()
    def _connect(self):
        try:
            db_host = os.getenv('DB_HOST', 'postgres')
            db_port = os.getenv('DB_PORT', '5432')
            db_user = os.getenv('DB_USER', 'postgres')
            db_password = os.getenv('DB_PASSWORD', 'postgres123')
            db_name = os.getenv('DB_NAME', 'bachelor_db')
            database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            logger.info(f"Attempting to connect to database: {db_host}:{db_port}/{db_name}")
            self.engine = create_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False
            )
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) as total FROM users"))
                user_count = result.fetchone()[0]
                logger.info(f"Database connection successful! Found {user_count} users in database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            logger.info("This is normal in development. Models will use fallback training data.")
            self.engine = None
            self.SessionLocal = None 