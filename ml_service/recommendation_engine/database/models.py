from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "recommendation_users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    age = Column(Integer)
    gender = Column(String(10))
    region = Column(String(50))
    registration_date = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    total_interactions = Column(Integer, default=0)
    avg_rating = Column(Float, default=0.0)
    cluster_id = Column(Integer)
    
    # Relationships
    interactions = relationship("Interaction", back_populates="user")
    sessions = relationship("Session", back_populates="user")

class Item(Base):
    __tablename__ = "recommendation_items"
    
    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    category = Column(String(100))
    brand = Column(String(100))
    price = Column(Float)
    avg_rating = Column(Float, default=0.0)
    num_reviews = Column(Integer, default=0)
    popularity_score = Column(Float, default=0.0)
    created_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    interactions = relationship("Interaction", back_populates="item")

class Interaction(Base):
    __tablename__ = "recommendation_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("recommendation_users.user_id"), nullable=False)
    item_id = Column(String, ForeignKey("recommendation_items.item_id"), nullable=False)
    interaction_type = Column(String(50))  # view, click, purchase, rating
    rating = Column(Float)
    session_id = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    context_data = Column(Text)  # JSON string for additional context
    
    # Relationships
    user = relationship("User", back_populates="interactions")
    item = relationship("Item", back_populates="interactions")

class Session(Base):
    __tablename__ = "recommendation_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(String, ForeignKey("recommendation_users.user_id"))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    device_type = Column(String(50))
    user_agent = Column(String(500))
    ip_address = Column(String(45))
    page_views = Column(Integer, default=0)
    recommendations_shown = Column(Integer, default=0)
    recommendations_clicked = Column(Integer, default=0)
    purchases = Column(Integer, default=0)
    total_spent = Column(Float, default=0.0)
    
    # Relationships
    user = relationship("User", back_populates="sessions") 