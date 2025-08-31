"""
Database models for the AI Misinformation Detector
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from app.core.config import settings

Base = declarative_base()


class Claim(Base):
    """Model for storing claims to be fact-checked"""
    __tablename__ = "claims"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    source_url = Column(String(1000), nullable=True)
    classification = Column(String(20), nullable=True)  # True, False, Misleading, Unverified
    reliability_score = Column(Float, nullable=True)
    evidence = Column(JSON, nullable=True)  # Supporting evidence snippets
    processed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


class FactSource(Base):
    """Model for storing verified fact-checking sources"""
    __tablename__ = "fact_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    source_name = Column(String(100), nullable=False)  # e.g., "Snopes", "PolitiFact"
    source_url = Column(String(1000), nullable=False)
    topic = Column(String(100), nullable=True)
    is_verified = Column(Boolean, default=True)
    reliability_rating = Column(Float, default=1.0)  # 0.0 to 1.0
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ProcessingLog(Base):
    """Model for logging processing activities"""
    __tablename__ = "processing_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, nullable=True)
    operation = Column(String(100), nullable=False)  # e.g., "embedding_generation", "classification"
    status = Column(String(20), nullable=False)  # success, error, pending
    details = Column(JSON, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database engine and session
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
