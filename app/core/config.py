"""
Configuration settings for the AI Misinformation Detector
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Database Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "misinformation_detector"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres123"
    
    # Qdrant Vector Database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6335
    qdrant_collection_name: str = "fact_embeddings"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "AI Misinformation Detector API"
    api_version: str = "1.0.0"
    
    # ML Model Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    classification_model: str = "distilbert-base-uncased"
    max_sequence_length: int = 512
    
    # Application Settings
    debug: bool = True
    log_level: str = "INFO"
    batch_size: int = 32
    similarity_threshold: float = 0.7
    
    @property
    def database_url(self) -> str:
        """Get PostgreSQL database URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def qdrant_url(self) -> str:
        """Get Qdrant URL"""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
