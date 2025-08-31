"""
Startup script for the AI Misinformation Detector
"""
import sys
import os
import asyncio
import uvicorn
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.models.database import create_tables
from loguru import logger


def setup_logging():
    """Configure logging"""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="10 days",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )


def check_dependencies():
    """Check if required services are available"""
    logger.info("Checking dependencies...")
    
    # Check if databases are running (for Docker setup)
    # This is optional - services will attempt to connect when needed
    try:
        import psycopg2
        logger.info("PostgreSQL driver available")
    except ImportError as e:
        logger.error(f"PostgreSQL driver not available: {e}")
        return False
    
    try:
        from qdrant_client import QdrantClient
        logger.info("Qdrant client available")
    except ImportError as e:
        logger.error(f"Qdrant client not available: {e}")
        return False
    
    return True


def initialize_app():
    """Initialize the application"""
    logger.info("Initializing AI Misinformation Detector...")
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Setup logging
    setup_logging()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        return False
    
    # Create database tables
    try:
        logger.info("Creating database tables...")
        create_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        logger.info("This is normal if databases are not yet running. Start databases first.")
    
    return True


def main():
    """Main entry point"""
    if not initialize_app():
        logger.error("Application initialization failed")
        sys.exit(1)
    
    logger.info(f"Starting AI Misinformation Detector API server...")
    logger.info(f"Server will run at: http://{settings.api_host}:{settings.api_port}")
    logger.info("API documentation available at: /docs")
    logger.info("Web interface available at: /")
    
    # Start the FastAPI server
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
