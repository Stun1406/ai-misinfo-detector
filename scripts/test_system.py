"""
Simple test script to verify the system works end-to-end
"""
import sys
import os
import time
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.models.database import SessionLocal, create_tables
from app.services.analysis_service import analysis_service
from app.services.database_service import db_service
from loguru import logger


async def test_basic_functionality():
    """Test basic system functionality"""
    logger.info("Starting basic system test...")
    
    # Test 1: Database connectivity
    logger.info("Testing database connectivity...")
    try:
        db = SessionLocal()
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        logger.info("✅ PostgreSQL connection successful")
        db.close()
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        return False
    
    # Test 2: Qdrant connectivity
    logger.info("Testing Qdrant connectivity...")
    try:
        if db_service.qdrant_client:
            collections = db_service.qdrant_client.get_collections()
            logger.info("✅ Qdrant connection successful")
        else:
            logger.error("❌ Qdrant client not initialized")
            return False
    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {e}")
        return False
    
    # Test 3: Create tables
    logger.info("Creating database tables...")
    try:
        create_tables()
        logger.info("✅ Database tables created")
    except Exception as e:
        logger.error(f"❌ Table creation failed: {e}")
        return False
    
    # Test 4: Analyze a simple claim
    logger.info("Testing claim analysis...")
    try:
        db = SessionLocal()
        
        test_claim = "The sky is blue because of light scattering"
        result = await analysis_service.analyze_claim(test_claim, None, db)
        
        logger.info(f"✅ Claim analysis successful:")
        logger.info(f"   Classification: {result['classification']}")
        logger.info(f"   Reliability: {result['reliability']}%")
        logger.info(f"   Processing time: {result['processing_time_ms']}ms")
        
        db.close()
        
    except Exception as e:
        logger.error(f"❌ Claim analysis failed: {e}")
        return False
    
    logger.info("🎉 All tests passed! System is working correctly.")
    return True


def test_database_setup():
    """Test database setup without ML components"""
    logger.info("Testing database setup...")
    
    try:
        # Test PostgreSQL
        db = SessionLocal()
        create_tables()
        
        # Test adding a fact source
        from app.models.database import FactSource
        test_source = FactSource(
            title="Test Fact Source",
            content="This is a test fact for system verification.",
            source_name="Test System",
            source_url="https://test.example.com",
            topic="test"
        )
        
        db.add(test_source)
        db.commit()
        db.refresh(test_source)
        
        logger.info(f"✅ Added test fact source with ID: {test_source.id}")
        
        # Clean up
        db.delete(test_source)
        db.commit()
        db.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup test failed: {e}")
        return False


def main():
    """Main test function"""
    logger.info("🧪 Starting AI Misinformation Detector System Tests")
    logger.info(f"Configuration: PostgreSQL on port {settings.postgres_port}, Qdrant on port {settings.qdrant_port}")
    
    # Test 1: Database setup
    if not test_database_setup():
        logger.error("Database setup test failed. Check your database connections.")
        return
    
    # Test 2: Full system test
    try:
        success = asyncio.run(test_basic_functionality())
        if success:
            logger.info("🎉 All system tests completed successfully!")
            logger.info("You can now start the application with: python scripts/start_app.py")
        else:
            logger.error("❌ System tests failed. Check the logs above.")
    except Exception as e:
        logger.error(f"❌ System test failed with exception: {e}")


if __name__ == "__main__":
    main()
