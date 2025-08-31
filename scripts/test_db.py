"""
Quick database connectivity test
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.models.database import SessionLocal, create_tables
from qdrant_client import QdrantClient

def test_databases():
    """Test database connections"""
    print(f"Testing AI Misinformation Detector Database Connectivity...")
    print(f"PostgreSQL: {settings.postgres_host}:{settings.postgres_port}")
    print(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    
    # Test PostgreSQL
    print("\n1. Testing PostgreSQL...")
    try:
        db = SessionLocal()
        from sqlalchemy import text
        result = db.execute(text("SELECT version()")).fetchone()
        print(f"✅ PostgreSQL connected successfully")
        print(f"   Version: {result[0][:50]}...")
        
        # Create tables
        create_tables()
        print("✅ Database tables created")
        
        db.close()
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False
    
    # Test Qdrant
    print("\n2. Testing Qdrant...")
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        collections = client.get_collections()
        print(f"✅ Qdrant connected successfully")
        print(f"   Collections: {[c.name for c in collections.collections]}")
        
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False
    
    print("\n🎉 All database tests passed!")
    print("Databases are ready for the application.")
    return True

if __name__ == "__main__":
    test_databases()
