"""
Database service for managing PostgreSQL and Qdrant connections
"""
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from loguru import logger

from app.core.config import settings
from app.models.database import get_db, create_tables, Claim, FactSource, ProcessingLog


class DatabaseService:
    """Service for managing database operations"""
    
    def __init__(self):
        self.qdrant_client = None
        self._initialize_qdrant()
        create_tables()  # Ensure PostgreSQL tables exist
    
    def _initialize_qdrant(self):
        """Initialize Qdrant client and create collection if needed"""
        try:
            self.qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
            
            # Check if collection exists, create if not
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if settings.qdrant_collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=settings.qdrant_collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # all-MiniLM-L6-v2 size
                )
                logger.info(f"Created Qdrant collection: {settings.qdrant_collection_name}")
            else:
                logger.info(f"Qdrant collection already exists: {settings.qdrant_collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            self.qdrant_client = None
    
    def store_embedding(self, claim_id: int, embedding: List[float], metadata: Dict[str, Any] = None):
        """Store embedding in Qdrant vector database"""
        if not self.qdrant_client:
            raise Exception("Qdrant client not initialized")
        
        if metadata is None:
            metadata = {}
        
        point = PointStruct(
            id=claim_id,
            vector=embedding,
            payload=metadata
        )
        
        self.qdrant_client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=[point]
        )
        logger.info(f"Stored embedding for claim {claim_id}")
    
    def search_similar_embeddings(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        """Search for similar embeddings in Qdrant"""
        if not self.qdrant_client:
            raise Exception("Qdrant client not initialized")
        
        search_result = self.qdrant_client.search(
            collection_name=settings.qdrant_collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=settings.similarity_threshold
        )
        
        results = []
        for hit in search_result:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
        
        return results
    
    def store_claim(self, db: Session, claim_text: str, source_url: Optional[str] = None) -> Claim:
        """Store a new claim in PostgreSQL"""
        claim = Claim(text=claim_text, source_url=source_url)
        db.add(claim)
        db.commit()
        db.refresh(claim)
        logger.info(f"Stored claim with ID: {claim.id}")
        return claim
    
    def update_claim_results(self, db: Session, claim_id: int, classification: str, 
                           reliability_score: float, evidence: List[str]):
        """Update claim with analysis results"""
        claim = db.query(Claim).filter(Claim.id == claim_id).first()
        if claim:
            claim.classification = classification
            claim.reliability_score = reliability_score
            claim.evidence = evidence
            claim.processed_at = datetime.utcnow()
            db.commit()
            logger.info(f"Updated claim {claim_id} with results")
        else:
            logger.error(f"Claim {claim_id} not found")
    
    def get_fact_sources(self, db: Session, topic: Optional[str] = None, limit: int = 100) -> List[FactSource]:
        """Retrieve fact-checking sources, optionally filtered by topic"""
        query = db.query(FactSource)
        if topic:
            query = query.filter(FactSource.topic == topic)
        return query.limit(limit).all()
    
    def add_fact_source(self, db: Session, title: str, content: str, source_name: str, 
                       source_url: str, topic: Optional[str] = None) -> FactSource:
        """Add a new fact-checking source"""
        fact_source = FactSource(
            title=title,
            content=content,
            source_name=source_name,
            source_url=source_url,
            topic=topic
        )
        db.add(fact_source)
        db.commit()
        db.refresh(fact_source)
        logger.info(f"Added fact source: {title}")
        return fact_source


# Global database service instance
db_service = DatabaseService()
