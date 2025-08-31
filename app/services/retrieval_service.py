"""
Hybrid retrieval system combining vector similarity and keyword search
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from loguru import logger

from app.core.config import settings
from app.services.database_service import db_service
from app.services.embedding_service import embedding_service
from app.services.text_processor_simple import text_processor
from app.models.database import FactSource


class RetrievalService:
    """Service for retrieving relevant fact-checking sources"""
    
    def __init__(self):
        pass
    
    def retrieve_evidence(self, claim: str, db: Session, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant evidence using hybrid search (vector + keyword)
        """
        try:
            # Step 1: Generate embedding for the claim
            claim_embedding = embedding_service.generate_embedding(claim)
            
            # Step 2: Vector similarity search in Qdrant
            vector_results = db_service.search_similar_embeddings(
                query_embedding=claim_embedding,
                limit=max_results * 2  # Get more candidates for re-ranking
            )
            
            # Step 3: Keyword-based search in PostgreSQL
            keyword_results = self._keyword_search(claim, db, max_results * 2)
            
            # Step 4: Combine and re-rank results
            combined_results = self._combine_and_rerank(
                claim, vector_results, keyword_results, max_results
            )
            
            logger.info(f"Retrieved {len(combined_results)} evidence sources for claim")
            return combined_results
            
        except Exception as e:
            logger.error(f"Failed to retrieve evidence: {e}")
            return []
    
    def _keyword_search(self, claim: str, db: Session, limit: int) -> List[Dict[str, Any]]:
        """Perform keyword-based search in fact sources"""
        try:
            # Extract keywords from claim
            keywords = text_processor.extract_keywords(claim, max_keywords=5)
            
            if not keywords:
                return []
            
            # Search fact sources using keywords
            fact_sources = db.query(FactSource).all()
            
            keyword_matches = []
            for source in fact_sources:
                # Calculate keyword overlap score
                source_text = (source.title + " " + source.content).lower()
                keyword_score = sum(1 for keyword in keywords if keyword.lower() in source_text)
                
                if keyword_score > 0:
                    keyword_matches.append({
                        "id": source.id,
                        "score": keyword_score / len(keywords),  # Normalize by number of keywords
                        "source": source,
                        "match_type": "keyword"
                    })
            
            # Sort by keyword score
            keyword_matches.sort(key=lambda x: x["score"], reverse=True)
            
            return keyword_matches[:limit]
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _combine_and_rerank(self, claim: str, vector_results: List[Dict], 
                          keyword_results: List[Dict], max_results: int) -> List[Dict[str, Any]]:
        """Combine vector and keyword search results with hybrid ranking"""
        
        # Create a unified result set
        unified_results = {}
        
        # Add vector search results
        for result in vector_results:
            source_id = result["id"]
            unified_results[source_id] = {
                "id": source_id,
                "vector_score": result["score"],
                "keyword_score": 0.0,
                "payload": result["payload"],
                "match_type": "vector"
            }
        
        # Add keyword search results
        for result in keyword_results:
            source_id = result["id"]
            if source_id in unified_results:
                # Combine scores if source found by both methods
                unified_results[source_id]["keyword_score"] = result["score"]
                unified_results[source_id]["match_type"] = "hybrid"
            else:
                # Add new keyword-only result
                unified_results[source_id] = {
                    "id": source_id,
                    "vector_score": 0.0,
                    "keyword_score": result["score"],
                    "source": result["source"],
                    "match_type": "keyword"
                }
        
        # Calculate hybrid scores and format results
        final_results = []
        for source_id, result in unified_results.items():
            # Weighted combination of vector and keyword scores
            hybrid_score = (0.7 * result["vector_score"]) + (0.3 * result["keyword_score"])
            
            # Get source information
            if "source" in result:
                source = result["source"]
                title = source.title if hasattr(source, 'title') else 'Unknown'
                content = source.content if hasattr(source, 'content') else ''
                source_name = source.source_name if hasattr(source, 'source_name') else 'Unknown'
                source_url = source.source_url if hasattr(source, 'source_url') else ''
            else:
                # Get from payload metadata
                payload = result.get("payload", {})
                title = payload.get('title', 'Unknown')
                content = payload.get('content', '')
                source_name = payload.get('source_name', 'Unknown')
                source_url = payload.get('source_url', '')
            
            final_results.append({
                "source_id": source_id,
                "hybrid_score": hybrid_score,
                "vector_score": result["vector_score"],
                "keyword_score": result["keyword_score"],
                "match_type": result["match_type"],
                "title": title,
                "content": content,
                "source_name": source_name,
                "source_url": source_url
            })
        
        # Sort by hybrid score
        final_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Return top results
        return final_results[:max_results]
    
    def extract_evidence_snippets(self, claim: str, retrieved_sources: List[Dict[str, Any]], 
                                max_snippet_length: int = 200) -> List[str]:
        """Extract relevant snippets from retrieved sources"""
        evidence_snippets = []
        
        claim_keywords = text_processor.extract_keywords(claim, max_keywords=3)
        
        for source in retrieved_sources:
            content = source.get("content", "")
            if not content:
                continue
            
            # Extract sentences from source content
            sentences = text_processor.extract_sentences(content)
            
            # Find sentences that contain claim keywords
            relevant_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                keyword_matches = sum(1 for keyword in claim_keywords if keyword.lower() in sentence_lower)
                
                if keyword_matches > 0:
                    relevant_sentences.append((sentence, keyword_matches))
            
            # Sort by keyword matches and take best sentences
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Create snippet from best sentences
            if relevant_sentences:
                snippet_text = relevant_sentences[0][0]  # Best matching sentence
                
                # Truncate if too long
                if len(snippet_text) > max_snippet_length:
                    snippet_text = snippet_text[:max_snippet_length] + "..."
                
                evidence_snippets.append(snippet_text)
        
        return evidence_snippets
    
    def search_by_topic(self, topic: str, db: Session, limit: int = 10) -> List[FactSource]:
        """Search fact sources by topic"""
        return db_service.get_fact_sources(db, topic=topic, limit=limit)
    
    def get_source_reliability(self, source_name: str) -> float:
        """Get reliability rating for a fact-checking source"""
        # Default reliability ratings for common fact-checking sources
        reliability_ratings = {
            "snopes": 0.95,
            "politifact": 0.90,
            "factcheck.org": 0.88,
            "reuters": 0.85,
            "ap news": 0.85,
            "bbc": 0.82,
            "cnn": 0.75,
            "fox news": 0.70,
            "unknown": 0.50
        }
        
        source_lower = source_name.lower()
        for known_source, rating in reliability_ratings.items():
            if known_source in source_lower:
                return rating
        
        return reliability_ratings["unknown"]


# Global retrieval service instance
retrieval_service = RetrievalService()
