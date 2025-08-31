"""
Main analysis service that orchestrates the complete misinformation detection pipeline
"""
import time
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from loguru import logger

from app.services.text_processor_simple import text_processor
from app.services.embedding_service import embedding_service
from app.services.classification_service import classification_service
from app.services.retrieval_service import retrieval_service
from app.services.database_service import db_service
from app.models.database import ProcessingLog


class AnalysisService:
    """Main service for analyzing claims for misinformation"""
    
    def __init__(self):
        pass
    
    async def analyze_claim(self, claim_text: str, source_url: Optional[str] = None, 
                          db: Session = None) -> Dict[str, Any]:
        """
        Complete analysis pipeline for a single claim
        """
        start_time = time.time()
        claim_id = None
        
        try:
            logger.info(f"Starting analysis for claim: {claim_text[:100]}...")
            
            # Validate input
            if not claim_text or not claim_text.strip():
                raise ValueError("Claim text cannot be empty")
            
            # Step 1: Store the claim in database
            try:
                claim_record = db_service.store_claim(db, claim_text, source_url)
                claim_id = claim_record.id
                logger.info(f"Stored claim with ID: {claim_id}")
            except Exception as e:
                logger.error(f"Failed to store claim: {e}")
                raise Exception(f"Database error: {str(e)}")
            
            # Step 2: Extract features and preprocess
            try:
                features = text_processor.extract_claim_features(claim_text)
                logger.debug(f"Extracted features: {len(features['keywords'])} keywords")
            except Exception as e:
                logger.error(f"Failed to extract features: {e}")
                raise Exception(f"Text processing error: {str(e)}")
            
            # Step 3: Generate embedding
            try:
                claim_embedding = embedding_service.generate_embedding(claim_text)
                logger.debug(f"Generated embedding of dimension {len(claim_embedding)}")
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                raise Exception(f"Embedding generation error: {str(e)}")
            
            # Step 4: Store embedding in vector database
            try:
                embedding_metadata = {
                    "claim_id": claim_id,
                    "text": claim_text,
                    "source_url": source_url,
                    "word_count": features["word_count"],
                    "created_at": claim_record.created_at.isoformat()
                }
                db_service.store_embedding(claim_id, claim_embedding, embedding_metadata)
                logger.debug(f"Stored embedding for claim {claim_id}")
            except Exception as e:
                logger.warning(f"Failed to store embedding: {e}")
                # Continue without embedding storage
            
            # Step 5: Retrieve relevant evidence
            try:
                evidence_sources = retrieval_service.retrieve_evidence(claim_text, db, max_results=5)
                logger.info(f"Retrieved {len(evidence_sources)} evidence sources")
            except Exception as e:
                logger.warning(f"Failed to retrieve evidence: {e}")
                evidence_sources = []
            
            # Step 6: Extract evidence snippets
            try:
                evidence_snippets = retrieval_service.extract_evidence_snippets(
                    claim_text, evidence_sources, max_snippet_length=200
                )
                logger.debug(f"Extracted {len(evidence_snippets)} evidence snippets")
            except Exception as e:
                logger.warning(f"Failed to extract evidence snippets: {e}")
                evidence_snippets = []
            
            # Step 7: Classify the claim
            try:
                classification_result = classification_service.classify_claim(
                    claim_text, evidence_snippets
                )
                logger.info(f"Classification result: {classification_result['classification']}")
            except Exception as e:
                logger.error(f"Failed to classify claim: {e}")
                classification_result = {
                    "classification": "Unverified",
                    "reliability_score": 0.0,
                    "confidence": 0.0,
                    "reasoning": f"Classification error: {str(e)}"
                }
            
            # Step 8: Update claim with results
            try:
                db_service.update_claim_results(
                    db, claim_id, 
                    classification_result["classification"],
                    classification_result["reliability_score"],
                    evidence_snippets
                )
                logger.debug(f"Updated claim {claim_id} with results")
            except Exception as e:
                logger.warning(f"Failed to update claim results: {e}")
            
            # Step 9: Prepare final response
            processing_time = int((time.time() - start_time) * 1000)
            
            result = {
                "claim_id": claim_id,
                "claim": claim_text,
                "classification": classification_result["classification"],
                "reliability": classification_result["reliability_score"],
                "confidence": classification_result["confidence"],
                "evidence": evidence_snippets,
                "reasoning": classification_result["reasoning"],
                "source_url": source_url,
                "processing_time_ms": processing_time,
                "evidence_sources": [
                    {
                        "title": source.get("title", "Unknown"),
                        "source_name": source.get("source_name", "Unknown"),
                        "source_url": source.get("source_url", ""),
                        "relevance_score": source.get("hybrid_score", 0.0)
                    }
                    for source in evidence_sources
                ]
            }
            
            # Log processing
            self._log_processing(db, claim_id, "analysis_complete", "success", 
                               {"result": result}, processing_time)
            
            logger.info(f"Analysis completed in {processing_time}ms")
            return result
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            
            # Log error
            if claim_id:
                self._log_processing(db, claim_id, "analysis_failed", "error", 
                                   {"error": error_msg}, processing_time)
            
            return {
                "claim_id": claim_id,
                "claim": claim_text,
                "classification": "Unverified",
                "reliability": 0.0,
                "confidence": 0.0,
                "evidence": [],
                "reasoning": error_msg,
                "source_url": source_url,
                "processing_time_ms": processing_time,
                "evidence_sources": []
            }
    
    async def analyze_claims_batch(self, claims: List[Dict[str, str]], db: Session) -> List[Dict[str, Any]]:
        """
        Analyze multiple claims in batch
        Each claim should be a dict with 'text' and optionally 'source_url'
        """
        results = []
        
        logger.info(f"Starting batch analysis for {len(claims)} claims")
        
        for i, claim_data in enumerate(claims):
            claim_text = claim_data.get("text", "")
            source_url = claim_data.get("source_url")
            
            if not claim_text.strip():
                results.append({
                    "claim": claim_text,
                    "classification": "Unverified",
                    "reliability": 0.0,
                    "evidence": [],
                    "reasoning": "Empty claim text",
                    "source_url": source_url
                })
                continue
            
            try:
                result = await self.analyze_claim(claim_text, source_url, db)
                results.append(result)
                logger.info(f"Completed analysis {i+1}/{len(claims)}")
                
            except Exception as e:
                logger.error(f"Failed to analyze claim {i+1}: {e}")
                results.append({
                    "claim": claim_text,
                    "classification": "Unverified", 
                    "reliability": 0.0,
                    "evidence": [],
                    "reasoning": f"Analysis error: {str(e)}",
                    "source_url": source_url
                })
        
        logger.info(f"Batch analysis completed: {len(results)} results")
        return results
    
    def get_claim_history(self, db: Session, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent claim analysis history"""
        try:
            from app.models.database import Claim
            
            claims = db.query(Claim).order_by(Claim.created_at.desc()).limit(limit).all()
            
            history = []
            for claim in claims:
                history.append({
                    "id": claim.id,
                    "text": claim.text,
                    "classification": claim.classification,
                    "reliability_score": claim.reliability_score,
                    "evidence": claim.evidence,
                    "source_url": claim.source_url,
                    "created_at": claim.created_at.isoformat(),
                    "processed_at": claim.processed_at.isoformat() if claim.processed_at else None
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to retrieve claim history: {e}")
            return []
    
    def get_analysis_stats(self, db: Session) -> Dict[str, Any]:
        """Get analysis statistics"""
        try:
            from app.models.database import Claim
            from sqlalchemy import func
            
            # Count by classification
            classification_counts = db.query(
                Claim.classification, 
                func.count(Claim.id)
            ).group_by(Claim.classification).all()
            
            # Average reliability score
            avg_reliability = db.query(func.avg(Claim.reliability_score)).scalar() or 0.0
            
            # Total claims processed
            total_claims = db.query(func.count(Claim.id)).scalar() or 0
            
            stats = {
                "total_claims": total_claims,
                "average_reliability": round(float(avg_reliability), 2),
                "classification_breakdown": {
                    classification: count for classification, count in classification_counts
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get analysis stats: {e}")
            return {
                "total_claims": 0,
                "average_reliability": 0.0,
                "classification_breakdown": {}
            }
    
    def _log_processing(self, db: Session, claim_id: int, operation: str, status: str, 
                       details: Dict[str, Any], processing_time_ms: int):
        """Log processing activity"""
        try:
            log_entry = ProcessingLog(
                claim_id=claim_id,
                operation=operation,
                status=status,
                details=details,
                processing_time_ms=processing_time_ms
            )
            db.add(log_entry)
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to log processing activity: {e}")


# Global analysis service instance
analysis_service = AnalysisService()
