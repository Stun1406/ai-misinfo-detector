"""
Main FastAPI application for AI Misinformation Detector
"""
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from sqlalchemy.orm import Session
import csv
import io
from loguru import logger

from app.core.config import settings
from app.models.database import get_db
from app.models.schemas import (
    ClaimRequest, ClaimResponse, BatchClaimRequest, BatchClaimResponse,
    FactSourceRequest, FactSourceResponse, ClaimHistoryResponse,
    AnalysisStats, HealthResponse, ErrorResponse
)
from app.services.analysis_service import analysis_service
from app.services.database_service import db_service

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="AI-powered misinformation detection system"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connections
        postgres_healthy = True
        qdrant_healthy = True
        
        try:
            # Test PostgreSQL
            from sqlalchemy import text
            db = next(get_db())
            db.execute(text("SELECT 1"))
            db.close()
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            postgres_healthy = False
        
        try:
            # Test Qdrant
            if db_service.qdrant_client:
                db_service.qdrant_client.get_collections()
            else:
                qdrant_healthy = False
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            qdrant_healthy = False
        
        services_status = {
            "postgresql": postgres_healthy,
            "qdrant": qdrant_healthy,
            "embedding_service": True,  # Will be tested when first used
            "classification_service": True
        }
        
        overall_status = "healthy" if all(services_status.values()) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version=settings.api_version,
            services=services_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/analyze", response_model=ClaimResponse)
async def analyze_claim(claim_request: ClaimRequest, db: Session = Depends(get_db)):
    """Analyze a single claim for misinformation"""
    try:
        # Validate input
        if not claim_request.text or not claim_request.text.strip():
            raise HTTPException(status_code=400, detail="Claim text cannot be empty")
        
        if len(claim_request.text) > 5000:
            raise HTTPException(status_code=400, detail="Claim text too long (max 5000 characters)")
        
        logger.info(f"Analyzing claim: {claim_request.text[:100]}...")
        
        result = await analysis_service.analyze_claim(
            claim_text=claim_request.text,
            source_url=claim_request.source_url,
            db=db
        )
        
        return ClaimResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch", response_model=BatchClaimResponse)
async def analyze_claims_batch(batch_request: BatchClaimRequest, db: Session = Depends(get_db)):
    """Analyze multiple claims in batch"""
    try:
        if not batch_request.claims:
            raise HTTPException(status_code=400, detail="No claims provided")
        
        if len(batch_request.claims) > 50:
            raise HTTPException(status_code=400, detail="Too many claims (max 50 per batch)")
        
        claims_data = [
            {"text": claim.text, "source_url": claim.source_url}
            for claim in batch_request.claims
        ]
        
        logger.info(f"Starting batch analysis for {len(claims_data)} claims")
        
        results = await analysis_service.analyze_claims_batch(claims_data, db)
        
        # Calculate summary statistics
        classifications = [r["classification"] for r in results]
        avg_reliability = sum(r["reliability"] for r in results) / len(results) if results else 0
        
        summary = {
            "total_analyzed": len(results),
            "average_reliability": round(avg_reliability, 2),
            "classification_counts": {
                "True": classifications.count("True"),
                "False": classifications.count("False"),
                "Misleading": classifications.count("Misleading"),
                "Unverified": classifications.count("Unverified")
            }
        }
        
        return BatchClaimResponse(
            results=[ClaimResponse(**result) for result in results],
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@app.post("/analyze/upload")
async def analyze_csv_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Analyze claims from uploaded CSV file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV content
        contents = await file.read()
        csv_content = contents.decode('utf-8')
        
        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        claims_data = []
        
        for row in csv_reader:
            if 'text' in row or 'claim' in row:
                claim_text = row.get('text') or row.get('claim', '').strip()
                source_url = row.get('source_url', None)
                
                if claim_text:
                    claims_data.append({"text": claim_text, "source_url": source_url})
        
        if not claims_data:
            raise HTTPException(status_code=400, detail="No valid claims found in CSV")
        
        if len(claims_data) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 claims per batch")
        
        logger.info(f"Analyzing {len(claims_data)} claims from CSV upload")
        
        # Analyze claims
        results = await analysis_service.analyze_claims_batch(claims_data, db)
        
        # Calculate summary
        classifications = [r["classification"] for r in results]
        avg_reliability = sum(r["reliability"] for r in results) / len(results) if results else 0
        
        summary = {
            "total_analyzed": len(results),
            "average_reliability": round(avg_reliability, 2),
            "classification_counts": {
                "True": classifications.count("True"),
                "False": classifications.count("False"),
                "Misleading": classifications.count("Misleading"),
                "Unverified": classifications.count("Unverified")
            }
        }
        
        return {
            "results": [ClaimResponse(**result) for result in results],
            "summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV upload analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"CSV upload analysis failed: {str(e)}")


@app.get("/claims/history", response_model=List[ClaimHistoryResponse])
async def get_claim_history(limit: int = 50, db: Session = Depends(get_db)):
    """Get recent claim analysis history"""
    try:
        if limit > 100:
            limit = 100  # Cap the limit
        
        history = analysis_service.get_claim_history(db, limit=limit)
        return [ClaimHistoryResponse(**claim) for claim in history]
        
    except Exception as e:
        logger.error(f"Failed to retrieve history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@app.get("/claims/{claim_id}", response_model=ClaimResponse)
async def get_claim_by_id(claim_id: int, db: Session = Depends(get_db)):
    """Get specific claim analysis by ID"""
    try:
        from app.models.database import Claim
        
        claim = db.query(Claim).filter(Claim.id == claim_id).first()
        if not claim:
            raise HTTPException(status_code=404, detail="Claim not found")
        
        response_data = {
            "claim_id": claim.id,
            "claim": claim.text,
            "classification": claim.classification or "Unverified",
            "reliability": claim.reliability_score or 0.0,
            "confidence": 75.0,  # Default confidence
            "evidence": claim.evidence or [],
            "reasoning": "Retrieved from database",
            "source_url": claim.source_url,
            "processing_time_ms": 0,
            "evidence_sources": []
        }
        
        return ClaimResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve claim: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve claim: {str(e)}")


@app.get("/stats", response_model=AnalysisStats)
async def get_analysis_stats(db: Session = Depends(get_db)):
    """Get analysis statistics"""
    try:
        stats = analysis_service.get_analysis_stats(db)
        return AnalysisStats(**stats)
        
    except Exception as e:
        logger.error(f"Failed to retrieve stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")


@app.post("/fact-sources", response_model=FactSourceResponse)
async def add_fact_source(fact_source: FactSourceRequest, db: Session = Depends(get_db)):
    """Add a new fact-checking source to the database"""
    try:
        # Validate input
        if not fact_source.title or not fact_source.content:
            raise HTTPException(status_code=400, detail="Title and content are required")
        
        source_record = db_service.add_fact_source(
            db=db,
            title=fact_source.title,
            content=fact_source.content,
            source_name=fact_source.source_name,
            source_url=fact_source.source_url,
            topic=fact_source.topic
        )
        
        return FactSourceResponse.from_orm(source_record)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add fact source: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add fact source: {str(e)}")


@app.get("/fact-sources", response_model=List[FactSourceResponse])
async def get_fact_sources(topic: Optional[str] = None, limit: int = 50, db: Session = Depends(get_db)):
    """Get fact-checking sources, optionally filtered by topic"""
    try:
        if limit > 100:
            limit = 100  # Cap the limit
        
        sources = db_service.get_fact_sources(db, topic=topic, limit=limit)
        return [FactSourceResponse.from_orm(source) for source in sources]
        
    except Exception as e:
        logger.error(f"Failed to retrieve fact sources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve fact sources: {str(e)}")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error(f"ValueError: {exc}")
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=settings.debug)
