"""
Pydantic models for API request and response validation
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class ClaimRequest(BaseModel):
    """Request model for single claim analysis"""
    text: str = Field(..., min_length=1, max_length=5000, description="The claim text to analyze")
    source_url: Optional[str] = Field(None, description="Optional URL where the claim was found")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Claim text cannot be empty')
        return v.strip()


class BatchClaimRequest(BaseModel):
    """Request model for batch claim analysis"""
    claims: List[ClaimRequest] = Field(..., min_items=1, max_items=100, description="List of claims to analyze")


class EvidenceSource(BaseModel):
    """Model for evidence sources"""
    title: str
    source_name: str
    source_url: str
    relevance_score: float


class ClaimResponse(BaseModel):
    """Response model for claim analysis"""
    claim_id: Optional[int] = None
    claim: str
    classification: str = Field(..., description="True, False, Misleading, or Unverified")
    reliability: float = Field(..., ge=0, le=100, description="Reliability score from 0-100")
    confidence: float = Field(..., ge=0, le=100, description="Confidence in the classification")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence snippets")
    reasoning: str = Field(..., description="Explanation of the classification")
    source_url: Optional[str] = None
    processing_time_ms: int
    evidence_sources: List[EvidenceSource] = Field(default_factory=list)


class BatchClaimResponse(BaseModel):
    """Response model for batch claim analysis"""
    results: List[ClaimResponse]
    summary: Dict[str, Any] = Field(default_factory=dict)


class FactSourceRequest(BaseModel):
    """Request model for adding fact-checking sources"""
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1, max_length=10000)
    source_name: str = Field(..., min_length=1, max_length=100)
    source_url: str = Field(..., min_length=1, max_length=1000)
    topic: Optional[str] = Field(None, max_length=100)


class FactSourceResponse(BaseModel):
    """Response model for fact sources"""
    id: int
    title: str
    content: str
    source_name: str
    source_url: str
    topic: Optional[str]
    is_verified: bool
    reliability_rating: float
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ClaimHistoryResponse(BaseModel):
    """Response model for claim history"""
    id: int
    text: str
    classification: Optional[str]
    reliability_score: Optional[float]
    evidence: Optional[List[str]]
    source_url: Optional[str]
    created_at: str
    processed_at: Optional[str]


class AnalysisStats(BaseModel):
    """Response model for analysis statistics"""
    total_claims: int
    average_reliability: float
    classification_breakdown: Dict[str, int]


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, bool]


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
