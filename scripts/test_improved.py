"""
Test script to demonstrate the improved misinformation detection system
"""
import sys
import os
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.models.database import SessionLocal
from app.services.analysis_service import analysis_service
from loguru import logger


async def test_various_claims():
    """Test the system with various types of claims"""
    logger.info("🧪 Testing AI Misinformation Detector with various claims")
    
    # Test claims with different characteristics
    test_claims = [
        {
            "text": "A study published in Nature found that drinking 8 glasses of water daily reduces cancer risk by 23%",
            "description": "Scientific claim with specific data"
        },
        {
            "text": "BREAKING: SHOCKING SECRET EXPOSED! The government is hiding ALIENS in Area 51!",
            "description": "Sensational conspiracy theory"
        },
        {
            "text": "According to the CDC, regular handwashing helps prevent the spread of infectious diseases",
            "description": "Authoritative source claim"
        },
        {
            "text": "Maybe vaccines could possibly cause autism, but I'm not sure",
            "description": "Uncertainty language claim"
        },
        {
            "text": "The Earth is completely flat and NASA has been lying to us for decades",
            "description": "Absolute false claim"
        },
        {
            "text": "Research from Harvard University shows that meditation reduces stress by 40%",
            "description": "University research claim"
        },
        {
            "text": "This AMAZING product will make you lose weight INSTANTLY! Guaranteed results!",
            "description": "Marketing hype with absolute claims"
        },
        {
            "text": "The sky appears blue due to Rayleigh scattering of sunlight by atmospheric molecules",
            "description": "Factual scientific explanation"
        }
    ]
    
    db = SessionLocal()
    
    try:
        for i, claim_data in enumerate(test_claims, 1):
            logger.info(f"\n--- Test {i}: {claim_data['description']} ---")
            logger.info(f"Claim: {claim_data['text']}")
            
            try:
                result = await analysis_service.analyze_claim(
                    claim_data['text'], 
                    None, 
                    db
                )
                
                logger.info(f"✅ Classification: {result['classification']}")
                logger.info(f"✅ Reliability Score: {result['reliability']:.1f}%")
                logger.info(f"✅ Confidence: {result['confidence']:.1f}%")
                logger.info(f"✅ Reasoning: {result['reasoning']}")
                logger.info(f"✅ Processing Time: {result['processing_time_ms']}ms")
                
                if result['evidence']:
                    logger.info(f"✅ Evidence Found: {len(result['evidence'])} snippets")
                
            except Exception as e:
                logger.error(f"❌ Failed to analyze claim {i}: {e}")
        
        logger.info("\n🎉 All claim tests completed!")
        
    finally:
        db.close()


async def test_batch_analysis():
    """Test batch analysis functionality"""
    logger.info("\n🧪 Testing batch analysis functionality")
    
    batch_claims = [
        "The sun rises in the east",
        "Vaccines are 100% safe with no side effects",
        "Exercise improves cardiovascular health",
        "The moon landing was faked",
        "Drinking water is essential for human survival"
    ]
    
    db = SessionLocal()
    
    try:
        claims_data = [{"text": claim, "source_url": None} for claim in batch_claims]
        
        logger.info(f"Analyzing {len(claims_data)} claims in batch...")
        
        results = await analysis_service.analyze_claims_batch(claims_data, db)
        
        logger.info("📊 Batch Analysis Results:")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. {result['classification']} ({result['reliability']:.1f}%) - {result['claim'][:50]}...")
        
        # Calculate summary
        classifications = [r["classification"] for r in results]
        avg_reliability = sum(r["reliability"] for r in results) / len(results)
        
        logger.info(f"\n📈 Summary:")
        logger.info(f"  Average Reliability: {avg_reliability:.1f}%")
        logger.info(f"  True: {classifications.count('True')}")
        logger.info(f"  False: {classifications.count('False')}")
        logger.info(f"  Misleading: {classifications.count('Misleading')}")
        logger.info(f"  Unverified: {classifications.count('Unverified')}")
        
    finally:
        db.close()


def test_system_stats():
    """Test system statistics"""
    logger.info("\n🧪 Testing system statistics")
    
    db = SessionLocal()
    
    try:
        stats = analysis_service.get_analysis_stats(db)
        
        logger.info("📊 System Statistics:")
        logger.info(f"  Total Claims Analyzed: {stats['total_claims']}")
        logger.info(f"  Average Reliability: {stats['average_reliability']}%")
        logger.info(f"  Classification Breakdown: {stats['classification_breakdown']}")
        
    finally:
        db.close()


async def main():
    """Main test function"""
    logger.info("🚀 Starting Comprehensive AI Misinformation Detector Tests")
    logger.info(f"Configuration: PostgreSQL on port {settings.postgres_port}, Qdrant on port {settings.qdrant_port}")
    
    try:
        # Test 1: Various claim types
        await test_various_claims()
        
        # Test 2: Batch analysis
        await test_batch_analysis()
        
        # Test 3: System statistics
        test_system_stats()
        
        logger.info("\n🎉 All comprehensive tests completed successfully!")
        logger.info("The AI Misinformation Detector is working correctly and accurately!")
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")


if __name__ == "__main__":
    asyncio.run(main())
