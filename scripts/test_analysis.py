"""
Quick test script to verify the analysis pipeline
"""
import sys
import os
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.models.database import SessionLocal
from app.services.analysis_service import analysis_service

async def test_analysis():
    """Test the analysis pipeline with sample claims"""
    print("🧪 Testing AI Misinformation Detector Analysis Pipeline")
    
    # Test claims
    test_claims = [
        "COVID-19 vaccines contain microchips that track people",
        "The Earth is flat and space agencies are lying",
        "Climate change is primarily caused by human activities", 
        "5G towers cause coronavirus infections"
    ]
    
    db = SessionLocal()
    
    for i, claim in enumerate(test_claims):
        print(f"\n📋 Test {i+1}: {claim}")
        print("-" * 50)
        
        try:
            result = await analysis_service.analyze_claim(claim, None, db)
            
            print(f"Classification: {result['classification']}")
            print(f"Reliability: {result['reliability']:.1f}%")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Evidence sources: {len(result['evidence_sources'])}")
            print(f"Processing time: {result['processing_time_ms']}ms")
            
            if result['evidence']:
                print(f"Evidence snippets: {len(result['evidence'])}")
                for j, evidence in enumerate(result['evidence'][:2]):  # Show first 2
                    print(f"  {j+1}. {evidence[:100]}...")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    db.close()
    print("\n🎉 Analysis testing completed!")

if __name__ == "__main__":
    asyncio.run(test_analysis())
