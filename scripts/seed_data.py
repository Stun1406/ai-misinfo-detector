"""
Script to seed the database with sample fact-checking data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from app.models.database import SessionLocal, FactSource
from app.services.database_service import db_service
from app.services.embedding_service import embedding_service
from loguru import logger


def create_sample_fact_sources():
    """Create sample fact-checking sources for testing"""
    
    sample_sources = [
        {
            "title": "COVID-19 Vaccines Are Safe and Effective",
            "content": "Multiple large-scale clinical trials have demonstrated that COVID-19 vaccines approved by the FDA are both safe and effective. The Pfizer-BioNTech vaccine showed 95% efficacy in preventing COVID-19 in clinical trials. The Moderna vaccine showed 94.1% efficacy. Serious adverse events are rare, occurring in fewer than 0.01% of vaccine recipients according to CDC data from over 400 million doses administered.",
            "source_name": "CDC",
            "source_url": "https://www.cdc.gov/coronavirus/2019-ncov/vaccines/safety/safety-of-vaccines.html",
            "topic": "health"
        },
        {
            "title": "Climate Change Is Caused by Human Activities",
            "content": "Scientific consensus shows that current climate change is primarily caused by human activities, particularly the emission of greenhouse gases from burning fossil fuels. NASA and NOAA data show that Earth's average temperature has risen by about 1.1 degrees Celsius since the late 19th century. The Intergovernmental Panel on Climate Change (IPCC) states with high confidence that human influence has warmed the planet.",
            "source_name": "NASA",
            "source_url": "https://climate.nasa.gov/evidence/",
            "topic": "climate"
        },
        {
            "title": "The 2020 US Presidential Election Was Secure",
            "content": "The 2020 United States presidential election was the most secure in American history according to the Department of Homeland Security. Multiple recounts, audits, and court cases found no evidence of widespread fraud. Over 60 court cases challenging the election results were dismissed due to lack of evidence. Election officials from both parties confirmed the integrity of the election process.",
            "source_name": "Department of Homeland Security",
            "source_url": "https://www.cisa.gov/news/2020/11/12/joint-statement-elections-infrastructure-government-coordinating-council-election",
            "topic": "politics"
        },
        {
            "title": "5G Technology Does Not Cause Cancer",
            "content": "There is no scientific evidence that 5G networks cause cancer or other health problems. The radio frequency energy used by 5G is non-ionizing, meaning it does not have enough energy to damage DNA directly. The Federal Communications Commission (FCC) and the Food and Drug Administration (FDA) have both stated that 5G technology poses no known health risks when operating within established safety guidelines.",
            "source_name": "FDA",
            "source_url": "https://www.fda.gov/radiation-emitting-products/cell-phones/scientific-evidence-cell-phone-safety",
            "topic": "technology"
        },
        {
            "title": "Vaccines Do Not Cause Autism",
            "content": "Extensive scientific research has found no link between vaccines and autism. A large-scale study of over 650,000 children published in the Annals of Internal Medicine found no association between the MMR vaccine and autism. The original 1998 study claiming a link was retracted due to serious methodological flaws and ethical violations. The CDC, WHO, and American Academy of Pediatrics all confirm vaccines are safe and do not cause autism.",
            "source_name": "CDC",
            "source_url": "https://www.cdc.gov/vaccinesafety/concerns/autism.html",
            "topic": "health"
        },
        {
            "title": "The Moon Landing Was Real",
            "content": "The Apollo moon landings were real achievements of human space exploration. Independent verification includes retroreflectors left on the moon that are still used today for laser ranging experiments, moon rocks brought back and studied by scientists worldwide, and photographic evidence that matches the physics of the lunar environment. Conspiracy theories have been thoroughly debunked by scientists and engineers.",
            "source_name": "NASA",
            "source_url": "https://www.nasa.gov/audience/forstudents/5-8/features/nasa-knows/who-was-first-on-moon.html",
            "topic": "science"
        },
        {
            "title": "Earth Is Not Flat",
            "content": "Earth is an oblate spheroid, not flat. This has been established through multiple lines of evidence including satellite imagery, physics experiments, astronomical observations, and direct measurement. Ships disappearing over the horizon, time zones, seasons, and the behavior of pendulums all confirm Earth's spherical shape. The flat Earth theory contradicts fundamental physics and observable phenomena.",
            "source_name": "National Geographic",
            "source_url": "https://www.nationalgeographic.com/science/article/why-earth-round-not-flat-globe",
            "topic": "science"
        },
        {
            "title": "Hydroxychloroquine Is Not an Effective COVID-19 Treatment",
            "content": "Multiple randomized controlled trials have shown that hydroxychloroquine is not effective for treating COVID-19 and may cause harmful side effects. The FDA revoked its emergency use authorization for hydroxychloroquine in June 2020 after reviewing clinical trial data. The WHO discontinued its hydroxychloroquine trial arm due to lack of efficacy. Major medical organizations do not recommend it for COVID-19 treatment.",
            "source_name": "FDA",
            "source_url": "https://www.fda.gov/drugs/drug-safety-and-availability/fda-cautions-against-use-hydroxychloroquine-or-chloroquine-covid-19-outside-hospital-setting-or",
            "topic": "health"
        }
    ]
    
    return sample_sources


def seed_database():
    """Seed the database with sample fact-checking data"""
    logger.info("Starting database seeding process...")
    
    db = SessionLocal()
    
    try:
        # Check if data already exists
        existing_count = db.query(FactSource).count()
        if existing_count > 0:
            logger.info(f"Database already contains {existing_count} fact sources. Skipping seeding.")
            return
        
        # Create sample fact sources
        sample_sources = create_sample_fact_sources()
        
        for source_data in sample_sources:
            logger.info(f"Adding fact source: {source_data['title']}")
            
            # Add to database
            fact_source = db_service.add_fact_source(
                db=db,
                title=source_data["title"],
                content=source_data["content"],
                source_name=source_data["source_name"],
                source_url=source_data["source_url"],
                topic=source_data["topic"]
            )
            
            # Generate and store embedding
            try:
                embedding = embedding_service.generate_embedding(source_data["content"])
                
                embedding_metadata = {
                    "fact_source_id": fact_source.id,
                    "title": fact_source.title,
                    "source_name": fact_source.source_name,
                    "topic": fact_source.topic,
                    "is_fact_source": True
                }
                
                db_service.store_embedding(fact_source.id, embedding, embedding_metadata)
                logger.info(f"Generated embedding for: {source_data['title']}")
                
            except Exception as e:
                logger.error(f"Failed to generate embedding for {source_data['title']}: {e}")
        
        logger.info(f"Successfully seeded database with {len(sample_sources)} fact sources")
        
    except Exception as e:
        logger.error(f"Database seeding failed: {e}")
        db.rollback()
        raise
    
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
