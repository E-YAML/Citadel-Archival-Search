import asyncio
import os
import sys
from loguru import logger

# Add the 'backend' folder to the python path to resolve absolute imports from the 'app' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.logging import setup_logging
from app.services.ingestion import ASOIAFIngestionPipeline

async def main() -> None:
    setup_logging()
    logger.info("Initializing Neo4j Seed Script...")
    
    pipeline = ASOIAFIngestionPipeline()
    
    try:
        # Verify connection and seed
        logger.info("Testing Neo4j connectivity and seeding lineages...")
        await pipeline.seed_sample_lineage()
        logger.success("Neo4j database seeding completed successfully!")
    except Exception as e:
        logger.error(f"Failed to seed Neo4j: {e}")
    finally:
        await pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())
