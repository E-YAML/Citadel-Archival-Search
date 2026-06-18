import os
import sys
import asyncio
from loguru import logger

# Add the 'backend' folder to the python path to resolve absolute imports from the 'app' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.logging import setup_logging
from app.services.ingestion import ASOIAFIngestionPipeline


async def main() -> None:
    """
    Main orchestrator function for the ASOIAF ETL Ingestion Pipeline.
    Initializes logging, prepares the Qdrant DB index, parses the mock data,
    chunks and uploads texts to Qdrant, and seeds Neo4j with lineage triples.
    """
    # Initialize JSON stdout logging
    setup_logging()
    
    logger.info("Initializing ASOIAF Ingestion CLI script...")

    # Determine absolute path to the data directory and the target mock file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    sample_file = os.path.join(data_dir, "game_of_thrones_sample.txt")

    pipeline = ASOIAFIngestionPipeline()

    try:
        # 1. Initialize Qdrant collection
        collection_name = "asoiaf_lore"
        # BAAI/bge-small-en-v1.5 has an output dimension of 384
        logger.info(f"Ensuring Qdrant collection '{collection_name}' is initialized...")
        await pipeline.qdrant_service.init_collection(
            collection_name=collection_name,
            vector_size=384
        )

        # 2. Parse target Plaintext/Markdown file
        logger.info(f"Scanning for file to ingest: {sample_file}")
        if not os.path.exists(sample_file):
            logger.error(f"Required target file not found at: {sample_file}")
            sys.exit(1)

        chapters = pipeline.parse_file(sample_file)
        for chap in chapters:
            logger.info(f"Parsed Chapter: {chap['chapter_title']} (POV: {chap['pov_character']}) in book '{chap['book_title']}'")

        # 3. Create semantic overlapping token chunks
        chunks = pipeline.chunk_chapters(chapters)

        # 4. Generate hybrid vectors and upload to Qdrant
        await pipeline.upload_to_qdrant(chunks=chunks, collection_name=collection_name)

        # 5. Seed Neo4j graph lineage triples
        await pipeline.seed_sample_lineage()

        logger.info("ASOIAF Ingestion and database seeding pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Ingestion process terminated with an error: {str(e)}")
        sys.exit(1)
        
    finally:
        # Tear down connection pools
        logger.info("Shutting down active database connections...")
        await pipeline.close()
        logger.info("Databases disconnected.")


if __name__ == "__main__":
    # Run pipeline in an event loop
    asyncio.run(main())
