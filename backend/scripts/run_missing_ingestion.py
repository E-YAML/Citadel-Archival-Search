import os
import sys
import glob
import asyncio
from loguru import logger

# Add the 'backend' folder to the python path to resolve absolute imports from the 'app' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.logging import setup_logging
from app.services.ingestion import ASOIAFIngestionPipeline

TARGET_BOOKS = {
    "A Storm of Swords.txt",
    "A Dance With Dragons.txt"
}

async def main() -> None:
    setup_logging()
    logger.info("Initializing ASOIAF Ingestion Pipeline for missing books...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))

    # Discover all converted book .txt files
    all_txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    book_files = [
        f for f in all_txt_files
        if os.path.basename(f) in TARGET_BOOKS
    ]

    if not book_files:
        logger.error(f"None of the target books found in '{data_dir}'.")
        sys.exit(1)

    logger.info(f"Found {len(book_files)} missing book file(s) to ingest:")
    for f in book_files:
        size_kb = os.path.getsize(f) // 1024
        logger.info(f"  - {os.path.basename(f)} ({size_kb} KB)")

    pipeline = ASOIAFIngestionPipeline()

    try:
        # Ensure Qdrant collection exists
        collection_name = "asoiaf_lore"
        logger.info(f"Ensuring Qdrant collection '{collection_name}' is initialized...")
        await pipeline.qdrant_service.init_collection(
            collection_name=collection_name,
            vector_size=384
        )

        total_chunks = 0
        for book_path in book_files:
            book_name = os.path.basename(book_path)
            logger.info(f"--- Processing: {book_name} ---")

            try:
                chapters = pipeline.parse_file(book_path)
                logger.info(f"  Parsed {len(chapters)} chapters from '{book_name}'")

                chunks = pipeline.chunk_chapters(chapters)
                logger.info(f"  Created {len(chunks)} text chunks")

                await pipeline.upload_to_qdrant(chunks=chunks, collection_name=collection_name)
                total_chunks += len(chunks)

            except Exception as book_err:
                logger.error(f"  Failed to process '{book_name}': {book_err}")
                continue

        logger.info(f"Vector ingestion complete. Total chunks uploaded: {total_chunks}")
        logger.info("✅ Missing books ingestion completed successfully.")

    except Exception as e:
        logger.error(f"Ingestion process terminated with an error: {str(e)}")
        sys.exit(1)

    finally:
        logger.info("Shutting down database connections...")
        await pipeline.close()
        logger.info("Connections closed.")

if __name__ == "__main__":
    asyncio.run(main())
