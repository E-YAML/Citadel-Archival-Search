import os
import re
import uuid
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.http import models
from loguru import logger

from app.services.qdrant_service import QdrantService, QdrantConnectionError
from app.services.neo4j_service import Neo4jService, Neo4jConnectionError


class ASOIAFIngestionPipeline:
    """
    Pipeline responsible for ETL processes:
    1. Parsing ASOIAF plaintext chapter files
    2. Generating semantic text chunks
    3. Generating dense & sparse embeddings (local Hybrid representation)
    4. Idempotently uploading to Qdrant
    5. Seeding the Neo4j Graph Database
    """

    def __init__(self) -> None:
        self.qdrant_service = QdrantService()
        self.neo4j_service = Neo4jService()
        self._dense_model: Optional[TextEmbedding] = None
        self._sparse_model: Optional[SparseTextEmbedding] = None

    def get_dense_model(self) -> TextEmbedding:
        """Lazy loader for dense embedding model to minimize load times."""
        if self._dense_model is None:
            logger.info("Loading dense embedding model (BAAI/bge-small-en-v1.5)...")
            self._dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return self._dense_model

    def get_sparse_model(self) -> SparseTextEmbedding:
        """Lazy loader for sparse embedding model to minimize load times."""
        if self._sparse_model is None:
            logger.info("Loading sparse embedding model (prithivida/Splade_PP_en_v1)...")
            self._sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        return self._sparse_model

    def parse_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Parses a target text file. Uses line-by-line scanning and regex to
        group content by Book and Chapter, extracting POV characters.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Source file not found at: {filepath}")

        logger.info(f"Parsing ASOIAF raw content from: {filepath}")
        chapters: List[Dict[str, Any]] = []
        current_book = "Unknown Book"
        current_chapter: Optional[str] = None
        current_pov: Optional[str] = None
        chapter_content: List[str] = []

        book_regex = re.compile(r"^BOOK:\s*(.+)$", re.IGNORECASE)
        chapter_regex = re.compile(r"^CHAPTER:\s*(.+)$", re.IGNORECASE)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()

                # Match BOOK marker
                book_match = book_regex.match(stripped)
                if book_match:
                    current_book = book_match.group(1).strip()
                    continue

                # Match CHAPTER marker
                chapter_match = chapter_regex.match(stripped)
                if chapter_match:
                    # Save preceding chapter
                    if current_chapter and chapter_content:
                        chapters.append({
                            "book_title": current_book,
                            "chapter_title": current_chapter,
                            "pov_character": current_pov,
                            "content": "\n".join(chapter_content).strip()
                        })
                    
                    current_chapter = chapter_match.group(1).strip()
                    # POV character is typically the first identifier in the chapter (e.g. "EDDARD")
                    current_pov = current_chapter.split()[0]
                    chapter_content = []
                    continue

                # Accumulate chapter lines
                if current_chapter is not None:
                    chapter_content.append(line.rstrip())

            # Save the final chapter
            if current_chapter and chapter_content:
                chapters.append({
                    "book_title": current_book,
                    "chapter_title": current_chapter,
                    "pov_character": current_pov,
                    "content": "\n".join(chapter_content).strip()
                })

        logger.info(f"Parsing complete. Found {len(chapters)} chapters in '{filepath}'.")
        return chapters

    def chunk_chapters(self, chapters: List[Dict[str, Any]]) -> List[Document]:
        """
        Converts parsed chapters into granular LangChain Documents.
        Splits on a token level (512 size, 50 overlap) with metadata stitching.
        """
        logger.info("Splitting chapter contents into semantic token chunks...")
        # Recursive splitter utilizing OpenAI cl100k_base tokenization
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512,
            chunk_overlap=50
        )

        documents: List[Document] = []
        for chapter in chapters:
            content = chapter["content"]
            chunk_texts = text_splitter.split_text(content)

            for index, text in enumerate(chunk_texts):
                doc = Document(
                    page_content=text,
                    metadata={
                        "book_title": chapter["book_title"],
                        "chapter_title": chapter["chapter_title"],
                        "pov_character": chapter["pov_character"],
                        "chunk_index": index
                    }
                )
                documents.append(doc)

        logger.info(f"Splitting complete. Created {len(documents)} chunk documents.")
        return documents

    async def upload_to_qdrant(self, chunks: List[Document], collection_name: str) -> None:
        """
        Calculates deterministic UUIDv5 coordinates for idempotency, generates
        hybrid (dense + sparse) embeddings in memory-safe batches, and pushes uploads to Qdrant.
        """
        if not chunks:
            logger.warning("No chunks provided for upload.")
            return

        client = self.qdrant_service.get_client()
        dense_model = self.get_dense_model()
        sparse_model = self.get_sparse_model()

        # Batch configuration to avoid MemoryError (especially with Splade on CPU)
        batch_size = 128
        total_chunks = len(chunks)
        namespace_uuid = uuid.UUID("8c2901fa-9a57-4b13-8cfb-eb4a961f6e24")

        logger.info(f"Uploading {total_chunks} chunks to Qdrant in batches of {batch_size}...")

        for batch_start in range(0, total_chunks, batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]
            batch_texts = [chunk.page_content for chunk in batch_chunks]

            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} (size={len(batch_texts)})...")
            
            # Generate vectors for current batch
            try:
                dense_embeddings = list(dense_model.embed(batch_texts))
                sparse_embeddings = list(sparse_model.embed(batch_texts))
            except Exception as embed_err:
                logger.error(f"Embedding generation failed for batch starting at {batch_start}: {embed_err}")
                raise

            points: List[models.PointStruct] = []
            for i, chunk in enumerate(batch_chunks):
                unique_key = f"{chunk.metadata['book_title']}_{chunk.metadata['chapter_title']}_{chunk.metadata['chunk_index']}"
                point_uuid = str(uuid.uuid5(namespace_uuid, unique_key))

                dense_vector = dense_embeddings[i].tolist()
                sparse_vector = sparse_embeddings[i]

                points.append(
                    models.PointStruct(
                        id=point_uuid,
                        vector={
                            "": dense_vector,
                            "sparse-text": models.SparseVector(
                                indices=sparse_vector.indices.tolist(),
                                values=sparse_vector.values.tolist()
                            )
                        },
                        payload={
                            "page_content": chunk.page_content,
                            **chunk.metadata
                        }
                    )
                )

            try:
                await client.upsert(
                    collection_name=collection_name,
                    points=points
                )
            except Exception as e:
                logger.error(f"Error during Qdrant batch upsert: {str(e)}")
                raise QdrantConnectionError(f"Qdrant bulk load failed: {str(e)}") from e

        logger.info("Qdrant upload completed successfully for all batches.")

    async def seed_sample_lineage(self) -> None:
        """
        Executes idempotent Cypher statements to seed sample relationships
        defining relationships inside Neo4j.
        """
        driver = self.neo4j_service.get_driver()
        
        # Cypher MERGE queries to avoid duplicates
        queries = [
            "MERGE (j:Character {name: 'Jon Snow'}) SET j.house = 'Stark', j.status = 'Bastard'",
            "MERGE (l:Character {name: 'Lyanna Stark'}) SET l.house = 'Stark'",
            "MERGE (r:Character {name: 'Rhaegar Targaryen'}) SET r.house = 'Targaryen'",
            "MATCH (j:Character {name: 'Jon Snow'}), (l:Character {name: 'Lyanna Stark'}) MERGE (j)-[:SON_OF]->(l)",
            "MATCH (r:Character {name: 'Rhaegar Targaryen'}), (j:Character {name: 'Jon Snow'}) MERGE (r)-[:FATHER_OF]->(j)"
        ]

        logger.info("Seeding Neo4j instance with standard character lineages...")
        try:
            async with driver.session() as session:
                for query in queries:
                    await session.run(query)
            logger.info("Neo4j database seeding completed successfully.")
        except Exception as e:
            logger.error(f"Error seeding Neo4j: {str(e)}")
            raise Neo4jConnectionError(f"Neo4j seeding transaction failed: {str(e)}") from e

    async def close(self) -> None:
        """Safely tears down active connection hooks."""
        await self.qdrant_service.close()
        await self.neo4j_service.close()
