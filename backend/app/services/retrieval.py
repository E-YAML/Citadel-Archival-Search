from typing import List, Dict, Any, Optional
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.http import models
from loguru import logger

from app.services.qdrant_service import QdrantService, QdrantConnectionError
from app.services.neo4j_service import Neo4jService, Neo4jConnectionError

# Single instances for DB lifecycle hooks
qdrant_service = QdrantService()
neo4j_service = Neo4jService()

# Lazily initialized models
_dense_model: Optional[TextEmbedding] = None
_sparse_model: Optional[SparseTextEmbedding] = None


def get_dense_model() -> TextEmbedding:
    """Lazy loader for query dense embedding generation."""
    global _dense_model
    if _dense_model is None:
        logger.info("Initializing query dense embedding model...")
        _dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _dense_model


def get_sparse_model() -> SparseTextEmbedding:
    """Lazy loader for query sparse embedding generation."""
    global _sparse_model
    if _sparse_model is None:
        logger.info("Initializing query sparse embedding model...")
        _sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    return _sparse_model


async def retrieve_vector_context(query: str, limit: int = 4) -> List[Dict[str, Any]]:
    """
    Queries the Qdrant 'asoiaf_lore' collection.
    Utilizes Reciprocal Rank Fusion (RRF) Hybrid Search to combine semantic search
    and sparse keyword matching.
    """
    logger.info(f"Initiating hybrid retrieval for query: '{query}'")
    try:
        client = qdrant_service.get_client()

        # Embed query text
        dense_model = get_dense_model()
        sparse_model = get_sparse_model()

        # Embed outputs are iterators; grab the first item
        dense_query_vec = list(dense_model.embed([query]))[0].tolist()
        sparse_query_vec = list(sparse_model.embed([query]))[0]

        # Query using prefetching and RRF fusion
        response = await client.query_points(
            collection_name="asoiaf_lore",
            prefetch=[
                models.Prefetch(
                    query=dense_query_vec,
                    limit=limit
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query_vec.indices.tolist(),
                        values=sparse_query_vec.values.tolist()
                    ),
                    using="sparse-text",
                    limit=limit
                )
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=limit
        )

        results: List[Dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            results.append({
                "text": payload.get("page_content", ""),
                "metadata": {
                    "book_title": payload.get("book_title", ""),
                    "chapter_title": payload.get("chapter_title", ""),
                    "pov_character": payload.get("pov_character", ""),
                    "chunk_index": payload.get("chunk_index", 0)
                }
            })

        logger.info(f"Hybrid search complete. Found {len(results)} relevant documents.")
        return results

    except Exception as e:
        logger.error(f"Error querying vector collection: {str(e)}")
        raise QdrantConnectionError(f"Vector search failed: {str(e)}") from e


async def query_knowledge_graph(cypher_query: str) -> List[Dict[str, Any]]:
    """
    Executes an asynchronous Cypher transaction against the Neo4j graph instance
    and returns a structured records list.
    """
    logger.info(f"Executing Cypher Query on Knowledge Graph: '{cypher_query}'")
    try:
        driver = neo4j_service.get_driver()
        async with driver.session() as session:
            result = await session.run(cypher_query)
            data = await result.data()
            logger.info(f"Cypher Query executed successfully. Fetched {len(data)} records.")
            return data
    except Exception as e:
        logger.error(f"Error executing Cypher query: {str(e)}")
        raise Neo4jConnectionError(f"Knowledge graph query failed: {str(e)}") from e
