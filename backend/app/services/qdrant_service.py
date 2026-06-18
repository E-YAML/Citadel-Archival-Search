from typing import Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from loguru import logger

from app.core.config import settings


class QdrantConnectionError(Exception):
    """Raised when connections to Qdrant fail or cannot be established."""
    pass


class QdrantService:
    """
    Service layer managing connections and administrative operations
    on the Qdrant vector database.
    """
    def __init__(self) -> None:
        self._client: Optional[AsyncQdrantClient] = None

    def get_client(self) -> AsyncQdrantClient:
        """
        Retrieves the singleton instance of the AsyncQdrantClient.
        Initializes it if not already instantiated.
        """
        if self._client is None:
            try:
                # Initialize AsyncQdrantClient using settings
                self._client = AsyncQdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                )
                logger.info("AsyncQdrantClient successfully initialized.")
            except Exception as e:
                logger.error(f"Initialization of AsyncQdrantClient failed: {str(e)}")
                raise QdrantConnectionError(f"Could not connect to Qdrant: {str(e)}") from e
        return self._client

    async def init_collection(self, collection_name: str, vector_size: int) -> None:
        """
        Defensively initializes a Qdrant collection.
        Checks if it exists; if not, creates it using Cosine distance.
        """
        client = self.get_client()
        try:
            logger.info(f"Checking if Qdrant collection '{collection_name}' exists...")
            # Check if collection exists
            exists = await client.collection_exists(collection_name)
            if not exists:
                logger.info(f"Collection '{collection_name}' not found. Creating collection...")
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    ),
                    sparse_vectors_config={
                        "sparse-text": models.SparseVectorParams(
                            index=models.SparseIndexParams(
                                on_disk=True
                            )
                        )
                    }
                )
                logger.info(f"Collection '{collection_name}' created successfully with size {vector_size}.")
            else:
                logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
        except UnexpectedResponse as e:
            logger.error(f"Qdrant server returned an unexpected error for collection '{collection_name}': {e.status_code} - {e.reason_phrase}")
            raise QdrantConnectionError(f"Unexpected response from Qdrant server: {str(e)}") from e
        except Exception as e:
            logger.error(f"Error during collection initialization for '{collection_name}': {str(e)}")
            raise QdrantConnectionError(f"Failed to initialize Qdrant collection: {str(e)}") from e

    async def close(self) -> None:
        """
        Closes the active AsyncQdrantClient connection.
        """
        if self._client is not None:
            try:
                await self._client.close()
                logger.info("AsyncQdrantClient connection closed.")
            except Exception as e:
                logger.error(f"Error closing AsyncQdrantClient: {str(e)}")
            finally:
                self._client = None
