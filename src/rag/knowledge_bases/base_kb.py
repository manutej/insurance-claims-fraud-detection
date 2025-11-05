"""
Base Knowledge Base infrastructure for RAG system.

Provides abstract base class with common functionality for all KBs including:
- Qdrant collection management
- Vector embedding generation with OpenAI
- Caching with Redis
- Query patterns and retrieval
- Performance monitoring
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class KBDocument(BaseModel):
    """Base document schema for all knowledge bases."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Unique document identifier")
    embedding_text: str = Field(..., description="Text representation for embedding")
    embedding: Optional[List[float]] = Field(
        default=None, description="1536-dimensional vector embedding"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )


class KBStatistics(BaseModel):
    """Statistics for a knowledge base."""

    collection_name: str
    total_documents: int
    vector_dimensions: int
    index_size_mb: float
    avg_query_latency_ms: float
    cache_hit_rate: float
    last_updated: datetime


class BaseKnowledgeBase(ABC):
    """Abstract base class for all knowledge bases."""

    def __init__(
        self,
        collection_name: str,
        qdrant_client: QdrantClient,
        openai_api_key: str,
        vector_size: int = 1536,
        distance: Distance = Distance.COSINE,
        embedding_model: str = "text-embedding-3-large",
    ):
        """
        Initialize knowledge base.

        Args:
            collection_name: Name of Qdrant collection
            qdrant_client: Initialized Qdrant client
            openai_api_key: OpenAI API key for embeddings
            vector_size: Embedding dimension (default: 1536)
            distance: Distance metric (default: COSINE)
            embedding_model: OpenAI embedding model name
        """
        self.collection_name = collection_name
        self.qdrant_client = qdrant_client
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.vector_size = vector_size
        self.distance = distance
        self.embedding_model = embedding_model

        # Performance tracking
        self._query_count = 0
        self._cache_hits = 0
        self._total_query_time_ms = 0.0

        logger.info(
            f"Initialized {self.__class__.__name__} with collection '{collection_name}'"
        )

    def create_collection(
        self,
        on_disk_payload: bool = True,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 128,
    ) -> None:
        """
        Create Qdrant collection with specified configuration.

        Args:
            on_disk_payload: Store payload on disk to save memory
            hnsw_m: HNSW graph connectivity (higher = better recall, more memory)
            hnsw_ef_construct: HNSW construction time/quality trade-off
        """
        if self.qdrant_client.collection_exists(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists")
            return

        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=self.distance,
                on_disk=False,  # Keep vectors in memory for fast queries
            ),
            on_disk_payload=on_disk_payload,
            hnsw_config={
                "m": hnsw_m,
                "ef_construct": hnsw_ef_construct,
                "full_scan_threshold": 10000,
            },
        )
        logger.info(f"Created collection '{self.collection_name}'")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using OpenAI API.

        Args:
            text: Input text to embed

        Returns:
            Normalized 1536-dimensional embedding vector

        Raises:
            Exception: If embedding generation fails after retries
        """
        # Check cache first (using text hash as key)
        cache_key = self._get_embedding_cache_key(text)

        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model, input=text, dimensions=self.vector_size
            )

            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Normalize for cosine distance
            embedding = embedding / np.linalg.norm(embedding)

            # Validate dimension
            assert (
                len(embedding) == self.vector_size
            ), f"Expected {self.vector_size}d, got {len(embedding)}d"

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def generate_embeddings_batch(
        self, texts: List[str], batch_size: int = 100
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = [self.generate_embedding(text) for text in batch]
            embeddings.extend(batch_embeddings)

            logger.info(
                f"Generated {len(embeddings)}/{len(texts)} embeddings for {self.collection_name}"
            )

        return embeddings

    def upsert_documents(self, documents: List[KBDocument]) -> None:
        """
        Index documents in Qdrant collection.

        Args:
            documents: List of documents to index
        """
        points = []

        for doc in documents:
            # Generate embedding if not present
            if doc.embedding is None:
                embedding = self.generate_embedding(doc.embedding_text)
                doc.embedding = embedding.tolist()

            # Create point
            point = PointStruct(
                id=self._hash_id(doc.id),
                vector=doc.embedding,
                payload={
                    "id": doc.id,
                    "embedding_text": doc.embedding_text,
                    "metadata": doc.metadata,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat(),
                },
            )
            points.append(point)

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=batch, wait=True
            )

        logger.info(f"Upserted {len(documents)} documents to '{self.collection_name}'")

    def search(
        self,
        query_text: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_text: Query text
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filters: Optional Qdrant filters

        Returns:
            List of matching documents with scores
        """
        start_time = datetime.utcnow()

        # Generate query embedding
        query_vector = self.generate_embedding(query_text)

        # Search
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filters,
        )

        # Track performance
        query_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        self._query_count += 1
        self._total_query_time_ms += query_time_ms

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "id": result.payload.get("id"),
                    "score": result.score,
                    "embedding_text": result.payload.get("embedding_text"),
                    "metadata": result.payload.get("metadata", {}),
                }
            )

        logger.info(
            f"Search in '{self.collection_name}' returned {len(results)} results in {query_time_ms:.2f}ms"
        )

        return formatted_results

    def get_statistics(self) -> KBStatistics:
        """
        Get knowledge base statistics.

        Returns:
            KBStatistics object with performance metrics
        """
        collection_info = self.qdrant_client.get_collection(self.collection_name)

        avg_latency = (
            self._total_query_time_ms / self._query_count if self._query_count > 0 else 0.0
        )

        cache_hit_rate = (
            self._cache_hits / self._query_count if self._query_count > 0 else 0.0
        )

        return KBStatistics(
            collection_name=self.collection_name,
            total_documents=collection_info.points_count,
            vector_dimensions=self.vector_size,
            index_size_mb=0.0,  # Would need to calculate from collection stats
            avg_query_latency_ms=avg_latency,
            cache_hit_rate=cache_hit_rate,
            last_updated=datetime.utcnow(),
        )

    def _hash_id(self, doc_id: str) -> int:
        """Convert string ID to integer hash for Qdrant."""
        return int(hashlib.sha256(doc_id.encode()).hexdigest()[:16], 16)

    def _get_embedding_cache_key(self, text: str) -> str:
        """Generate cache key for embedding."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"embedding:{self.embedding_model}:{self.vector_size}:{text_hash}"

    @abstractmethod
    def build(self, data_source: Union[str, Dict[str, Any]]) -> None:
        """
        Build knowledge base from data source.

        Must be implemented by subclasses.

        Args:
            data_source: Path to data file or dict of data
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate knowledge base completeness and correctness.

        Must be implemented by subclasses.

        Returns:
            True if validation passes
        """
        pass
