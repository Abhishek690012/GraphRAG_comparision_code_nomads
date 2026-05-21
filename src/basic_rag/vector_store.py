"""
Vector Store Client for Basic RAG Module

Provides a wrapper around ChromaDB for semantic search and ingestion.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Structured result from a vector store search."""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    source_doc_id: str


class VectorStoreClient:
    """
    ChromaDB client wrapper.
    """

    def __init__(self, config: Dict[str, Any]):
        self._collection_name = config.get("collection_name", "medical_chunks")
        self._persist_directory = config.get("persist_directory", "data/chroma_db")
        self._distance_metric = config.get("distance_metric", "cosine")
        self._expected_dimension = config.get("embedding_dimension", 384)
        
        if chromadb is None:
            raise ImportError("chromadb is not installed.")
            
        logger.info(f"Initializing ChromaDB client at {self._persist_directory}")
        # Initialize persistent client
        self._client = chromadb.PersistentClient(
            path=self._persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._distance_metric, "embedding_dimension": self._expected_dimension}
        )
        
        # Validate existing collection dimension if not empty
        if self._collection.count() > 0:
            peek_data = self._collection.peek(1)
            if peek_data and peek_data.get("embeddings") is not None and len(peek_data["embeddings"]) > 0:
                actual_dim = len(peek_data["embeddings"][0])
                if actual_dim != self._expected_dimension:
                    raise RuntimeError(
                        f"Collection {self._collection_name} has embedding dimension {actual_dim}, "
                        f"expected {self._expected_dimension}. Delete collection or update config.embedding_dimension. "
                        f"To reset, run: chromadb reset --collection {self._collection_name} or delete the persistence directory."
                    )

        logger.info(f"Connected to ChromaDB collection '{self._collection_name}' "
                    f"with {self.collection_count} items (dimension: {self._expected_dimension}).")

    @property
    def collection_count(self) -> int:
        """Return the number of items in the collection."""
        return self._collection.count()

    def search(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        score_threshold: float = 0.0, 
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Perform a semantic similarity search.
        
        Args:
            query_embedding: The encoded query vector.
            k: Number of top results to return.
            score_threshold: Minimum similarity score (for cosine/ip, higher is better; 
                             for l2, distance is returned, so lower is better. Chroma returns distance).
                             NOTE: Chroma returns distance. We convert to a similarity score 
                             (1 - distance for cosine/l2 proxy, or handle accordingly).
            metadata_filters: Optional where clause for ChromaDB.
            
        Returns:
            List of RetrievalResult objects.
        """
        if self.collection_count == 0:
            logger.warning(f"Collection {self._collection_name} is empty. Returning no results.")
            return []
            
        where_clause = metadata_filters or None
        
        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieval_results = []
        
        # Extract the single query results
        if not results["ids"] or not results["ids"][0]:
            return []
            
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        for i in range(len(ids)):
            distance = distances[i]
            
            # Convert distance to a similarity score (assuming cosine distance where 0 is identical)
            # Chroma returns cosine distance = 1 - cosine_similarity
            # So similarity = 1 - distance
            if self._distance_metric == "cosine":
                similarity = 1.0 - distance
            else:
                # For L2 or IP, mapping distance to score can be more complex, 
                # we'll use a simple inverse for L2 or raw for IP if needed.
                # Just use the raw distance as the "score" and let thresholding be careful.
                # Assuming cosine for this implementation as per config.
                similarity = 1.0 - distance
                
            if similarity < score_threshold:
                continue
                
            metadata = metadatas[i] or {}
            source_doc_id = metadata.get("source_doc_id", "unknown")
            
            retrieval_results.append(RetrievalResult(
                chunk_id=ids[i],
                text=documents[i],
                score=similarity,
                metadata=metadata,
                source_doc_id=source_doc_id
            ))
            
        logger.debug(f"Retrieved {len(retrieval_results)} chunks for query.")
        return retrieval_results

    def upsert(
        self, 
        ids: List[str], 
        embeddings: List[List[float]], 
        documents: List[str], 
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Upsert a batch of documents into the collection.
        
        Args:
            ids: List of unique chunk IDs.
            embeddings: List of embedding vectors.
            documents: List of text chunks.
            metadatas: List of metadata dicts.
        """
        if not ids:
            return
            
        if embeddings and len(embeddings[0]) != self._expected_dimension:
            raise ValueError(
                f"Upsert failed: Expected embedding dimension {self._expected_dimension}, "
                f"got {len(embeddings[0])}."
            )
            
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
