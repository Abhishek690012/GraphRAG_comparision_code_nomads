"""
Embedding Engine for Basic RAG Module

Local embedding generator using sentence-transformers.
Supports single encoding and batch encoding with L2 normalization.
"""

import time
import logging
import math
from typing import Any, Dict, List, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Local embedding generator.
    """

    def __init__(self, config: Dict[str, Any], cache_manager: Optional[CacheManager] = None):
        self._mode = config.get("mode", "local")
        self._model_name = config.get("model_name_or_path", "BAAI/bge-small-en-v1.5")
        self._device = config.get("device", "cpu")
        self._normalize = config.get("normalize_embeddings", True)
        self._batch_size = config.get("batch_size", 16)
        self._cache_dir = config.get("cache_dir", "./cache/embeddings")
        self._trust_remote_code = config.get("trust_remote_code", False)
        
        self._cache_manager = cache_manager
        
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Please install it to use local embeddings.")

        logger.info(f"Initializing local EmbeddingEngine for '{self._model_name}' on {self._device}...")
        logger.info("This may download the model if not cached locally. Please ensure internet connectivity or pre-download.")
        
        try:
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                cache_folder=self._cache_dir,
                trust_remote_code=self._trust_remote_code
            )
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load or download model {self._model_name}. Error: {e}")
            if "disk space" in str(e).lower() or "no space left" in str(e).lower():
                raise RuntimeError("Insufficient disk space for model cache. Free space or set cache_dir to larger volume.") from e
            raise RuntimeError(f"Failed to download {self._model_name}. Check internet connection or pre-download model.") from e

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding model."""
        # For BGE-small-en-v1.5 the dimension is 384
        return 384

    def encode(self, text: str) -> List[float]:
        """
        Encode a single text string into an embedding vector.
        
        Returns:
            List of floats representing the embedding.
        """
        start_time = time.perf_counter()
        
        embeddings = self.encode_batch([text])
        if not embeddings:
            raise ValueError("No embeddings returned from model")
            
        vector = embeddings[0]
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Encoded text in {elapsed_ms:.1f}ms")
        
        return vector

    def encode_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Encode a batch of text strings locally.
        
        Args:
            texts: List of text strings to encode.
            batch_size: Override the default batch size.
            
        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
            
        bs = batch_size or self._batch_size
        start_time = time.perf_counter()
        
        logger.debug(f"Encoding batch of {len(texts)} texts...")
        
        embeddings = self._model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        
        vectors = [emb.tolist() for emb in embeddings]
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Encoded batch of {len(texts)} texts in {elapsed_ms:.1f}ms")
        
        return vectors
