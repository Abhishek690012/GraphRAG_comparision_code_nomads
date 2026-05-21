"""
RAG Configuration Validator for Basic RAG Module

Extends the base ConfigValidator to include RAG-specific schema validation.
"""

import os
import logging
from typing import Any, Dict, List, Optional

from src.llm_only.config_validator import ConfigValidator, REQUIRED_SCHEMA

logger = logging.getLogger(__name__)

# Extend the base schema (excluding embeddings which are conditionally validated)
RAG_REQUIRED_SCHEMA = REQUIRED_SCHEMA.copy()
RAG_REQUIRED_SCHEMA.update({
    "chroma": {
        "collection_name": str,
        "persist_directory": str,
        "distance_metric": str,
        "embedding_dimension": int,
    },
    "retrieval": {
        "top_k": int,
        "score_threshold": (int, float),
        "metadata_filters": (dict, type(None)),
        "use_cache": bool,
    },
    "cache": {
        "redis_url": str,
        "ttl_seconds": int,
        "key_prefix": str,
    }
})

VALID_DISTANCE_METRICS = {"cosine", "l2", "ip"}


class RAGConfigValidator(ConfigValidator):
    """
    Validates RAG module configuration against the extended required schema.
    """

    def validate(self, dry_run_override: bool = None) -> Dict[str, Any]:
        """
        Validate the configuration.
        """
        self._errors = []
        
        self._validate_schema(self._config, RAG_REQUIRED_SCHEMA, prefix="")
        self._validate_value_constraints()
        self._validate_rag_constraints()
        self._validate_env_var(dry_run_override)
        
        if self._errors:
            from src.llm_only.config_validator import ConfigurationError
            raise ConfigurationError(self._errors)
            
        logger.info("RAG Configuration validation passed.")
        return self._config

    def _validate_rag_constraints(self) -> None:
        """Validate RAG-specific value ranges and allowed values."""
        chroma = self._config.get("chroma", {})
        embed = self._config.get("embeddings", {})
        retrieval = self._config.get("retrieval", {})
        cache = self._config.get("cache", {})
        
        # Embeddings conditional schema
        mode = embed.get("mode")
        if mode == "local":
            expected_keys = {
                "mode": str, 
                "model_name_or_path": str, 
                "device": str, 
                "batch_size": int, 
                "cache_dir": str, 
                "trust_remote_code": bool
            }
        elif mode == "api":
            expected_keys = {
                "mode": str, 
                "api_endpoint": str, 
                "api_key_env_var": str, 
                "model_id": str, 
                "normalize_embeddings": bool
            }
        else:
            self._errors.append("embeddings.mode must be 'local' or 'api'")
            expected_keys = {}
            
        for key, expected_type in expected_keys.items():
            if key not in embed:
                self._errors.append(f"Missing required key: 'embeddings.{key}' for mode '{mode}'")
            elif not isinstance(embed[key], expected_type):
                self._errors.append(f"Invalid type for 'embeddings.{key}': expected {expected_type.__name__}, got {type(embed[key]).__name__}")
        
        # Chroma constraints
        if isinstance(chroma.get("distance_metric"), str):
            if chroma["distance_metric"] not in VALID_DISTANCE_METRICS:
                self._errors.append(
                    f"chroma.distance_metric must be one of {VALID_DISTANCE_METRICS}, "
                    f"got '{chroma['distance_metric']}'"
                )
                
        if isinstance(chroma.get("persist_directory"), str):
            pd = chroma["persist_directory"]
            if not os.path.exists(pd):
                try:
                    os.makedirs(pd, exist_ok=True)
                except OSError as e:
                    self._errors.append(
                        f"chroma.persist_directory '{pd}' does not exist and could not be created: {e}"
                    )
                    
        # Retrieval constraints
        if isinstance(retrieval.get("top_k"), int):
            if retrieval["top_k"] <= 0:
                self._errors.append(
                    f"retrieval.top_k must be > 0, got {retrieval['top_k']}"
                )
                
        if isinstance(retrieval.get("score_threshold"), (int, float)):
            if not 0.0 <= retrieval["score_threshold"] <= 1.0:
                self._errors.append(
                    f"retrieval.score_threshold must be in [0.0, 1.0], got {retrieval['score_threshold']}"
                )
                
        # Cache constraints
        if isinstance(cache.get("ttl_seconds"), int):
            if cache["ttl_seconds"] <= 0:
                self._errors.append(
                    f"cache.ttl_seconds must be > 0, got {cache['ttl_seconds']}"
                )

    def _validate_env_var(self, dry_run_override: Optional[bool] = None) -> None:
        """Check that all required API key environment variables are set."""
        super()._validate_env_var(dry_run_override)
        
        embed = self._config.get("embeddings", {})
        if embed.get("mode") == "api":
            is_dry_run = dry_run_override if dry_run_override is not None else self._config.get("dry_run", False)
            env_var = embed.get("api_key_env_var", "")
            
            if not env_var:
                self._errors.append("embeddings.api_key_env_var must be a non-empty string for API mode")
                return
                
            if not os.environ.get(env_var):
                if is_dry_run:
                    logger.warning(
                        f"Environment variable '{env_var}' is not set. "
                        f"Proceeding in dry-run mode."
                    )
                else:
                    self._errors.append(
                        f"Environment variable '{env_var}' is not set. "
                        f"Set it or enable dry_run mode."
                    )
