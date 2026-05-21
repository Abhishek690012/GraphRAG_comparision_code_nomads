"""
Basic RAG Pipeline Orchestrator

Single entry-point class that wires together all sub-components and executes
the full API-based Basic RAG flow.
"""

import asyncio
import copy
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import yaml

from .rag_config_validator import RAGConfigValidator
from .cache_manager import CacheManager
from .embedding_engine import EmbeddingEngine
from .vector_store import VectorStoreClient
from .rag_prompt_formatter import RAGPromptFormatter
from .rag_metrics import RAGMetricsCollector

from src.llm_only.token_counter import TokenCounter
from src.llm_only.rate_limiter import AsyncRateLimiter
from src.llm_only.api_client import LLMAPIClient

logger = logging.getLogger(__name__)


class BasicRAGPipeline:
    """
    Orchestrates the full Basic RAG API inference flow.
    """

    def __init__(self, config_path: str, dry_run_override: Optional[bool] = None):
        """Initialize the pipeline from a YAML configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

        if dry_run_override is not None:
            self._config["dry_run"] = dry_run_override

        self._dry_run = self._config.get("dry_run", False)

        self._setup_logging()

        # Validate config
        validator = RAGConfigValidator(self._config)
        validator.validate(dry_run_override=self._dry_run)
        
        # Determine config hash for cache keying
        config_str = json.dumps({
            "retrieval": self._config["retrieval"],
            "generation": self._config["generation"],
            "api": {"model_id": self._config["api"]["model_id"]}
        }, sort_keys=True)
        self._config_hash = hashlib.md5(config_str.encode()).hexdigest()

        # Initialize components
        self._cache_manager = CacheManager(self._config["cache"])
        self._use_cache = self._config["retrieval"].get("use_cache", True)
        
        self._embedding_engine = EmbeddingEngine(self._config["embeddings"], self._cache_manager)
        self._vector_store = VectorStoreClient(self._config["chroma"])
        
        self._prompt_formatter = RAGPromptFormatter(self._config["generation"])
        self._token_counter = TokenCounter(self._config["tokenizer"])
        self._rate_limiter = AsyncRateLimiter(self._config["rate_limiting"])
        
        self._api_client = LLMAPIClient(
            self._config["api"],
            self._rate_limiter,
            dry_run=self._dry_run,
        )
        
        self._metrics = RAGMetricsCollector(
            pipeline_id=self._config["pipeline_id"],
            cost_config=self._config["cost"],
            api_provider=self._config["api"]["endpoint"],
        )

        logger.info(
            f"BasicRAGPipeline initialized: "
            f"model={self._config['api']['model_id']}, "
            f"dry_run={self._dry_run}, "
            f"collection={self._config['chroma']['collection_name']}"
        )

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_config = self._config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
        log_file = log_config.get("file", "logs/basic_rag.log")

        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        rag_logger = logging.getLogger("src.basic_rag")
        rag_logger.setLevel(log_level)

        if not rag_logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            rag_logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            rag_logger.addHandler(console_handler)

    def _build_config_snapshot(self) -> Dict[str, Any]:
        """Build a sanitized config snapshot (no API keys) for the result."""
        snapshot = copy.deepcopy(self._config)
        if "api" in snapshot:
            snapshot["api"]["api_key_env_var"] = "***REDACTED***"
        return snapshot

    async def run(
        self,
        query: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the full Basic RAG inference flow for a single query."""
        config_snapshot = self._build_config_snapshot()
        retrieval_metadata = {
            "chunks_retrieved": 0,
            "avg_similarity_score": 0.0,
            "cache_hit": False,
            "source_doc_ids": []
        }
        retrieval_latency_ms = 0.0
        
        # --- Step 1: Validate input ---
        try:
            self._prompt_formatter._validate_query(query)
        except ValueError as e:
            error_result = self._metrics.build_rag_result(
                query=query or "", response=str(e),
                prompt_tokens=0, completion_tokens=0,
                latency_ms=0.0, retrieval_latency_ms=0.0,
                retrieval_metadata=retrieval_metadata,
                config_snapshot=config_snapshot,
                error_type="empty_query" if "empty" in str(e).lower() else "invalid_query",
            )
            self._metrics.record_query(error_result)
            return error_result

        # --- Step 2: Response Cache Check ---
        response_cache_key = self._cache_manager.generate_cache_key("response", query, self._config_hash)
        if self._use_cache:
            cached_response = await self._cache_manager.get(response_cache_key)
            if cached_response:
                logger.info("Response cache hit.")
                # Record as cache hit
                cached_response["retrieval_metadata"]["cache_hit"] = True
                self._metrics.record_query(cached_response)
                return cached_response

        # --- Step 3: Retrieval ---
        retrieval_start_time = time.perf_counter()
        retrieved_chunks = []
        
        retrieval_cache_key = self._cache_manager.generate_cache_key("retrieval", query, self._config_hash)
        cached_retrieval = None
        
        if self._use_cache:
            cached_retrieval = await self._cache_manager.get(retrieval_cache_key)
            
        if cached_retrieval is not None:
            logger.info("Retrieval cache hit.")
            from .vector_store import RetrievalResult
            retrieved_chunks = [RetrievalResult(**c) for c in cached_retrieval]
            retrieval_metadata["cache_hit"] = True
        else:
            try:
                # Embed query
                query_embedding = self._embedding_engine.encode(query)
                
                # Search ChromaDB
                retrieved_chunks = self._vector_store.search(
                    query_embedding=query_embedding,
                    k=self._config["retrieval"]["top_k"],
                    score_threshold=self._config["retrieval"]["score_threshold"],
                    metadata_filters=self._config["retrieval"]["metadata_filters"]
                )
                
                # Cache retrieval results
                if self._use_cache and retrieved_chunks:
                    chunks_dicts = [
                        {"chunk_id": c.chunk_id, "text": c.text, "score": c.score, 
                         "metadata": c.metadata, "source_doc_id": c.source_doc_id} 
                        for c in retrieved_chunks
                    ]
                    await self._cache_manager.set(retrieval_cache_key, chunks_dicts)
                    
            except Exception as e:
                logger.error(f"Retrieval error: {e}")
                # We log but continue, allowing fallback prompt
                pass
                
        retrieval_latency_ms = (time.perf_counter() - retrieval_start_time) * 1000
        
        # Populate retrieval metadata
        retrieval_metadata["chunks_retrieved"] = len(retrieved_chunks)
        if retrieved_chunks:
            avg_score = sum(c.score for c in retrieved_chunks) / len(retrieved_chunks)
            retrieval_metadata["avg_similarity_score"] = round(avg_score, 4)
            # Maintain order but remove duplicates for source list
            sources = []
            for c in retrieved_chunks:
                if c.source_doc_id not in sources:
                    sources.append(c.source_doc_id)
            retrieval_metadata["source_doc_ids"] = sources

        # --- Step 4: Format RAG prompt ---
        try:
            max_context_tokens = self._config["tokenizer"].get("max_context_tokens", 4096)
            messages = self._prompt_formatter.format_rag_prompt(
                query=query, 
                retrieved_chunks=retrieved_chunks,
                system_prompt=system_prompt,
                max_context_tokens=max_context_tokens
            )
        except Exception as e:
            logger.error(f"Prompt formatting failed: {e}")
            error_result = self._metrics.build_rag_result(
                query=query, response=f"Prompt formatting error: {e}",
                prompt_tokens=0, completion_tokens=0,
                latency_ms=0.0, retrieval_latency_ms=retrieval_latency_ms,
                retrieval_metadata=retrieval_metadata,
                config_snapshot=config_snapshot,
                error_type="prompt_formatting_error",
            )
            self._metrics.record_query(error_result)
            return error_result

        # --- Step 5: Count input tokens and check context window ---
        prompt_tokens = self._token_counter.count_messages_tokens(messages)
        max_tokens = self._config["generation"]["max_tokens"]
        window_check = self._token_counter.check_context_window(prompt_tokens, max_tokens)

        if not window_check["fits"]:
            if window_check["action"] == "reject":
                error_result = self._metrics.build_rag_result(
                    query=query,
                    response=f"Context window exceeded",
                    prompt_tokens=prompt_tokens, completion_tokens=0,
                    latency_ms=0.0, retrieval_latency_ms=retrieval_latency_ms,
                    retrieval_metadata=retrieval_metadata,
                    config_snapshot=config_snapshot,
                    error_type="context_window_exceeded",
                )
                self._metrics.record_query(error_result)
                return error_result
            elif window_check["action"] == "truncate":
                max_tokens = max(1, window_check["available"])

        # --- Step 6: Execute API call ---
        generation_params = {
            "max_tokens": max_tokens,
            "temperature": self._config["generation"]["temperature"],
            "top_p": self._config["generation"]["top_p"],
            "repetition_penalty": self._config["generation"]["repetition_penalty"],
            "stop_sequences": self._config["generation"]["stop_sequences"],
        }

        api_response = await self._api_client.generate(
            messages=messages,
            generation_params=generation_params,
            estimated_prompt_tokens=prompt_tokens,
        )

        # --- Step 7: Final token counts and metrics ---
        final_prompt_tokens = api_response.prompt_tokens or prompt_tokens
        final_completion_tokens = api_response.completion_tokens
        
        if api_response.success and not api_response.is_dry_run and final_completion_tokens == 0 and api_response.content:
            final_completion_tokens = self._token_counter.count_tokens(api_response.content)

        # Total latency is retrieval + API latency (approx, ignoring formatting overhead)
        total_latency_ms = retrieval_latency_ms + api_response.latency_ms

        result = self._metrics.build_rag_result(
            query=query,
            response=api_response.content if api_response.success else (api_response.error_message or "Unknown error"),
            prompt_tokens=final_prompt_tokens,
            completion_tokens=final_completion_tokens,
            latency_ms=total_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            config_snapshot=config_snapshot,
            rate_limit_status=self._rate_limiter.api_rate_limit_status,
            error_type=api_response.error_type,
            retrieval_metadata=retrieval_metadata
        )

        self._metrics.record_query(result)
        
        # Cache successful response
        if api_response.success and not api_response.is_dry_run and self._use_cache:
            await self._cache_manager.set(response_cache_key, result)

        return result

    async def run_batch(
        self,
        queries: List[str],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute inference for multiple queries sequentially."""
        results = []
        total = len(queries)
        logger.info(f"Starting batch execution: {total} queries")
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i + 1}/{total}")
            result = await self.run(query, system_prompt)
            results.append(result)
        return results

    def generate_manifest(self) -> Dict[str, Any]:
        """Generate an execution manifest summarizing the session."""
        return self._metrics.generate_manifest(
            rate_limiter_metrics=self._rate_limiter.get_metrics()
        )

    async def close(self) -> None:
        """Cleanup resources."""
        await self._api_client.close()
        await self._cache_manager.close()
        logger.info("BasicRAGPipeline closed.")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
