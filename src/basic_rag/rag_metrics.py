"""
RAG Metrics Collector for Basic RAG Module

Extends the base MetricsCollector with retrieval-specific metrics.
"""

import logging
from typing import Any, Dict, List, Optional

from src.llm_only.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class RAGMetricsCollector(MetricsCollector):
    """
    Collects and aggregates inference metrics including retrieval stats.
    """

    def build_rag_result(
        self,
        query: str,
        response: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        retrieval_latency_ms: float,
        retrieval_metadata: Dict[str, Any],
        config_snapshot: Dict[str, Any],
        rate_limit_status: Optional[Dict[str, Any]] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build the structured per-query result dict with RAG extensions.
        """
        # Call base class to get standard fields
        result = super().build_result(
            query=query,
            response=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            config_snapshot=config_snapshot,
            rate_limit_status=rate_limit_status,
            error_type=error_type,
        )
        
        # Add RAG-specific fields
        result["retrieval_latency_ms"] = round(retrieval_latency_ms, 3)
        result["retrieval_metadata"] = retrieval_metadata
        
        logger.debug(
            f"Built RAG result: total_latency={latency_ms:.1f}ms, "
            f"retrieval_latency={retrieval_latency_ms:.1f}ms, "
            f"chunks={retrieval_metadata.get('chunks_retrieved', 0)}"
        )
        return result

    def generate_manifest(
        self,
        rate_limiter_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an execution manifest summarizing the session, 
        including retrieval quality stats.
        """
        # Get base manifest
        manifest = super().generate_manifest(rate_limiter_metrics)
        
        successful_rag = [
            r for r in self._query_results 
            if r.get("error_type") is None and "retrieval_metadata" in r
        ]
        
        if not successful_rag:
            manifest.update({
                "avg_chunks_retrieved": 0.0,
                "avg_similarity_score": 0.0,
                "cache_hit_ratio": 0.0,
                "avg_retrieval_latency_ms": 0.0,
            })
            return manifest
            
        total_chunks = sum(r["retrieval_metadata"].get("chunks_retrieved", 0) for r in successful_rag)
        
        # Average similarity score (only for queries that retrieved chunks)
        valid_scores = [r["retrieval_metadata"].get("avg_similarity_score", 0.0) 
                        for r in successful_rag if r["retrieval_metadata"].get("chunks_retrieved", 0) > 0]
        avg_sim = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        # Cache hit ratio
        cache_hits = sum(1 for r in successful_rag if r["retrieval_metadata"].get("cache_hit", False))
        hit_ratio = cache_hits / len(successful_rag)
        
        # Avg retrieval latency
        avg_ret_lat = sum(r.get("retrieval_latency_ms", 0.0) for r in successful_rag) / len(successful_rag)
        
        manifest.update({
            "avg_chunks_retrieved": round(total_chunks / len(successful_rag), 2),
            "avg_similarity_score": round(avg_sim, 4),
            "cache_hit_ratio": round(hit_ratio, 2),
            "avg_retrieval_latency_ms": round(avg_ret_lat, 3),
        })
        
        # Log retrieval quality warnings
        if avg_sim < 0.5 and total_chunks > 0:
            logger.warning("Retrieval Quality Warning: Average similarity score is low (< 0.5)")
            
        return manifest
