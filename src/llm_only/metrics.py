"""
Metrics Collector for LLM-Only Inference Module

Aggregates per-query and session-level metrics including token counts,
latency, cost estimation, rate limit events, and error classifications.

Pure data aggregation — no I/O, no side effects.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and aggregates inference metrics for the LLM-only pipeline.

    Builds structured per-query result dicts and generates session-level
    execution manifests.

    Usage:
        collector = MetricsCollector(pipeline_id, cost_config)
        result = collector.build_result(query=..., response=..., ...)
        collector.record_query(result)
        manifest = collector.generate_manifest()
    """

    def __init__(
        self,
        pipeline_id: str,
        cost_config: Dict[str, Any],
        api_provider: str,
    ):
        self._pipeline_id = pipeline_id
        self._cost_config = cost_config
        self._api_provider = api_provider
        self._query_results: List[Dict[str, Any]] = []

        logger.info(
            f"MetricsCollector initialized: pipeline_id={pipeline_id}, "
            f"api_provider={api_provider}"
        )

    @property
    def query_count(self) -> int:
        """Number of queries recorded."""
        return len(self._query_results)

    def compute_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """
        Calculate theoretical cost per query.

        Uses configurable pricing per million input/output tokens.

        Args:
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.

        Returns:
            Cost estimate in dollars.
        """
        input_cost = (
            prompt_tokens
            * self._cost_config["per_million_input_tokens"]
            / 1_000_000
        )
        output_cost = (
            completion_tokens
            * self._cost_config["per_million_output_tokens"]
            / 1_000_000
        )
        total = round(input_cost + output_cost, 10)
        return total

    def build_result(
        self,
        query: str,
        response: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        config_snapshot: Dict[str, Any],
        rate_limit_status: Optional[Dict[str, Any]] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build the structured per-query result dict.

        Matches the output specification exactly.

        Args:
            query: Original input query string.
            response: Generated text or error message.
            prompt_tokens: Exact or estimated input token count.
            completion_tokens: Exact or estimated output token count.
            latency_ms: End-to-end API round-trip time in milliseconds.
            config_snapshot: Generation parameters and API settings applied.
            rate_limit_status: Requests remaining / reset time if available.
            error_type: Null or classification of failure.

        Returns:
            Structured result dict matching the output specification.
        """
        total_tokens = prompt_tokens + completion_tokens
        cost_estimate = self.compute_cost(prompt_tokens, completion_tokens)

        result = {
            "pipeline_id": self._pipeline_id,
            "query": query,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": round(latency_ms, 3),
            "cost_estimate": cost_estimate,
            "api_provider": self._api_provider,
            "rate_limit_status": rate_limit_status,
            "error_type": error_type,
            "config_snapshot": config_snapshot,
        }

        logger.debug(
            f"Built result: tokens={total_tokens}, "
            f"latency={latency_ms:.1f}ms, cost=${cost_estimate:.8f}, "
            f"error={error_type}"
        )
        return result

    def record_query(self, result: Dict[str, Any]) -> None:
        """
        Record a query result for manifest aggregation.

        Args:
            result: Structured result dict from build_result().
        """
        self._query_results.append(result)
        logger.debug(f"Recorded query {len(self._query_results)}: {result.get('query', '')[:50]}...")

    def generate_manifest(
        self,
        rate_limiter_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an execution manifest summarizing the session.

        Args:
            rate_limiter_metrics: Metrics from the rate limiter.

        Returns:
            Dict with session summary including averages, totals,
            error breakdown, and rate limit events.
        """
        total_queries = len(self._query_results)

        if total_queries == 0:
            return {
                "pipeline_id": self._pipeline_id,
                "api_provider": self._api_provider,
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "avg_prompt_tokens": 0.0,
                "avg_completion_tokens": 0.0,
                "avg_total_tokens": 0.0,
                "avg_latency_ms": 0.0,
                "total_cost_estimate": 0.0,
                "rate_limit_events": 0,
                "error_classifications": {},
                "rate_limiter_metrics": rate_limiter_metrics or {},
            }

        # Separate successful and failed queries
        successful = [r for r in self._query_results if r.get("error_type") is None]
        failed = [r for r in self._query_results if r.get("error_type") is not None]

        # Compute averages from successful queries
        if successful:
            avg_prompt = sum(r["prompt_tokens"] for r in successful) / len(successful)
            avg_completion = sum(r["completion_tokens"] for r in successful) / len(successful)
            avg_total = sum(r["total_tokens"] for r in successful) / len(successful)
            avg_latency = sum(r["latency_ms"] for r in successful) / len(successful)
        else:
            avg_prompt = avg_completion = avg_total = avg_latency = 0.0

        # Total cost across all queries
        total_cost = sum(r.get("cost_estimate", 0.0) for r in self._query_results)

        # Error classification breakdown
        error_classifications: Dict[str, int] = {}
        for r in failed:
            err = r.get("error_type", "unknown")
            error_classifications[err] = error_classifications.get(err, 0) + 1

        # Rate limit event count
        rate_limit_events = 0
        if rate_limiter_metrics:
            rate_limit_events = rate_limiter_metrics.get("throttle_event_count", 0)

        manifest = {
            "pipeline_id": self._pipeline_id,
            "api_provider": self._api_provider,
            "total_queries": total_queries,
            "successful_queries": len(successful),
            "failed_queries": len(failed),
            "avg_prompt_tokens": round(avg_prompt, 2),
            "avg_completion_tokens": round(avg_completion, 2),
            "avg_total_tokens": round(avg_total, 2),
            "avg_latency_ms": round(avg_latency, 3),
            "total_cost_estimate": round(total_cost, 8),
            "rate_limit_events": rate_limit_events,
            "error_classifications": error_classifications,
            "rate_limiter_metrics": rate_limiter_metrics or {},
        }

        logger.info(f"Generated execution manifest: {total_queries} queries processed.")
        return manifest

    def reset(self) -> None:
        """Clear all recorded query results."""
        self._query_results = []
        logger.info("MetricsCollector reset.")
