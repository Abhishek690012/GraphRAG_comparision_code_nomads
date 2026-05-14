"""
LLM-Only Pipeline Orchestrator

Single entry-point class that wires together all sub-components and executes
the full API-based LLM-only inference flow.

Stateless per-call, async-compatible, fully configuration-driven.
"""

import asyncio
import copy
import json
import logging
import os
from typing import Any, Dict, List, Optional

import yaml

from .config_validator import ConfigValidator, ConfigurationError
from .token_counter import TokenCounter
from .prompt_formatter import PromptFormatter
from .rate_limiter import AsyncRateLimiter
from .api_client import LLMAPIClient
from .metrics import MetricsCollector

logger = logging.getLogger(__name__)


class LLMOnlyPipeline:
    """
    Orchestrates the full LLM-only API inference flow.

    Wires together: ConfigValidator → PromptFormatter → TokenCounter →
    RateLimiter → APIClient → MetricsCollector

    Usage:
        async with LLMOnlyPipeline("config/llm_only_config.yaml") as pipeline:
            result = await pipeline.run("What is diabetes?")
            print(json.dumps(result, indent=2))

        # Or batch mode:
        async with LLMOnlyPipeline("config/llm_only_config.yaml") as pipeline:
            results = await pipeline.run_batch(["Q1", "Q2", "Q3"])
            manifest = pipeline.generate_manifest()
    """

    def __init__(self, config_path: str, dry_run_override: Optional[bool] = None):
        """
        Initialize the pipeline from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file.
            dry_run_override: If provided, overrides the config's dry_run flag.

        Raises:
            ConfigurationError: If config validation fails.
            FileNotFoundError: If config file doesn't exist.
        """
        # Load config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

        # Apply dry-run override
        if dry_run_override is not None:
            self._config["dry_run"] = dry_run_override

        self._dry_run = self._config.get("dry_run", False)

        # Setup logging
        self._setup_logging()

        # Validate config
        validator = ConfigValidator(self._config)
        validator.validate(dry_run_override=self._dry_run)

        # Initialize components
        self._prompt_formatter = PromptFormatter(self._config["generation"])
        self._token_counter = TokenCounter(self._config["tokenizer"])
        self._rate_limiter = AsyncRateLimiter(self._config["rate_limiting"])
        self._api_client = LLMAPIClient(
            self._config["api"],
            self._rate_limiter,
            dry_run=self._dry_run,
        )
        self._metrics = MetricsCollector(
            pipeline_id=self._config["pipeline_id"],
            cost_config=self._config["cost"],
            api_provider=self._config["api"]["endpoint"],
        )

        logger.info(
            f"LLMOnlyPipeline initialized: "
            f"model={self._config['api']['model_id']}, "
            f"dry_run={self._dry_run}, "
            f"tokenizer={self._token_counter.backend}"
        )

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_config = self._config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
        log_file = log_config.get("file", "logs/llm_only.log")

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Configure the llm_only logger hierarchy
        llm_logger = logging.getLogger("src.llm_only")
        llm_logger.setLevel(log_level)

        # Avoid duplicate handlers
        if not llm_logger.handlers:
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            llm_logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            llm_logger.addHandler(console_handler)

    def _build_config_snapshot(self) -> Dict[str, Any]:
        """Build a sanitized config snapshot (no API keys) for the result."""
        snapshot = copy.deepcopy(self._config)
        # Redact the env var name (not even the var name in output)
        if "api" in snapshot:
            snapshot["api"]["api_key_env_var"] = "***REDACTED***"
        return snapshot

    async def run(
        self,
        query: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full LLM-only inference flow for a single query.

        Steps:
            1. Validate input
            2. Format prompt
            3. Count input tokens
            4. Check context window
            5. Execute API call (or dry-run)
            6. Extract/verify token counts
            7. Build structured result

        Args:
            query: The user's input query string.
            system_prompt: Optional override for the system prompt.

        Returns:
            Structured result dict matching the output specification.
            Never raises — errors are captured in the result.
        """
        config_snapshot = self._build_config_snapshot()

        # --- Step 1: Validate input ---
        try:
            PromptFormatter._validate_query(query)
        except ValueError as e:
            error_result = self._metrics.build_result(
                query=query or "",
                response=str(e),
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0.0,
                config_snapshot=config_snapshot,
                error_type="empty_query" if "empty" in str(e).lower() else "invalid_query",
            )
            self._metrics.record_query(error_result)
            return error_result

        # --- Step 2: Format prompt ---
        try:
            messages = self._prompt_formatter.format_prompt(query, system_prompt)
        except Exception as e:
            logger.error(f"Prompt formatting failed: {e}")
            error_result = self._metrics.build_result(
                query=query,
                response=f"Prompt formatting error: {e}",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0.0,
                config_snapshot=config_snapshot,
                error_type="prompt_formatting_error",
            )
            self._metrics.record_query(error_result)
            return error_result

        # --- Step 3: Count input tokens ---
        prompt_tokens = self._token_counter.count_messages_tokens(messages)

        # --- Step 4: Check context window ---
        max_tokens = self._config["generation"]["max_tokens"]
        window_check = self._token_counter.check_context_window(
            prompt_tokens, max_tokens
        )

        if not window_check["fits"]:
            if window_check["action"] == "reject":
                error_result = self._metrics.build_result(
                    query=query,
                    response=(
                        f"Context window exceeded: prompt ({prompt_tokens} tokens) + "
                        f"max_tokens ({max_tokens}) = {window_check['total_required']} "
                        f"> context_window ({self._token_counter.context_window})"
                    ),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    latency_ms=0.0,
                    config_snapshot=config_snapshot,
                    error_type="context_window_exceeded",
                )
                self._metrics.record_query(error_result)
                return error_result
            elif window_check["action"] == "truncate":
                # Reduce max_tokens to fit
                max_tokens = max(1, window_check["available"])
                logger.warning(
                    f"Truncating max_tokens to {max_tokens} "
                    f"to fit context window."
                )

        # --- Step 5: Execute API call ---
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

        # --- Step 6: Extract/verify token counts ---
        if api_response.success and not api_response.is_dry_run:
            # Use API-reported tokens when available
            final_prompt_tokens = api_response.prompt_tokens or prompt_tokens
            final_completion_tokens = api_response.completion_tokens

            # If API didn't report completion tokens, estimate client-side
            if final_completion_tokens == 0 and api_response.content:
                final_completion_tokens = self._token_counter.count_tokens(
                    api_response.content
                )
                logger.info(
                    f"API did not report completion tokens. "
                    f"Client estimate: {final_completion_tokens}"
                )

            # Validate alignment
            if api_response.prompt_tokens > 0:
                validation = TokenCounter.validate_against_api(
                    client_count=prompt_tokens,
                    api_count=api_response.prompt_tokens,
                )
                if not validation["aligned"]:
                    logger.warning(
                        f"Token count discrepancy: {validation}"
                    )
        else:
            final_prompt_tokens = api_response.prompt_tokens or prompt_tokens
            final_completion_tokens = api_response.completion_tokens

        # --- Step 7: Build structured result ---
        result = self._metrics.build_result(
            query=query,
            response=api_response.content if api_response.success else (
                api_response.error_message or "Unknown error"
            ),
            prompt_tokens=final_prompt_tokens,
            completion_tokens=final_completion_tokens,
            latency_ms=api_response.latency_ms,
            config_snapshot=config_snapshot,
            rate_limit_status=self._rate_limiter.api_rate_limit_status,
            error_type=api_response.error_type,
        )

        self._metrics.record_query(result)

        logger.info(
            f"Query completed: tokens={result['total_tokens']}, "
            f"latency={result['latency_ms']}ms, "
            f"cost=${result['cost_estimate']:.8f}, "
            f"error={result['error_type']}"
        )

        return result

    async def run_batch(
        self,
        queries: List[str],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute inference for multiple queries sequentially.

        Args:
            queries: List of query strings.
            system_prompt: Optional override applied to all queries.

        Returns:
            List of structured result dicts.
        """
        results = []
        total = len(queries)

        logger.info(f"Starting batch execution: {total} queries")

        for i, query in enumerate(queries):
            logger.info(f"Processing query {i + 1}/{total}")
            result = await self.run(query, system_prompt)
            results.append(result)

        logger.info(f"Batch execution complete: {total} queries processed")
        return results

    def generate_manifest(self) -> Dict[str, Any]:
        """
        Generate an execution manifest summarizing the session.

        Returns:
            Dict with session summary including averages, totals,
            error breakdown, and rate limit events.
        """
        return self._metrics.generate_manifest(
            rate_limiter_metrics=self._rate_limiter.get_metrics()
        )

    async def close(self) -> None:
        """Cleanup resources (API client session)."""
        await self._api_client.close()
        logger.info("LLMOnlyPipeline closed.")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
