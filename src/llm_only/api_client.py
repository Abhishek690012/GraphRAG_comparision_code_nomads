"""
Async API Client for LLM-Only Inference Module

Handles HTTP communication with OpenAI-compatible LLM API endpoints.
Implements retry logic, error classification, latency measurement,
and dry-run simulation.

Strictly separated from prompt formatting, token counting, and metrics.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

from .rate_limiter import AsyncRateLimiter

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Structured response from the LLM API call."""

    success: bool
    content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    rate_limit_status: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None
    is_dry_run: bool = False


# --- Error Classification ---

HTTP_ERROR_MAP = {
    400: "bad_request",
    401: "auth_failure",
    403: "auth_failure",
    404: "model_unavailable",
    422: "invalid_request",
    429: "rate_limited",
    500: "server_error",
    502: "server_error",
    503: "service_unavailable",
    504: "gateway_timeout",
}


class LLMAPIClient:
    """
    Async HTTP client for OpenAI-compatible LLM API endpoints.

    Features:
        - Configurable timeout and auth
        - Retry with exponential backoff on 429/5xx
        - Wall-clock latency measurement (excludes backoff waits)
        - Structured error responses (never raises to caller)
        - Dry-run mode for testing without network calls
        - Context manager support (async with)

    Usage:
        async with LLMAPIClient(api_config, rate_limiter) as client:
            response = await client.generate(messages, gen_params)
    """

    def __init__(
        self,
        api_config: Dict[str, Any],
        rate_limiter: AsyncRateLimiter,
        dry_run: bool = False,
    ):
        self._endpoint = api_config["endpoint"]
        self._model_id = api_config["model_id"]
        self._timeout = api_config["request_timeout_seconds"]
        self._dry_run = dry_run

        # Resolve API key from environment
        env_var = api_config["api_key_env_var"]
        self._api_key = os.environ.get(env_var, "")

        self._rate_limiter = rate_limiter
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(
            f"LLMAPIClient initialized: endpoint={self._endpoint}, "
            f"model={self._model_id}, timeout={self._timeout}s, "
            f"dry_run={self._dry_run}"
        )

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Create or return the aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
            )
        return self._session

    async def generate(
        self,
        messages: List[Dict[str, str]],
        generation_params: Dict[str, Any],
        estimated_prompt_tokens: int = 0,
    ) -> APIResponse:
        """
        Send a generation request to the API endpoint.

        Args:
            messages: OpenAI-compatible messages array.
            generation_params: Dict with max_tokens, temperature, top_p, etc.
            estimated_prompt_tokens: Client-side prompt token estimate (for dry-run).

        Returns:
            APIResponse with content, token counts, latency, and error info.
            Never raises exceptions — errors are captured in the response.
        """
        if self._dry_run:
            return self._simulate_response(messages, generation_params, estimated_prompt_tokens)

        return await self._execute_with_retry(messages, generation_params)

    def _simulate_response(
        self,
        messages: List[Dict[str, str]],
        generation_params: Dict[str, Any],
        estimated_prompt_tokens: int,
    ) -> APIResponse:
        """Generate a simulated response for dry-run mode."""
        simulated_completion = (
            "[DRY RUN] This is a simulated response. "
            "No actual API call was made."
        )
        # Estimate completion tokens from simulated text (~1 token per 4 chars)
        estimated_completion_tokens = max(1, len(simulated_completion) // 4)

        logger.info(
            f"Dry-run: simulated response with "
            f"~{estimated_prompt_tokens} prompt tokens, "
            f"~{estimated_completion_tokens} completion tokens."
        )

        return APIResponse(
            success=True,
            content=simulated_completion,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=estimated_completion_tokens,
            total_tokens=estimated_prompt_tokens + estimated_completion_tokens,
            latency_ms=0.0,
            error_type=None,
            is_dry_run=True,
        )

    async def _execute_with_retry(
        self,
        messages: List[Dict[str, str]],
        generation_params: Dict[str, Any],
    ) -> APIResponse:
        """Execute the API request with retry logic on transient failures."""
        payload = self._build_payload(messages, generation_params)

        for attempt in range(self._rate_limiter.max_retries + 1):
            # Acquire rate limit slot
            await self._rate_limiter.acquire()

            try:
                response = await self._send_request(payload)

                if response.success:
                    return response

                # Handle retryable errors
                if response.error_type == "rate_limited":
                    if self._rate_limiter.can_retry(attempt):
                        retry_after = self._extract_retry_after(response.raw_response)
                        delay = self._rate_limiter.handle_429(attempt, retry_after)
                        await asyncio.sleep(delay)
                        continue
                    else:
                        response.error_message = (
                            f"Rate limit exceeded after {attempt + 1} attempts. "
                            f"{response.error_message or ''}"
                        )
                        return response

                if response.error_type in ("server_error", "service_unavailable", "gateway_timeout"):
                    if self._rate_limiter.can_retry(attempt):
                        delay = self._rate_limiter.handle_429(attempt)
                        await asyncio.sleep(delay)
                        continue

                # Non-retryable error
                return response

            except asyncio.TimeoutError:
                logger.error(f"Request timeout (attempt {attempt + 1})")
                if self._rate_limiter.can_retry(attempt):
                    delay = self._rate_limiter.handle_429(attempt)
                    await asyncio.sleep(delay)
                    continue
                return APIResponse(
                    success=False,
                    error_type="timeout",
                    error_message=f"Request timed out after {self._timeout}s "
                                  f"({attempt + 1} attempts)",
                )

            except aiohttp.ClientError as e:
                logger.error(f"Client error (attempt {attempt + 1}): {e}")
                if self._rate_limiter.can_retry(attempt):
                    delay = self._rate_limiter.handle_429(attempt)
                    await asyncio.sleep(delay)
                    continue
                return APIResponse(
                    success=False,
                    error_type="connection_error",
                    error_message=f"Connection failed: {str(e)} "
                                  f"({attempt + 1} attempts)",
                )

            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
                return APIResponse(
                    success=False,
                    error_type="unexpected_error",
                    error_message=f"Unexpected error: {str(e)}",
                )

        # Exhausted all retries
        return APIResponse(
            success=False,
            error_type="max_retries_exceeded",
            error_message=f"All {self._rate_limiter.max_retries + 1} attempts failed.",
        )

    async def _send_request(self, payload: Dict[str, Any]) -> APIResponse:
        """
        Send a single HTTP request and parse the response.

        Measures wall-clock latency only around the actual HTTP call.
        """
        session = await self._ensure_session()

        start_time = time.perf_counter()

        async with session.post(self._endpoint, json=payload) as resp:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            status = resp.status
            response_headers = dict(resp.headers)

            # Update rate limit info from headers
            self._rate_limiter.update_api_rate_limit(response_headers)

            if status == 200:
                data = await resp.json()
                return self._parse_success_response(data, elapsed_ms)
            else:
                error_body = await resp.text()
                return self._parse_error_response(status, error_body, elapsed_ms)

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        generation_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the API request payload."""
        payload = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": generation_params.get("max_tokens", 1024),
            "temperature": generation_params.get("temperature", 0.7),
            "top_p": generation_params.get("top_p", 0.9),
        }

        # Add stop sequences if provided
        stop = generation_params.get("stop_sequences", [])
        if stop:
            payload["stop"] = stop

        # Add repetition penalty if supported (some providers use this)
        rep_penalty = generation_params.get("repetition_penalty")
        if rep_penalty is not None and rep_penalty != 1.0:
            payload["repetition_penalty"] = rep_penalty

        return payload

    @staticmethod
    def _parse_success_response(
        data: Dict[str, Any],
        latency_ms: float,
    ) -> APIResponse:
        """Parse a successful API response."""
        # Extract generated text
        choices = data.get("choices", [])
        content = ""
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")

        # Extract token usage
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        return APIResponse(
            success=True,
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 3),
            raw_response=data,
        )

    @staticmethod
    def _parse_error_response(
        status: int,
        body: str,
        latency_ms: float,
    ) -> APIResponse:
        """Parse an error API response with classification."""
        error_type = HTTP_ERROR_MAP.get(status, f"http_{status}")

        logger.error(
            f"API error: status={status}, type={error_type}, body={body[:200]}"
        )

        return APIResponse(
            success=False,
            latency_ms=round(latency_ms, 3),
            error_type=error_type,
            error_message=f"HTTP {status}: {body[:500]}",
            raw_response={"status": status, "body": body[:500]},
        )

    @staticmethod
    def _extract_retry_after(raw_response: Optional[Dict[str, Any]]) -> Optional[float]:
        """Extract Retry-After value from response if available."""
        if not raw_response:
            return None
        # Some APIs include retry_after in the error body
        retry_after = raw_response.get("retry_after")
        if retry_after is not None:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                return None
        return None

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("API client session closed.")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
