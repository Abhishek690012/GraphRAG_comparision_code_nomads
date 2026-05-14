"""
Async Rate Limiter for LLM-Only Inference Module

Implements client-side request throttling and adaptive backoff on HTTP 429
responses. Async-native, no threads.

Tracks throttle events for metrics reporting.
"""

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AsyncRateLimiter:
    """
    Async rate limiter with token-bucket-style client-side throttling
    and exponential backoff for server-side 429 responses.

    Usage:
        limiter = AsyncRateLimiter(config["rate_limiting"])
        await limiter.acquire()       # blocks if necessary to respect RPM
        delay = limiter.handle_429(retry_after=2.0, attempt=1)
        await asyncio.sleep(delay)
    """

    def __init__(self, rate_config: Dict[str, Any]):
        self._requests_per_minute = rate_config["requests_per_minute"]
        self._max_retries = rate_config["max_retries"]
        self._backoff_base = rate_config["backoff_base_seconds"]
        self._backoff_max = rate_config["backoff_max_seconds"]

        # Token bucket state
        self._interval = 60.0 / self._requests_per_minute  # seconds between requests
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()

        # Metrics tracking
        self._throttle_events: List[Dict[str, Any]] = []
        self._total_wait_seconds: float = 0.0
        self._requests_made: int = 0

        # API-reported rate limit info (updated from response headers)
        self._api_rate_limit: Optional[Dict[str, Any]] = None

        logger.info(
            f"AsyncRateLimiter initialized: {self._requests_per_minute} RPM, "
            f"max_retries={self._max_retries}, "
            f"backoff_base={self._backoff_base}s, "
            f"backoff_max={self._backoff_max}s"
        )

    @property
    def max_retries(self) -> int:
        """Maximum number of retry attempts on 429."""
        return self._max_retries

    @property
    def throttle_events(self) -> List[Dict[str, Any]]:
        """List of all recorded throttle events."""
        return list(self._throttle_events)

    @property
    def total_wait_seconds(self) -> float:
        """Total time spent waiting due to rate limiting."""
        return self._total_wait_seconds

    @property
    def requests_made(self) -> int:
        """Total number of requests that passed through the limiter."""
        return self._requests_made

    @property
    def api_rate_limit_status(self) -> Optional[Dict[str, Any]]:
        """Last known API-reported rate limit status."""
        return self._api_rate_limit

    async def acquire(self) -> None:
        """
        Acquire permission to make a request.

        Blocks (async sleep) if necessary to maintain the configured
        requests-per-minute rate. Uses a lock to ensure thread-safe
        sequential access.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time

            if elapsed < self._interval and self._last_request_time > 0:
                wait_time = self._interval - elapsed
                logger.debug(f"Rate limiter: waiting {wait_time:.3f}s before next request.")
                self._total_wait_seconds += wait_time
                await asyncio.sleep(wait_time)

            self._last_request_time = time.monotonic()
            self._requests_made += 1

    def handle_429(
        self,
        attempt: int,
        retry_after: Optional[float] = None,
    ) -> float:
        """
        Compute the backoff delay for a 429 rate-limit response.

        Uses exponential backoff with jitter, respecting the server's
        Retry-After header when available.

        Args:
            attempt: Current retry attempt (0-indexed).
            retry_after: Server-suggested retry delay in seconds (from
                         Retry-After header), or None if not provided.

        Returns:
            Delay in seconds to wait before retrying.
        """
        if retry_after is not None and retry_after > 0:
            # Respect server's Retry-After, but cap at max
            delay = min(retry_after, self._backoff_max)
        else:
            # Exponential backoff with full jitter
            base_delay = self._backoff_base * (2 ** attempt)
            delay = min(base_delay, self._backoff_max)
            # Add jitter: uniform random in [0, delay]
            delay = random.uniform(0, delay)

        # Record the throttle event
        event = {
            "timestamp": time.time(),
            "attempt": attempt,
            "delay_seconds": round(delay, 3),
            "retry_after_header": retry_after,
            "type": "http_429",
        }
        self._throttle_events.append(event)
        self._total_wait_seconds += delay

        logger.warning(
            f"Rate limited (429): attempt {attempt + 1}/{self._max_retries}, "
            f"backing off {delay:.3f}s "
            f"(retry_after={retry_after})"
        )

        return delay

    def update_api_rate_limit(self, headers: Dict[str, str]) -> None:
        """
        Update rate limit status from API response headers.

        Extracts standard rate limit headers:
            - x-ratelimit-remaining
            - x-ratelimit-reset
            - x-ratelimit-limit

        Args:
            headers: HTTP response headers dict.
        """
        remaining = headers.get("x-ratelimit-remaining")
        reset_time = headers.get("x-ratelimit-reset")
        limit = headers.get("x-ratelimit-limit")

        if any(v is not None for v in [remaining, reset_time, limit]):
            self._api_rate_limit = {
                "requests_remaining": int(remaining) if remaining else None,
                "reset_time": reset_time,
                "limit": int(limit) if limit else None,
            }
            logger.debug(f"API rate limit status updated: {self._api_rate_limit}")

    def can_retry(self, attempt: int) -> bool:
        """
        Check if another retry attempt is allowed.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            True if attempt < max_retries.
        """
        return attempt < self._max_retries

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return rate limiter metrics summary.

        Returns:
            Dict with throttle event count, total wait time, and
            current API rate limit status.
        """
        return {
            "throttle_event_count": len(self._throttle_events),
            "total_wait_seconds": round(self._total_wait_seconds, 3),
            "requests_made": self._requests_made,
            "api_rate_limit_status": self._api_rate_limit,
            "throttle_events": self._throttle_events,
        }
