"""
Prompt Formatter for LLM-Only Inference Module

Formats user queries and system prompts into the structured chat template
expected by Mistral-7B-Instruct-v0.3 and OpenAI-compatible API endpoints.

No API calls, no state — pure formatting logic.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default system prompt used when none is provided via config or call args
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant. "
    "Answer the user's question clearly and concisely."
)


class PromptFormatter:
    """
    Formats queries into the chat message structure required by the API.

    Produces OpenAI-compatible messages arrays:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Also provides raw [INST] template formatting for direct token counting.

    Usage:
        formatter = PromptFormatter(config["generation"])
        messages = formatter.format_prompt("What is diabetes?")
        raw_text = formatter.format_prompt_raw("What is diabetes?")
    """

    def __init__(self, generation_config: Dict[str, Any]):
        self._default_system_prompt = generation_config.get(
            "system_prompt", DEFAULT_SYSTEM_PROMPT
        )
        logger.info(
            f"PromptFormatter initialized with system prompt "
            f"({len(self._default_system_prompt)} chars)."
        )

    @property
    def default_system_prompt(self) -> str:
        """Return the configured default system prompt."""
        return self._default_system_prompt

    def format_prompt(
        self,
        query: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Format a query into an OpenAI-compatible messages array.

        Args:
            query: The user's input query string.
            system_prompt: Optional override for the system prompt.
                           If None, uses the configured default.

        Returns:
            List of message dicts with "role" and "content" keys.

        Raises:
            ValueError: If query is empty or not a string.
        """
        self._validate_query(query)

        effective_system_prompt = (
            system_prompt if system_prompt is not None else self._default_system_prompt
        )

        messages = []

        if effective_system_prompt:
            messages.append({
                "role": "system",
                "content": effective_system_prompt.strip(),
            })

        messages.append({
            "role": "user",
            "content": query.strip(),
        })

        logger.debug(
            f"Formatted prompt: {len(messages)} messages, "
            f"system={bool(effective_system_prompt)}"
        )
        return messages

    def format_prompt_raw(
        self,
        query: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Format a query into the raw Mistral-Instruct [INST] template string.

        Useful for direct token counting without message structure overhead.
        Follows the Mistral-Instruct v0.3 template format:
            <s>[INST] {system_prompt}\n\n{user_query} [/INST]

        Args:
            query: The user's input query string.
            system_prompt: Optional override for the system prompt.

        Returns:
            Formatted template string.

        Raises:
            ValueError: If query is empty or not a string.
        """
        self._validate_query(query)

        effective_system_prompt = (
            system_prompt if system_prompt is not None else self._default_system_prompt
        )

        if effective_system_prompt:
            content = f"{effective_system_prompt.strip()}\n\n{query.strip()}"
        else:
            content = query.strip()

        raw = f"<s>[INST] {content} [/INST]"

        logger.debug(f"Formatted raw prompt: {len(raw)} chars")
        return raw

    @staticmethod
    def _validate_query(query: str) -> None:
        """
        Validate that the query is a non-empty string.

        Raises:
            ValueError: If query is None, not a string, or empty/whitespace-only.
        """
        if query is None:
            raise ValueError("Query must not be None.")
        if not isinstance(query, str):
            raise ValueError(
                f"Query must be a string, got {type(query).__name__}."
            )
        if not query.strip():
            raise ValueError("Query must not be empty or whitespace-only.")
