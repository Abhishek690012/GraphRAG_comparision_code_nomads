"""
Token Counter for LLM-Only Inference Module

Provides deterministic token counting aligned with the target model's tokenizer.
Supports multiple tokenizer backends: mistral-common, transformers, tiktoken.
Stateless after initialization.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Deterministic token counter with configurable backend.

    Supported backends:
        - "mistral-common": Mistral's official tokenizer (exact for Mistral models)
        - "transformers": HuggingFace AutoTokenizer (exact but heavier)
        - "tiktoken_cl100k": OpenAI's tiktoken cl100k_base (approximation)

    Usage:
        counter = TokenCounter(config["tokenizer"])
        count = counter.count_tokens("Hello, world!")
        fits = counter.check_context_window(prompt_tokens=100, max_tokens=1024)
    """

    def __init__(self, tokenizer_config: Dict[str, Any]):
        self._model_name = tokenizer_config["model"]
        self._context_window = tokenizer_config["context_window"]
        self._truncation_strategy = tokenizer_config.get("truncation_strategy", "reject")
        self._tokenizer = None

        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        """Initialize the tokenizer backend based on configuration."""
        if self._model_name == "mistral-common":
            self._init_mistral_common()
        elif self._model_name == "transformers":
            self._init_transformers()
        elif self._model_name == "tiktoken_cl100k":
            self._init_tiktoken()
        else:
            raise ValueError(
                f"Unsupported tokenizer model: '{self._model_name}'. "
                f"Must be one of: mistral-common, transformers, tiktoken_cl100k"
            )

    def _init_mistral_common(self) -> None:
        """Initialize Mistral's official tokenizer."""
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            self._tokenizer = MistralTokenizer.v3()
            self._backend = "mistral-common"
            logger.info("Initialized mistral-common tokenizer (v3).")
        except ImportError:
            logger.warning(
                "mistral-common not installed. "
                "Falling back to tiktoken cl100k_base approximation."
            )
            self._init_tiktoken()

    def _init_transformers(self) -> None:
        """Initialize HuggingFace AutoTokenizer."""
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3"
            )
            self._backend = "transformers"
            logger.info("Initialized HuggingFace AutoTokenizer for Mistral-7B-Instruct-v0.3.")
        except ImportError:
            logger.warning(
                "transformers not installed. "
                "Falling back to tiktoken cl100k_base approximation."
            )
            self._init_tiktoken()

    def _init_tiktoken(self) -> None:
        """Initialize tiktoken cl100k_base as fallback."""
        import tiktoken
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._backend = "tiktoken_cl100k"
        logger.info("Initialized tiktoken cl100k_base tokenizer (approximation).")

    @property
    def backend(self) -> str:
        """Return the name of the active tokenizer backend."""
        return self._backend

    @property
    def context_window(self) -> int:
        """Return the configured context window size."""
        return self._context_window

    @property
    def truncation_strategy(self) -> str:
        """Return the configured truncation strategy."""
        return self._truncation_strategy

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text using the configured tokenizer.

        Args:
            text: Input text to tokenize.

        Returns:
            Exact or estimated token count.
        """
        if not text:
            return 0

        if self._backend == "mistral-common":
            from mistral_common.protocol.instruct.messages import UserMessage
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            # For raw text counting, we wrap in a minimal request
            request = ChatCompletionRequest(
                messages=[UserMessage(content=text)]
            )
            encoded = self._tokenizer.encode_chat_completion(request)
            return len(encoded.tokens)

        elif self._backend == "transformers":
            tokens = self._tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)

        elif self._backend == "tiktoken_cl100k":
            tokens = self._tokenizer.encode(text)
            return len(tokens)

        return 0

    def count_messages_tokens(self, messages: list) -> int:
        """
        Count tokens for a full messages array (chat format).

        Accounts for chat template overhead (role tokens, special tokens).

        Args:
            messages: List of {"role": str, "content": str} dicts.

        Returns:
            Total token count including template overhead.
        """
        if self._backend == "mistral-common":
            from mistral_common.protocol.instruct.messages import (
                UserMessage,
                SystemMessage,
                AssistantMessage,
            )
            from mistral_common.protocol.instruct.request import ChatCompletionRequest

            mistral_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    mistral_messages.append(SystemMessage(content=content))
                elif role == "user":
                    mistral_messages.append(UserMessage(content=content))
                elif role == "assistant":
                    mistral_messages.append(AssistantMessage(content=content))

            request = ChatCompletionRequest(messages=mistral_messages)
            encoded = self._tokenizer.encode_chat_completion(request)
            return len(encoded.tokens)

        else:
            # For non-mistral backends, estimate overhead per message
            # ~4 tokens per message for role/separators (OpenAI convention)
            total = 0
            for msg in messages:
                total += self.count_tokens(msg.get("content", ""))
                total += 4  # role + separators overhead
            total += 2  # priming tokens
            return total

    def check_context_window(
        self,
        prompt_tokens: int,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """
        Validate that prompt + max generation tokens fit within the context window.

        Args:
            prompt_tokens: Number of tokens in the prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            Dict with keys:
                - fits (bool): Whether the request fits
                - total_required (int): prompt_tokens + max_tokens
                - available (int): context_window - prompt_tokens
                - action (str): "ok", "reject", or "truncate"
        """
        total_required = prompt_tokens + max_tokens
        available = self._context_window - prompt_tokens

        if total_required <= self._context_window:
            return {
                "fits": True,
                "total_required": total_required,
                "available": available,
                "action": "ok",
            }

        return {
            "fits": False,
            "total_required": total_required,
            "available": max(0, available),
            "action": self._truncation_strategy,
        }

    @staticmethod
    def validate_against_api(
        client_count: int,
        api_count: int,
        tolerance: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Compare client-side token count with API-reported count.

        Args:
            client_count: Token count computed client-side.
            api_count: Token count reported by the API.
            tolerance: Acceptable relative discrepancy (default 10%).

        Returns:
            Dict with discrepancy info for logging.
        """
        if api_count == 0:
            return {
                "aligned": True,
                "client_count": client_count,
                "api_count": api_count,
                "discrepancy": 0,
                "discrepancy_pct": 0.0,
                "note": "API did not report token count.",
            }

        discrepancy = abs(client_count - api_count)
        discrepancy_pct = discrepancy / api_count if api_count > 0 else 0.0
        aligned = discrepancy_pct <= tolerance

        result = {
            "aligned": aligned,
            "client_count": client_count,
            "api_count": api_count,
            "discrepancy": discrepancy,
            "discrepancy_pct": round(discrepancy_pct * 100, 2),
        }

        if not aligned:
            logger.warning(
                f"Token count discrepancy: client={client_count}, "
                f"api={api_count}, diff={discrepancy} ({result['discrepancy_pct']}%)"
            )
            result["note"] = "Discrepancy exceeds tolerance threshold."
        else:
            result["note"] = "Token counts aligned within tolerance."

        return result
