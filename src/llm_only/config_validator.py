"""
Configuration Validator for LLM-Only Inference Module

Validates the full YAML configuration schema before any component initialization.
Pure validation — no side effects, no I/O beyond env var checks.
"""

import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        super().__init__(message)


# --- Schema Definition ---

REQUIRED_SCHEMA = {
    "pipeline_id": str,
    "api": {
        "endpoint": str,
        "api_key_env_var": str,
        "model_id": str,
        "request_timeout_seconds": (int, float),
    },
    "generation": {
        "max_tokens": int,
        "temperature": (int, float),
        "top_p": (int, float),
        "repetition_penalty": (int, float),
        "stop_sequences": list,
        "system_prompt": str,
    },
    "tokenizer": {
        "model": str,
        "context_window": int,
        "truncation_strategy": str,
    },
    "cost": {
        "per_million_input_tokens": (int, float),
        "per_million_output_tokens": (int, float),
    },
    "rate_limiting": {
        "max_retries": int,
        "backoff_base_seconds": (int, float),
        "backoff_max_seconds": (int, float),
        "requests_per_minute": int,
    },
    "dry_run": bool,
    "logging": {
        "level": str,
        "file": str,
    },
}

VALID_TOKENIZER_MODELS = {"mistral-common", "transformers", "tiktoken_cl100k"}
VALID_TRUNCATION_STRATEGIES = {"reject", "truncate"}
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigValidator:
    """
    Validates LLM-only module configuration against the required schema.

    Usage:
        validator = ConfigValidator(config_dict)
        validated_config = validator.validate()  # raises ConfigurationError on failure
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._errors: List[str] = []

    def validate(self, dry_run_override: Optional[bool] = None) -> Dict[str, Any]:
        """
        Validate the configuration and return it if valid.

        Args:
            dry_run_override: If provided, overrides the config's dry_run setting
                              for env var validation purposes.

        Returns:
            The validated configuration dict.

        Raises:
            ConfigurationError: If validation fails, with all errors collected.
        """
        self._errors = []

        self._validate_schema(self._config, REQUIRED_SCHEMA, prefix="")
        self._validate_value_constraints()
        self._validate_env_var(dry_run_override)

        if self._errors:
            raise ConfigurationError(self._errors)

        logger.info("Configuration validation passed.")
        return self._config

    def _validate_schema(
        self,
        config: Any,
        schema: Any,
        prefix: str,
    ) -> None:
        """Recursively validate that config matches the expected schema structure."""
        if isinstance(schema, dict):
            if not isinstance(config, dict):
                self._errors.append(
                    f"{prefix or 'root'}: expected a mapping, got {type(config).__name__}"
                )
                return
            for key, expected_type in schema.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if key not in config:
                    self._errors.append(f"Missing required key: '{full_key}'")
                else:
                    self._validate_schema(config[key], expected_type, full_key)
        elif isinstance(schema, tuple):
            # Multiple acceptable types
            if not isinstance(config, schema):
                type_names = "/".join(t.__name__ for t in schema)
                self._errors.append(
                    f"'{prefix}': expected type {type_names}, "
                    f"got {type(config).__name__}"
                )
        elif isinstance(schema, type):
            if not isinstance(config, schema):
                self._errors.append(
                    f"'{prefix}': expected type {schema.__name__}, "
                    f"got {type(config).__name__}"
                )

    def _validate_value_constraints(self) -> None:
        """Validate value ranges and allowed values."""
        gen = self._config.get("generation", {})
        tok = self._config.get("tokenizer", {})
        rate = self._config.get("rate_limiting", {})
        cost = self._config.get("cost", {})
        log = self._config.get("logging", {})
        api = self._config.get("api", {})

        # Generation constraints
        if isinstance(gen.get("temperature"), (int, float)):
            if not 0 <= gen["temperature"] <= 2:
                self._errors.append(
                    f"generation.temperature must be in [0, 2], got {gen['temperature']}"
                )

        if isinstance(gen.get("top_p"), (int, float)):
            if not 0 <= gen["top_p"] <= 1:
                self._errors.append(
                    f"generation.top_p must be in [0, 1], got {gen['top_p']}"
                )

        if isinstance(gen.get("max_tokens"), int):
            if gen["max_tokens"] <= 0:
                self._errors.append(
                    f"generation.max_tokens must be > 0, got {gen['max_tokens']}"
                )

        if isinstance(gen.get("repetition_penalty"), (int, float)):
            if gen["repetition_penalty"] <= 0:
                self._errors.append(
                    f"generation.repetition_penalty must be > 0, got {gen['repetition_penalty']}"
                )

        # Tokenizer constraints
        if isinstance(tok.get("model"), str):
            if tok["model"] not in VALID_TOKENIZER_MODELS:
                self._errors.append(
                    f"tokenizer.model must be one of {VALID_TOKENIZER_MODELS}, "
                    f"got '{tok['model']}'"
                )

        if isinstance(tok.get("context_window"), int):
            if tok["context_window"] <= 0:
                self._errors.append(
                    f"tokenizer.context_window must be > 0, got {tok['context_window']}"
                )

        if isinstance(tok.get("truncation_strategy"), str):
            if tok["truncation_strategy"] not in VALID_TRUNCATION_STRATEGIES:
                self._errors.append(
                    f"tokenizer.truncation_strategy must be one of "
                    f"{VALID_TRUNCATION_STRATEGIES}, got '{tok['truncation_strategy']}'"
                )

        # Rate limiting constraints
        if isinstance(rate.get("max_retries"), int):
            if rate["max_retries"] < 0:
                self._errors.append(
                    f"rate_limiting.max_retries must be >= 0, got {rate['max_retries']}"
                )

        if isinstance(rate.get("backoff_base_seconds"), (int, float)):
            if rate["backoff_base_seconds"] <= 0:
                self._errors.append(
                    f"rate_limiting.backoff_base_seconds must be > 0, "
                    f"got {rate['backoff_base_seconds']}"
                )

        if isinstance(rate.get("requests_per_minute"), int):
            if rate["requests_per_minute"] <= 0:
                self._errors.append(
                    f"rate_limiting.requests_per_minute must be > 0, "
                    f"got {rate['requests_per_minute']}"
                )

        # Cost constraints
        for field in ["per_million_input_tokens", "per_million_output_tokens"]:
            if isinstance(cost.get(field), (int, float)):
                if cost[field] < 0:
                    self._errors.append(
                        f"cost.{field} must be >= 0, got {cost[field]}"
                    )

        # API constraints
        if isinstance(api.get("request_timeout_seconds"), (int, float)):
            if api["request_timeout_seconds"] <= 0:
                self._errors.append(
                    f"api.request_timeout_seconds must be > 0, "
                    f"got {api['request_timeout_seconds']}"
                )

        # Logging level
        if isinstance(log.get("level"), str):
            if log["level"].upper() not in VALID_LOG_LEVELS:
                self._errors.append(
                    f"logging.level must be one of {VALID_LOG_LEVELS}, "
                    f"got '{log['level']}'"
                )

    def _validate_env_var(self, dry_run_override: Optional[bool] = None) -> None:
        """Check that the API key environment variable is set."""
        is_dry_run = dry_run_override if dry_run_override is not None else self._config.get("dry_run", False)
        env_var = self._config.get("api", {}).get("api_key_env_var", "")

        if not env_var:
            self._errors.append("api.api_key_env_var must be a non-empty string")
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
