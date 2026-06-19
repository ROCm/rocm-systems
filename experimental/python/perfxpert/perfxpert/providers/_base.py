"""Provider base class + ProviderResponse dataclass.

Every concrete provider (anthropic, openai, ollama, private, opencode)
inherits Provider and implements complete() with identical signature.

ProviderResponse is frozen — callers can safely cache / hash it without
worrying about mutation. total_tokens is a derived property, not a stored
field, so providers only set input_tokens + output_tokens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from perfxpert.providers._exceptions import DryRunResponse as _DryRunResponseT


@dataclass(frozen=True)
class ProviderResponse:
    """Uniform LLM response across all providers."""

    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Sum of input + output tokens (derived, not stored)."""
        return self.input_tokens + self.output_tokens


class Provider(ABC):
    """Abstract base for all LLM providers.

    Subclasses must implement complete(). dry_run=True MUST return the
    DryRunResponse singleton without any network I/O.
    """

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        dry_run: bool = False,
    ) -> Union[ProviderResponse, _DryRunResponseT]:
        """Run a chat completion.

        Args:
            messages: OpenAI-style messages list [{"role": "...", "content": "..."}].
            system: Optional system prompt (provider-specific embedding).
            model: Provider-specific model ID (None = provider default).
            max_tokens: Generation cap (None = provider default).
            dry_run: If True, return DryRunResponse without network I/O.

        Returns:
            ProviderResponse on success, DryRunResponse if dry_run.

        Raises:
            AuthError, RateLimitError, QuotaExceededError, TransientError,
            FatalError, TimeoutError, ProviderChainExhausted, ProviderError.
        """
        raise NotImplementedError


__all__ = ["Provider", "ProviderResponse"]
