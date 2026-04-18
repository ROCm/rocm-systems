"""Provider error taxonomy + DryRunResponse singleton.

Every provider maps its backend-specific exception types into this small
unified taxonomy so callers do not branch on SDK-specific classes.

DryRunResponse is a module-level singleton returned whenever `dry_run=True`
is passed to Provider.complete(); it guarantees zero network I/O and is
used by air-gap mode + cost estimation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


class ProviderError(Exception):
    """Base class for all provider failures."""


class AuthError(ProviderError):
    """Authentication failed — bad API key, missing env var, expired token."""

    def __init__(self, provider: str, message: str = "") -> None:
        self.provider = provider
        detail = f": {message}" if message else ""
        super().__init__(f"[{provider}] auth failed{detail}")


class RateLimitError(ProviderError):
    """Provider returned 429 / explicit rate-limit."""

    def __init__(self, provider: str, retry_after: float = 0.0, message: str = "") -> None:
        self.provider = provider
        self.retry_after = retry_after
        detail = f": {message}" if message else ""
        super().__init__(
            f"[{provider}] rate limited (retry_after={retry_after}s){detail}"
        )


class TimeoutError(ProviderError):
    """Request exceeded the per-call timeout budget."""

    def __init__(self, provider: str, timeout_seconds: float = 0.0, message: str = "") -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        detail = f": {message}" if message else ""
        super().__init__(
            f"[{provider}] timed out after {timeout_seconds}s{detail}"
        )


class UnknownProvider(ProviderError):
    """Requested provider name is not in the registry.

    Raised (instead of a bare ``KeyError`` from ``registry.get_provider``)
    so callers never have to branch on non-taxonomy exceptions.
    """

    def __init__(self, name: str, known: Optional[List[str]] = None) -> None:
        self.name = name
        self.known = list(known) if known else []
        known_str = ", ".join(self.known) if self.known else "<none>"
        super().__init__(f"unknown provider {name!r}; known: {known_str}")


class ProviderChainExhausted(ProviderError):
    """All providers in a FallbackProvider chain failed.

    Raised by ``FallbackProvider.complete()`` when every entry in the
    chain either could not be resolved (``UnknownProvider``) or raised a
    cascade-worthy error (``RateLimitError``). The per-entry history is
    preserved on ``.attempts`` and the last exception is chained via
    ``__cause__`` for full tracebacks.
    """

    def __init__(
        self,
        providers: Sequence[str],
        attempts: Sequence[Tuple[str, BaseException]],
    ) -> None:
        self.providers: List[str] = list(providers)
        self.attempts: List[Tuple[str, BaseException]] = list(attempts)
        summary = "; ".join(
            f"{name}: {type(exc).__name__}: {exc}" for name, exc in self.attempts
        ) or "<no attempts recorded>"
        super().__init__(
            f"FallbackProvider chain exhausted ({len(self.providers)} entries): {summary}"
        )


class _DryRunResponseType:
    """Sentinel response returned when Provider.complete(dry_run=True)."""

    __slots__ = ()

    content = ""
    provider = "dry_run"
    model = "dry_run"
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "<DryRunResponse>"


# Module-level singleton — import this, do not instantiate _DryRunResponseType.
DryRunResponse = _DryRunResponseType()


__all__ = [
    "ProviderError",
    "AuthError",
    "RateLimitError",
    "TimeoutError",
    "UnknownProvider",
    "ProviderChainExhausted",
    "DryRunResponse",
]

import warnings as _warnings


def _legacy_env_warn(legacy_name: str, canonical_name: str) -> None:
    """Emit a DeprecationWarning when a pre-rename env var alias is used.

    Providers call this when they fall through to a pre-rename alias so
    users get a clear migration signal toward the canonical
    PERFXPERT_LLM_* name.
    """
    _warnings.warn(
        (
            f"Environment variable {legacy_name!r} is deprecated; "
            f"rename to {canonical_name!r}. "
            f"Legacy name will be removed in a future perfxpert release."
        ),
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = list(__all__) + ["_legacy_env_warn"]  # type: ignore[misc]
