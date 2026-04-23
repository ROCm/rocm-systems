"""Provider error taxonomy + DryRunResponse singleton.

Every provider maps its backend-specific exception types into this small
unified taxonomy so callers do not branch on SDK-specific classes.

DryRunResponse is a module-level singleton returned whenever `dry_run=True`
is passed to Provider.complete(); it guarantees zero network I/O and is
used by air-gap mode + cost estimation.
"""

from __future__ import annotations


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
    "DryRunResponse",
]

import warnings as _warnings


def _legacy_env_warn(legacy_name: str, canonical_name: str) -> None:
    """Emit a DeprecationWarning when a legacy env var is still used.

    Called by providers when they fall through to ROCINSIGHT_LLM_* or
    ROCPD_LLM_* variables so users get a clear migration signal.
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
