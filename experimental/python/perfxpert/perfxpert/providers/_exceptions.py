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
    """Provider returned 429 / explicit rate-limit.

    Distinct from :class:`QuotaExceededError`: a rate-limit is a short-term
    throttle (retry after a backoff) while quota-exhausted means the
    account has run out of credit (retry won't help — top up, switch
    provider, or use airgap).
    """

    def __init__(self, provider: str, retry_after: float = 0.0, message: str = "") -> None:
        self.provider = provider
        self.retry_after = retry_after
        detail = f": {message}" if message else ""
        super().__init__(
            f"[{provider}] rate limited (retry_after={retry_after}s){detail}"
        )


class QuotaExceededError(ProviderError):
    """Provider returned 429 ``insufficient_quota`` / ``quota_exceeded``.

    Distinct from a transient rate-limit: retry won't help. Surface as a
    one-line user-actionable message (``top up the account, switch
    provider, or use PERFXPERT_AIRGAP=1``).
    """

    def __init__(self, provider: str, model: str = "", message: str = "") -> None:
        self.provider = provider
        self.model = model
        self.raw_message = message
        detail = f": {message}" if message else ""
        suffix = f" (model={model})" if model else ""
        super().__init__(f"[{provider}] quota exhausted{suffix}{detail}")


class TransientError(ProviderError):
    """Provider returned a transient error (5xx, connection reset, timeout).

    Unlike :class:`RateLimitError` this covers server-side issues that are
    not rate-related: connection drops, 5xx responses, read timeouts.
    Callers may retry or cascade.
    """

    def __init__(self, provider: str, kind: str = "", message: str = "") -> None:
        self.provider = provider
        self.kind = kind
        self.raw_message = message
        detail = f": {message}" if message else ""
        kind_suffix = f" ({kind})" if kind else ""
        super().__init__(f"[{provider}] transient error{kind_suffix}{detail}")


class FatalError(ProviderError):
    """Any provider failure not classified as auth/rate/quota/transient.

    Surfaced as a one-line user-facing message. The raw backend message is
    retained on ``.raw_message`` so operators can diagnose; set
    ``PERFXPERT_DEBUG=1`` to see the full traceback.
    """

    def __init__(self, provider: str, message: str = "") -> None:
        self.provider = provider
        self.raw_message = message
        detail = f": {message}" if message else ""
        super().__init__(f"[{provider}] fatal error{detail}")


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
    "QuotaExceededError",
    "TransientError",
    "FatalError",
    "TimeoutError",
    "UnknownProvider",
    "ProviderChainExhausted",
    "DryRunResponse",
    "classify_sdk_error",
]

def classify_sdk_error(
    exc: BaseException,
    *,
    provider: str,
    model: str = "",
) -> ProviderError:
    """Map an arbitrary backend SDK exception into the perfxpert taxonomy.

    Detects the error kind by walking the ``__cause__``/``__context__``
    chain and inspecting both the exception class names and the message
    text. Returns a new :class:`ProviderError` subclass instance — the
    caller should ``raise`` it (optionally chaining via ``from exc``).

    Classification priority:
      1. Quota — ``insufficient_quota`` / ``quota_exceeded`` / ``quota exceeded``
         (even when paired with a 429 — quota takes precedence over
         generic rate-limit because retry won't help).
      2. Rate-limit — HTTP 429 or SDK ``RateLimitError`` class without
         the quota substring.
      3. Auth — HTTP 401/403 / ``AuthenticationError`` / ``PermissionDeniedError``
         / ``invalid_api_key``.
      4. Transient — 5xx / connection errors / timeouts / ``InternalServerError``.
      5. Fatal — anything else.
    """
    seen: set = set()
    current = exc
    chain_text_parts = []
    class_names = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        class_names.append(type(current).__name__.lower())
        chain_text_parts.append(str(current))
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    full_text = " | ".join(chain_text_parts).lower()
    class_blob = " ".join(class_names)

    def _contains(*tokens: str) -> bool:
        return any(tok in full_text for tok in tokens)

    raw_msg = str(exc)
    # Trim the raw message — we only want the first line for user display;
    # the full traceback is gated behind PERFXPERT_DEBUG=1 at the boundary.
    short_msg = raw_msg.strip().splitlines()[0] if raw_msg.strip() else ""
    # Cap to a sensible display length.
    if len(short_msg) > 240:
        short_msg = short_msg[:237] + "..."

    # 1. Quota first — beats generic 429 because retry is futile.
    if _contains("insufficient_quota", "quota_exceeded", "quota exceeded", "exceeded your current quota"):
        return QuotaExceededError(provider=provider, model=model, message=short_msg)

    # 2. Rate-limit — 429 without quota.
    if "ratelimit" in class_blob or _contains("429", "rate limit", "rate_limit", "too many requests"):
        return RateLimitError(provider=provider, message=short_msg)

    # 3. Auth failures.
    if (
        "authentication" in class_blob
        or "permissiondenied" in class_blob
        or _contains("invalid_api_key", "authentication_error", "authentication failed", "unauthorized", "401", "403", "permission_denied")
    ):
        return AuthError(provider=provider, message=short_msg)

    # 4. Transient — 5xx / timeout / connection issues.
    if (
        "timeout" in class_blob
        or "timeouterror" in class_blob
        or "connectionerror" in class_blob
        or "internalserver" in class_blob
        or "apiconnection" in class_blob
        or _contains("timeout", "timed out", "connection reset", "connection refused", "500", "502", "503", "504", "bad gateway", "service unavailable")
    ):
        kind = "timeout" if _contains("timeout", "timed out") else "server"
        return TransientError(provider=provider, kind=kind, message=short_msg)

    # 5. Default: fatal with the raw message preserved.
    return FatalError(provider=provider, message=short_msg)


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
