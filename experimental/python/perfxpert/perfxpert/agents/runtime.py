"""Session factory + provider wiring.

Single entry point shared by: batch CLI, library API, MCP server.

Responsibilities:
- Pick provider from arg / env / config
- Validate against PROVIDER_REGISTRY (providers/__init__.py)
- Call runtime.recursion_guard.ensure_not_recursive(provider)
- Honor PERFXPERT_AIRGAP env var (short-circuits ALL provider resolution)
- Honor PERFXPERT_LLM_FALLBACK_CHAIN env var (cascade across providers on
  RateLimitError; ignored in airgap mode)
- Generate session_id if missing
- Expose run_root / run_correctness / run_analysis / run_recommendation
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from perfxpert.agents import analysis, correctness, recommendation, root, schemas
from perfxpert.runtime import ensure_not_recursive

_LOG = logging.getLogger("perfxpert.agents.runtime")


# Defensive import — fallback registry keeps this module importable when
# the providers package is not yet loaded in isolated test contexts.
#
# Phase 8 fix: ``claude-code`` is advertised by the CLI as a valid
# ``--llm`` choice (analyze.py) but was missing here, so
# `perfxpert analyze --llm claude-code` raised ``ValueError: unknown
# provider``. It is a credential alias that routes through Anthropic —
# see framework._api_key_for() for the auth fallback chain.
try:
    from perfxpert.providers import PROVIDER_REGISTRY  # type: ignore
except ImportError:
    PROVIDER_REGISTRY = {
        "anthropic": "Claude API",
        "openai": "OpenAI GPT",
        "ollama": "Local Ollama",
        "private": "Custom OpenAI-compatible endpoint",
        "opencode": "Bundled opencode CLI",
        "claude-code": "Anthropic Claude via claude-code CLI credentials",
    }


DEFAULT_PROVIDER = "anthropic"


# Cycle-4 B3 — substrings that identify a rate-limit-class error raised
# by the underlying SDK (OpenAI Agents SDK wraps 429 as a bare Python
# exception; the message contains these tokens). Conservative list —
# only patterns we *know* are rate-limit, not generic network errors.
_RATE_LIMIT_TOKENS = (
    "429",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "insufficient_quota",
    "quota exceeded",
    "too many requests",
)


def _is_rate_limit_like(exc: BaseException) -> bool:
    """Walk the __cause__/__context__ chain; True iff any link is rate-limit.

    We match on exception class name AND message text — the OpenAI SDK
    class is `openai.RateLimitError` and the Anthropic SDK class is
    `anthropic.RateLimitError`, so the class-name check alone catches the
    common cases; the text check is a belt-and-suspenders fallback for
    providers that wrap errors in ``ProviderError(msg)``.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        class_name = type(current).__name__.lower()
        if "ratelimit" in class_name:
            return True
        msg = str(current).lower()
        if any(tok in msg for tok in _RATE_LIMIT_TOKENS):
            return True
        # Walk both the __cause__ (from ... raise) and __context__ (implicit).
        current = current.__cause__ or current.__context__
    return False


@dataclass(frozen=True)
class AnalysisSession:
    """Handle returned by build_session() — expose run_* wrappers.

    ``provider`` remains the canonical string name of the primary provider
    (used by run_* dispatchers). ``fallback_provider`` is an optional
    FallbackProvider instance, populated when PERFXPERT_LLM_FALLBACK_CHAIN
    is set. It is always ``None`` in airgap mode.

    Cycle-4 B3 — the ``run_*`` methods now actually USE the fallback chain
    on the hot path: if the primary provider raises anything rate-limit-y
    (OpenAI ``RateLimitError``, Anthropic ``RateLimitError``, or any
    exception whose message mentions 429 / insufficient_quota / rate
    limit), we iterate through ``_fallback_chain()`` and retry with the
    next provider. Every attempt is logged; if the whole chain is
    exhausted we raise ``ProviderChainExhausted`` so the caller sees a
    single typed error rather than a bare 429.
    """
    session_id: str
    provider: Optional[str]
    airgap: bool
    fallback_provider: Optional[Any] = field(default=None)

    def _fallback_chain(self) -> List[str]:
        """Return the ordered provider chain for this session.

        Primary provider is position 0; the fallback entries follow in
        the order the user configured them. Duplicates are removed
        preserving first occurrence. Returns a single-element list when
        no fallback_provider is configured.
        """
        primary = self.provider or DEFAULT_PROVIDER
        if self.fallback_provider is None:
            return [primary]
        # FallbackProvider stores its raw entries on ._providers; when
        # constructed via build_session they are always strings. But we
        # defensively coerce to handle the pre-instantiated-Provider
        # branch too.
        raw = getattr(self.fallback_provider, "_providers", [])
        chain: List[str] = [primary]
        for entry in raw:
            name = entry if isinstance(entry, str) else type(entry).__name__.lower()
            if name and name not in chain:
                chain.append(name)
        return chain

    def _cascade(
        self,
        runner: Callable[[str], Any],
        op_name: str,
    ) -> Any:
        """Execute ``runner(provider_name)`` across the fallback chain.

        First attempt uses the primary provider. On a rate-limit-like
        exception (detected by :func:`_is_rate_limit_like`), we log and
        proceed to the next provider. Every other exception propagates
        immediately (auth / validation errors are not cascade-worthy).
        Returns the first successful result; raises
        ``ProviderChainExhausted`` when no provider succeeds.
        """
        chain = self._fallback_chain()
        if len(chain) == 1:
            # No fallback configured — preserve the legacy single-provider
            # behaviour (and exception shape) exactly.
            return runner(chain[0])

        # Lazy import to avoid pulling providers package in airgap unit tests.
        from perfxpert.providers._exceptions import ProviderChainExhausted

        attempts: List[Tuple[str, BaseException]] = []
        for idx, name in enumerate(chain):
            try:
                return runner(name)
            except Exception as exc:
                if not _is_rate_limit_like(exc):
                    # Auth / schema / timeout / anything else: surface
                    # immediately. Cascading across non-rate-limit errors
                    # would mask bugs.
                    raise
                _LOG.warning(
                    "fallback: %s on provider %d/%d (%r) hit rate-limit; trying next",
                    op_name,
                    idx + 1,
                    len(chain),
                    name,
                )
                attempts.append((name, exc))
        # Everyone rate-limited — surface a typed chain-exhausted error.
        err = ProviderChainExhausted(providers=chain, attempts=attempts)
        if attempts:
            raise err from attempts[-1][1]
        raise err

    def run_root(self, payload: schemas.RootInput) -> schemas.RootOutput:
        if self.airgap:
            return root.run_root(payload, airgap=True)
        return self._cascade(
            lambda prov: root.run_root(payload, provider=prov),
            op_name="run_root",
        )

    def run_analysis(self, payload: schemas.AnalysisInput) -> schemas.AnalysisOutput:
        if self.airgap:
            return analysis.run_analysis(payload, airgap=True)
        return self._cascade(
            lambda prov: analysis.run_analysis(payload, provider=prov),
            op_name="run_analysis",
        )

    def run_recommendation(
        self, payload: schemas.RecommendationInput
    ) -> schemas.RecommendationOutput:
        if self.airgap:
            return recommendation.run_recommendation(payload, airgap=True)
        return self._cascade(
            lambda prov: recommendation.run_recommendation(payload, provider=prov),
            op_name="run_recommendation",
        )

    def run_correctness(
        self, payload: schemas.CorrectnessInput
    ) -> schemas.CorrectnessOutput:
        if self.airgap:
            return correctness.run_correctness(payload, airgap=True)
        return self._cascade(
            lambda prov: correctness.run_correctness(payload, provider=prov),
            op_name="run_correctness",
        )


def _airgap_from_env() -> bool:
    return os.environ.get("PERFXPERT_AIRGAP", "0") == "1"


def build_session(
    *,
    provider: Optional[str] = None,
    session_id: Optional[str] = None,
    airgap: Optional[bool] = None,
) -> AnalysisSession:
    """Build an AnalysisSession handle.

    Provider resolution rules (cycle-2 I8):

    1. ``airgap=True`` (or ``PERFXPERT_AIRGAP=1``): skip ALL provider
       resolution, ``fallback_provider`` is always ``None``. The
       FallbackProvider chain is deliberately ignored — a session
       configured for airgap must never make network calls.
    2. ``airgap=False`` and ``PERFXPERT_LLM_FALLBACK_CHAIN`` is set: build
       a :class:`FallbackProvider` from the env chain. If the ``provider``
       argument is also set, it is prepended as the primary entry (so the
       explicit choice goes first).
    3. ``airgap=False`` and no chain env var: legacy single-provider path
       via ``get_provider(provider)``.

    Args:
        provider: LLM provider name (anthropic/openai/ollama/private/opencode).
                  Ignored when airgap=True.
        session_id: Explicit id for the task store; uuid4() if None.
        airgap: If True (or PERFXPERT_AIRGAP=1), skip LLM entirely.

    Raises:
        ValueError: unknown provider.
        RecursionGuardViolation: provider='opencode' inside an opencode session.
    """
    is_airgap = airgap if airgap is not None else _airgap_from_env()

    fallback: Optional[Any] = None

    if is_airgap:
        # Airgap is the ONLY deterministic-only path. Never build a chain.
        prov = None
    else:
        prov = provider or DEFAULT_PROVIDER
        if prov not in PROVIDER_REGISTRY:
            valid = ", ".join(PROVIDER_REGISTRY.keys())
            raise ValueError(f"unknown provider {prov!r}; valid: {valid}")
        ensure_not_recursive(prov)

        # Opt-in fallback chain. Import lazily so tests that stub
        # PROVIDER_REGISTRY in isolation still work.
        from perfxpert.providers._fallback import (
            ENV_FALLBACK_CHAIN,
            FallbackProvider,
            parse_chain_env,
        )

        chain = parse_chain_env(os.environ.get(ENV_FALLBACK_CHAIN))
        if chain:
            # Rule 4 (from brief): primary provider arg goes first in the
            # chain. De-dup so we never attempt the same provider twice in
            # a row.
            if provider and provider not in chain:
                chain = [provider] + list(chain)
            fallback = FallbackProvider(list(chain))

    return AnalysisSession(
        session_id=session_id or str(uuid.uuid4()),
        provider=prov,
        airgap=is_airgap,
        fallback_provider=fallback,
    )


__all__ = ["AnalysisSession", "build_session", "PROVIDER_REGISTRY", "DEFAULT_PROVIDER"]
