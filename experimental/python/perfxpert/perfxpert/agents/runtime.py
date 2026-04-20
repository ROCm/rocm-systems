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
  plus run_compute_specialist / run_memory_specialist / run_latency_specialist
  (one session method per agent in the hierarchy — used by the MCP
  tool wrappers under perfxpert.tools.agents.*)
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from perfxpert.agents import (
    analysis,
    compute_specialist,
    correctness,
    latency_specialist,
    memory_specialist,
    recommendation,
    root,
    schemas,
)
from perfxpert.runtime import ensure_not_recursive

_LOG = logging.getLogger("perfxpert.agents.runtime")


# Defensive import — fallback registry keeps this module importable when
# the providers package is not yet loaded in isolated test contexts.
try:
    from perfxpert.providers import PROVIDER_REGISTRY  # type: ignore
except ImportError:
    PROVIDER_REGISTRY = {
        "anthropic": "Claude API",
        "openai": "OpenAI GPT",
        "ollama": "Local Ollama",
        "private": "Custom OpenAI-compatible endpoint",
        "opencode": "Bundled opencode CLI",
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
    """Walk the __cause__/__context__ chain; True iff any link is rate-limit
    or transient (but NOT quota/auth/fatal).

    The cascade ONLY fires for errors where trying the next provider
    could plausibly succeed — rate-limit and server-transient failures.
    Quota-exhausted errors mean the account has no credit (next provider
    might work, but we still want the user to know the first provider's
    quota ran out so they can top up); auth errors mean the key is bad
    and cascading hides the real fix; fatal errors need to surface.

    Cycle-2 I8: only transient + rate-limit cascade; auth/fatal surface
    immediately (per memory index).
    """
    # New taxonomy types take precedence over substring matching.
    from perfxpert.providers._exceptions import (
        AuthError,
        FatalError,
        QuotaExceededError,
        RateLimitError,
        TransientError,
    )

    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        # Explicit typed matches first — unambiguous classification.
        if isinstance(current, (QuotaExceededError, AuthError, FatalError)):
            return False
        if isinstance(current, (RateLimitError, TransientError)):
            return True
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

    ``progress_callback`` is an optional ``Callable[[str], None]`` that
    receives short human-readable status strings as the session
    progresses through each agent phase (``"entering <phase>"`` /
    ``"exit <phase>"`` and ``"provider <prov> failed with rate-limit,
    trying next"`` for fallback cascades). The CLI wires this to a Rich
    spinner so users can see progress during long LLM calls; MCP
    consumers generally leave it None and pay zero overhead. A per-call
    callback argument on the ``run_*`` methods always overrides the
    session-level default.

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
    progress_callback: Optional[Callable[[str], None]] = field(default=None)

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

    def _emit(
        self,
        callback: Optional[Callable[[str], None]],
        msg: str,
    ) -> None:
        """Fire a progress callback, swallowing any exception it raises.

        A buggy UI callback must never break the analysis pipeline. The
        per-call ``callback`` argument always wins over ``self.progress_callback``;
        both may be None.
        """
        cb = callback if callback is not None else self.progress_callback
        if cb is None:
            return
        try:
            cb(msg)
        except Exception:  # pragma: no cover — defensive; log and move on
            _LOG.debug("progress_callback raised; ignoring", exc_info=True)

    def _cascade(
        self,
        runner: Callable[[str], Any],
        op_name: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Any:
        """Execute ``runner(provider_name)`` across the fallback chain.

        First attempt uses the primary provider. On a rate-limit-like
        exception (detected by :func:`_is_rate_limit_like`), we log and
        proceed to the next provider. Every other exception propagates
        immediately (auth / validation errors are not cascade-worthy).
        Returns the first successful result; raises
        ``ProviderChainExhausted`` when no provider succeeds.

        When a ``progress_callback`` is supplied (per-call or via the
        session) it is fired once per provider transition with a short
        ``"provider <name> failed with rate-limit/transient, trying next"``
        message so the CLI spinner can surface the cascade to the user.
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
                self._emit(
                    progress_callback,
                    f"provider {name} failed with rate-limit/transient, trying next",
                )
                attempts.append((name, exc))
        # Everyone rate-limited — surface a typed chain-exhausted error.
        err = ProviderChainExhausted(providers=chain, attempts=attempts)
        if attempts:
            raise err from attempts[-1][1]
        raise err

    def run_root(
        self,
        payload: schemas.RootInput,
        *,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.RootOutput:
        self._emit(progress_callback, "entering root")
        try:
            if self.airgap:
                return root.run_root(payload, airgap=True)
            return self._cascade(
                lambda prov: root.run_root(payload, provider=prov),
                op_name="run_root",
                progress_callback=progress_callback,
            )
        finally:
            self._emit(progress_callback, "exit root")

    def run_analysis(
        self,
        payload: schemas.AnalysisInput,
        *,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.AnalysisOutput:
        self._emit(progress_callback, "entering analysis")
        try:
            if self.airgap:
                return analysis.run_analysis(payload, airgap=True)
            return self._cascade(
                lambda prov: analysis.run_analysis(payload, provider=prov),
                op_name="run_analysis",
                progress_callback=progress_callback,
            )
        finally:
            self._emit(progress_callback, "exit analysis")

    def run_recommendation(
        self,
        payload: schemas.RecommendationInput,
        *,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.RecommendationOutput:
        self._emit(progress_callback, "entering recommendation")
        try:
            if self.airgap:
                return recommendation.run_recommendation(payload, airgap=True)
            return self._cascade(
                lambda prov: recommendation.run_recommendation(payload, provider=prov),
                op_name="run_recommendation",
                progress_callback=progress_callback,
            )
        finally:
            self._emit(progress_callback, "exit recommendation")

    def run_correctness(
        self,
        payload: schemas.CorrectnessInput,
        *,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.CorrectnessOutput:
        self._emit(progress_callback, "entering correctness")
        try:
            if self.airgap:
                return correctness.run_correctness(payload, airgap=True)
            return self._cascade(
                lambda prov: correctness.run_correctness(payload, provider=prov),
                op_name="run_correctness",
                progress_callback=progress_callback,
            )
        finally:
            self._emit(progress_callback, "exit correctness")

    def run_compute_specialist(
        self,
        payload: schemas.ComputeSpecialistInput,
        *,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.ComputeSpecialistOutput:
        """Compute-Techniques specialist (Layer 2) via the session cascade."""
        self._emit(progress_callback, "entering compute_specialist")
        try:
            if self.airgap:
                return compute_specialist.run_compute_specialist(payload, airgap=True)
            return self._cascade(
                lambda prov: compute_specialist.run_compute_specialist(
                    payload, provider=prov
                ),
                op_name="run_compute_specialist",
                progress_callback=progress_callback,
            )
        finally:
            self._emit(progress_callback, "exit compute_specialist")

    def run_memory_specialist(
        self,
        payload: schemas.MemorySpecialistInput,
        *,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.MemorySpecialistOutput:
        """Memory-Techniques specialist (Layer 2) via the session cascade."""
        self._emit(progress_callback, "entering memory_specialist")
        try:
            if self.airgap:
                return memory_specialist.run_memory_specialist(payload, airgap=True)
            return self._cascade(
                lambda prov: memory_specialist.run_memory_specialist(
                    payload, provider=prov
                ),
                op_name="run_memory_specialist",
                progress_callback=progress_callback,
            )
        finally:
            self._emit(progress_callback, "exit memory_specialist")

    def run_latency_specialist(
        self,
        payload: schemas.LatencySpecialistInput,
        *,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.LatencySpecialistOutput:
        """Latency-Techniques specialist (Layer 2) via the session cascade."""
        self._emit(progress_callback, "entering latency_specialist")
        try:
            if self.airgap:
                return latency_specialist.run_latency_specialist(payload, airgap=True)
            return self._cascade(
                lambda prov: latency_specialist.run_latency_specialist(
                    payload, provider=prov
                ),
                op_name="run_latency_specialist",
                progress_callback=progress_callback,
            )
        finally:
            self._emit(progress_callback, "exit latency_specialist")


def _airgap_from_env() -> bool:
    return os.environ.get("PERFXPERT_AIRGAP", "0") == "1"


def build_session(
    *,
    provider: Optional[str] = None,
    session_id: Optional[str] = None,
    airgap: Optional[bool] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
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
        progress_callback: Optional ``Callable[[str], None]`` that receives
            short status strings as each agent phase enters / exits. Used
            by the CLI to drive a Rich spinner during long LLM calls. A
            ``None`` value disables the feature with zero overhead.

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
        progress_callback=progress_callback,
    )


__all__ = ["AnalysisSession", "build_session", "PROVIDER_REGISTRY", "DEFAULT_PROVIDER"]
