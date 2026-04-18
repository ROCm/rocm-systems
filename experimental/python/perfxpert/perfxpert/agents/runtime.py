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

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from perfxpert.agents import analysis, correctness, recommendation, root, schemas
from perfxpert.runtime import ensure_not_recursive


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


@dataclass(frozen=True)
class AnalysisSession:
    """Handle returned by build_session() — expose run_* wrappers.

    ``provider`` remains the canonical string name of the primary provider
    (used by run_* dispatchers). ``fallback_provider`` is an optional
    FallbackProvider instance, populated when PERFXPERT_LLM_FALLBACK_CHAIN
    is set. It is always ``None`` in airgap mode.
    """
    session_id: str
    provider: Optional[str]
    airgap: bool
    fallback_provider: Optional[Any] = field(default=None)

    def run_root(self, payload: schemas.RootInput) -> schemas.RootOutput:
        if self.airgap:
            return root.run_root(payload, airgap=True)
        return root.run_root(payload, provider=self.provider or DEFAULT_PROVIDER)

    def run_analysis(self, payload: schemas.AnalysisInput) -> schemas.AnalysisOutput:
        if self.airgap:
            return analysis.run_analysis(payload, airgap=True)
        return analysis.run_analysis(payload, provider=self.provider or DEFAULT_PROVIDER)

    def run_recommendation(
        self, payload: schemas.RecommendationInput
    ) -> schemas.RecommendationOutput:
        if self.airgap:
            return recommendation.run_recommendation(payload, airgap=True)
        return recommendation.run_recommendation(
            payload, provider=self.provider or DEFAULT_PROVIDER
        )

    def run_correctness(
        self, payload: schemas.CorrectnessInput
    ) -> schemas.CorrectnessOutput:
        if self.airgap:
            return correctness.run_correctness(payload, airgap=True)
        return correctness.run_correctness(
            payload, provider=self.provider or DEFAULT_PROVIDER
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
