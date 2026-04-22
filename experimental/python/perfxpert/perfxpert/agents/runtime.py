"""Session factory + provider wiring.

Single entry point shared by: batch CLI, library API, MCP server.

Responsibilities:
- Pick provider from arg / env / config
- Validate against PROVIDER_REGISTRY (Phase 2 providers/__init__.py)
- Call runtime.recursion_guard.ensure_not_recursive(provider)
- Honor PERFXPERT_AIRGAP env var
- Generate session_id if missing
- Expose run_root / run_correctness / run_analysis / run_recommendation
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Optional

from perfxpert.agents import analysis, correctness, recommendation, root, schemas
from perfxpert.runtime import ensure_not_recursive


# Phase 2 exposes this — defensive import with fallback for Phase 3 isolation.
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
    """Handle returned by build_session() — expose run_* wrappers."""
    session_id: str
    provider: Optional[str]
    airgap: bool

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

    Args:
        provider: LLM provider name (anthropic/openai/ollama/private/opencode).
                  Ignored when airgap=True.
        session_id: Explicit session identifier; uuid4() if None.
        airgap: If True (or PERFXPERT_AIRGAP=1), skip LLM entirely.

    Raises:
        ValueError: unknown provider.
        RecursionGuardViolation: provider='opencode' inside an opencode session.
    """
    is_airgap = airgap if airgap is not None else _airgap_from_env()

    if not is_airgap:
        prov = provider or DEFAULT_PROVIDER
        if prov not in PROVIDER_REGISTRY:
            valid = ", ".join(PROVIDER_REGISTRY.keys())
            raise ValueError(f"unknown provider {prov!r}; valid: {valid}")
        ensure_not_recursive(prov)
    else:
        prov = None

    return AnalysisSession(
        session_id=session_id or str(uuid.uuid4()),
        provider=prov,
        airgap=is_airgap,
    )


__all__ = ["AnalysisSession", "build_session", "PROVIDER_REGISTRY", "DEFAULT_PROVIDER"]
