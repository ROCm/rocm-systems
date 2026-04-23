"""Session factory + provider wiring.

Single entry point shared by: batch CLI, library API, MCP server.

Responsibilities:
- Pick provider from arg / env / config
- Validate against PROVIDER_REGISTRY (providers/__init__.py)
- Call runtime.recursion_guard.ensure_not_recursive(provider)
- Honor PERFXPERT_AIRGAP env var
- Generate session_id if missing
- Expose run_root / run_correctness / run_analysis / run_recommendation
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

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
    """Handle returned by build_session() — expose run_* wrappers."""
    session_id: str
    provider: Optional[str]
    airgap: bool

    def run_root(
        self,
        payload: schemas.RootInput,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.RootOutput:
        # ``progress_callback`` is accepted for compatibility with the
        # newer agent-tool wrappers restored during PR cleanup. This
        # rebased runtime does not stream progress yet, so the callback
        # is intentionally unused here.
        del progress_callback
        if self.airgap:
            return root.run_root(payload, airgap=True)
        return root.run_root(payload, provider=self.provider or DEFAULT_PROVIDER)

    def run_analysis(
        self,
        payload: schemas.AnalysisInput,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.AnalysisOutput:
        del progress_callback
        if self.airgap:
            return analysis.run_analysis(payload, airgap=True)
        return analysis.run_analysis(payload, provider=self.provider or DEFAULT_PROVIDER)

    def run_recommendation(
        self,
        payload: schemas.RecommendationInput,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.RecommendationOutput:
        del progress_callback
        if self.airgap:
            return recommendation.run_recommendation(payload, airgap=True)
        return recommendation.run_recommendation(
            payload, provider=self.provider or DEFAULT_PROVIDER
        )

    def run_correctness(
        self,
        payload: schemas.CorrectnessInput,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.CorrectnessOutput:
        del progress_callback
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
    progress_callback: Optional[Callable[[str], None]] = None,
    api_key: Optional[str] = None,
) -> AnalysisSession:
    """Build an AnalysisSession handle.

    Args:
        provider: LLM provider name (anthropic/openai/ollama/private/opencode).
                  Ignored when airgap=True.
        session_id: Explicit id for the task store; uuid4() if None.
        airgap: If True (or PERFXPERT_AIRGAP=1), skip LLM entirely.
        progress_callback: Accepted for API compatibility with newer
                           stacked-PR wrappers. Unused in this runtime.
        api_key: Accepted for API compatibility with newer stacked-PR
                 wrappers. Unused in this runtime.

    Raises:
        ValueError: unknown provider.
        RecursionGuardViolation: provider='opencode' inside an opencode session.
    """
    del progress_callback
    del api_key
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
