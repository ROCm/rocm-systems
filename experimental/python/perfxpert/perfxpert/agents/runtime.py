"""Session factory + provider wiring.

Single entry point shared by: batch CLI, library API, MCP server.

Responsibilities:
- Pick provider from arg / env / config
- Validate against PROVIDER_REGISTRY (providers/__init__.py)
- Call runtime.recursion_guard.ensure_not_recursive(provider)
- Honor PERFXPERT_AIRGAP env var
- Generate session_id if missing
- Expose run_root / run_correctness / run_analysis / run_recommendation
  plus the restored specialist session methods used by the public agent
  wrappers.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from perfxpert.agents import (
    analysis,
    compute_specialist,
    correctness,
    diff_specialist,
    latency_specialist,
    memory_specialist,
    recommendation,
    root,
    schemas,
)
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


_PROVIDER_CANONICAL_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "private": "PERFXPERT_LLM_PRIVATE_API_KEY",
    "ollama": "PERFXPERT_LLM_OLLAMA_KEY",
    "opencode": "PERFXPERT_LLM_OPENCODE_KEY",
}


@contextlib.contextmanager
def _override_provider_env(
    provider: Optional[str], api_key: Optional[str]
) -> Iterator[None]:
    """Temporarily inject ``api_key`` into the provider's canonical env var."""
    if not provider or not api_key:
        yield
        return
    env_name = _PROVIDER_CANONICAL_ENV.get(provider)
    if not env_name:
        yield
        return

    sentinel = object()
    previous = os.environ.get(env_name, sentinel)
    os.environ[env_name] = api_key
    try:
        yield
    finally:
        if previous is sentinel:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous  # type: ignore[assignment]


@dataclass(frozen=True)
class AnalysisSession:
    """Handle returned by build_session() — expose run_* wrappers."""

    session_id: str
    provider: Optional[str]
    airgap: bool
    api_key: Optional[str] = None

    def _provider_name(self) -> str:
        return self.provider or DEFAULT_PROVIDER

    def _run_live(self, fn: Callable[[str], object]) -> object:
        provider = self._provider_name()
        with _override_provider_env(provider, self.api_key):
            return fn(provider)

    def run_root(
        self,
        payload: schemas.RootInput,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.RootOutput:
        del progress_callback
        if self.airgap:
            return root.run_root(payload, airgap=True)
        return self._run_live(
            lambda prov: root.run_root(payload, provider=prov)
        )  # type: ignore[return-value]

    def run_analysis(
        self,
        payload: schemas.AnalysisInput,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.AnalysisOutput:
        del progress_callback
        if self.airgap:
            return analysis.run_analysis(payload, airgap=True)
        return self._run_live(
            lambda prov: analysis.run_analysis(payload, provider=prov)
        )  # type: ignore[return-value]

    def run_recommendation(
        self,
        payload: schemas.RecommendationInput,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.RecommendationOutput:
        del progress_callback
        if self.airgap:
            return recommendation.run_recommendation(payload, airgap=True)
        return self._run_live(
            lambda prov: recommendation.run_recommendation(payload, provider=prov)
        )  # type: ignore[return-value]

    def run_correctness(
        self,
        payload: schemas.CorrectnessInput,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.CorrectnessOutput:
        del progress_callback
        if self.airgap:
            return correctness.run_correctness(payload, airgap=True)
        return self._run_live(
            lambda prov: correctness.run_correctness(payload, provider=prov)
        )  # type: ignore[return-value]

    def run_compute_specialist(
        self,
        payload: schemas.ComputeSpecialistInput,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.ComputeSpecialistOutput:
        del progress_callback
        if self.airgap:
            return compute_specialist.run_compute_specialist(payload, airgap=True)
        return self._run_live(
            lambda prov: compute_specialist.run_compute_specialist(
                payload, provider=prov
            )
        )  # type: ignore[return-value]

    def run_memory_specialist(
        self,
        payload: schemas.MemorySpecialistInput,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.MemorySpecialistOutput:
        del progress_callback
        if self.airgap:
            return memory_specialist.run_memory_specialist(payload, airgap=True)
        return self._run_live(
            lambda prov: memory_specialist.run_memory_specialist(
                payload, provider=prov
            )
        )  # type: ignore[return-value]

    def run_latency_specialist(
        self,
        payload: schemas.LatencySpecialistInput,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.LatencySpecialistOutput:
        del progress_callback
        if self.airgap:
            return latency_specialist.run_latency_specialist(payload, airgap=True)
        return self._run_live(
            lambda prov: latency_specialist.run_latency_specialist(
                payload, provider=prov
            )
        )  # type: ignore[return-value]

    def run_diff_specialist(
        self,
        payload: schemas.DiffSpecialistInput,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> schemas.DiffSpecialistOutput:
        del progress_callback
        if self.airgap:
            return diff_specialist.run_diff_specialist(payload, airgap=True)
        return self._run_live(
            lambda prov: diff_specialist.run_diff_specialist(payload, provider=prov)
        )  # type: ignore[return-value]


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
        progress_callback: Accepted for compatibility with the public agent
                           wrappers. This runtime does not stream progress.
        api_key: Optional explicit API key forwarded via the provider's
                 canonical env var for the duration of each live call.

    Raises:
        ValueError: unknown provider.
        RecursionGuardViolation: provider='opencode' inside an opencode session.
    """
    del progress_callback
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
        api_key=None if is_airgap else api_key,
    )


__all__ = ["AnalysisSession", "build_session", "PROVIDER_REGISTRY", "DEFAULT_PROVIDER"]
