"""Cycle-4 B3 — fallback chain kicks in on the hot path.

Prior to the fix, ``PERFXPERT_LLM_FALLBACK_CHAIN=anthropic,openai``
``perfxpert analyze --llm openai`` with a 429 from OpenAI would
immediately surface the 429 — the ``FallbackProvider`` wrapper was
wired into ``build_session()`` but the ``run_root`` dispatcher ignored
it entirely, so the chain never got a chance.

These tests pin the new contract:

  1. ``AnalysisSession._fallback_chain()`` returns [primary, *fallbacks]
     with duplicates removed.
  2. ``AnalysisSession._cascade()`` retries on rate-limit-like errors
     but surfaces other errors immediately.
  3. ``run_root`` uses the cascade end-to-end: when the primary raises
     a rate-limit error, the second entry in the chain is invoked.
  4. When the whole chain rate-limits, a ``ProviderChainExhausted`` is
     raised (never a bare 429).
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from perfxpert.agents import runtime, schemas
from perfxpert.providers import ProviderChainExhausted
from perfxpert.providers._fallback import FallbackProvider


def _root_output_stub(marker: str) -> schemas.RootOutput:
    return schemas.RootOutput(
        narrative=f"ok from {marker}",
        recommendations=[
            {"type": "analyze", "target": "analysis", "summary": "x", "source": "test"}
        ],
        primary_bottleneck="mixed",
        warnings=[],
        metadata={"routed_to": "analysis", "intent": "analyze"},
    )


@pytest.fixture(autouse=True)
def _no_chain_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure the test env doesn't leak PERFXPERT_LLM_FALLBACK_CHAIN."""
    monkeypatch.delenv("PERFXPERT_LLM_FALLBACK_CHAIN", raising=False)


# ------------------------------------------------------------------
# _is_rate_limit_like detection
# ------------------------------------------------------------------


def test_is_rate_limit_like_catches_429_in_message() -> None:
    from perfxpert.agents.runtime import _is_rate_limit_like

    assert _is_rate_limit_like(RuntimeError("oops: 429 rate limit"))
    assert _is_rate_limit_like(RuntimeError("insufficient_quota"))
    assert _is_rate_limit_like(RuntimeError("Too Many Requests"))


def test_is_rate_limit_like_catches_class_name() -> None:
    from perfxpert.agents.runtime import _is_rate_limit_like

    class RateLimitError(Exception):
        pass

    assert _is_rate_limit_like(RateLimitError("something"))


def test_is_rate_limit_like_walks_cause_chain() -> None:
    from perfxpert.agents.runtime import _is_rate_limit_like

    inner = RuntimeError("429 rate_limit exceeded")
    outer = RuntimeError("framework: SDK Runner.run_sync failed")
    outer.__cause__ = inner
    assert _is_rate_limit_like(outer)


def test_is_rate_limit_like_negative_for_auth() -> None:
    from perfxpert.agents.runtime import _is_rate_limit_like

    assert not _is_rate_limit_like(RuntimeError("401 Unauthorized"))
    assert not _is_rate_limit_like(RuntimeError("SSL handshake failed"))
    assert not _is_rate_limit_like(ValueError("schema mismatch"))


# ------------------------------------------------------------------
# _fallback_chain composition
# ------------------------------------------------------------------


def test_fallback_chain_returns_just_primary_when_no_fallback() -> None:
    session = runtime.AnalysisSession(
        session_id="sid", provider="anthropic", airgap=False, fallback_provider=None
    )
    assert session._fallback_chain() == ["anthropic"]


def test_fallback_chain_merges_primary_with_env_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp = FallbackProvider(["openai", "anthropic", "opencode"])
    session = runtime.AnalysisSession(
        session_id="sid",
        provider="anthropic",
        airgap=False,
        fallback_provider=fp,
    )
    # Primary is position 0; duplicates (anthropic) dropped.
    assert session._fallback_chain() == ["anthropic", "openai", "opencode"]


# ------------------------------------------------------------------
# _cascade dispatch
# ------------------------------------------------------------------


def test_cascade_rate_limit_falls_through_to_next_provider() -> None:
    fp = FallbackProvider(["openai", "anthropic"])
    session = runtime.AnalysisSession(
        session_id="sid", provider="openai", airgap=False, fallback_provider=fp
    )

    calls: List[str] = []

    def runner(prov: str) -> str:
        calls.append(prov)
        if prov == "openai":
            raise RuntimeError("429: Too Many Requests (insufficient_quota)")
        return f"ok-{prov}"

    result = session._cascade(runner, op_name="test")
    assert result == "ok-anthropic"
    assert calls == ["openai", "anthropic"]


def test_cascade_non_rate_limit_error_surfaces_immediately() -> None:
    fp = FallbackProvider(["openai", "anthropic"])
    session = runtime.AnalysisSession(
        session_id="sid", provider="openai", airgap=False, fallback_provider=fp
    )

    calls: List[str] = []

    def runner(prov: str) -> str:
        calls.append(prov)
        raise RuntimeError("401 Unauthorized — bad key")

    with pytest.raises(RuntimeError, match="401"):
        session._cascade(runner, op_name="test")
    # Only the primary was called — auth errors are NOT cascade-worthy.
    assert calls == ["openai"]


def test_cascade_all_rate_limit_raises_chain_exhausted() -> None:
    fp = FallbackProvider(["openai", "anthropic"])
    session = runtime.AnalysisSession(
        session_id="sid", provider="openai", airgap=False, fallback_provider=fp
    )

    calls: List[str] = []

    def runner(prov: str) -> str:
        calls.append(prov)
        raise RuntimeError(f"429 rate limit on {prov}")

    with pytest.raises(ProviderChainExhausted) as ei:
        session._cascade(runner, op_name="test")

    # Provenance: the error lists every attempted provider.
    assert "openai" in ei.value.providers
    assert "anthropic" in ei.value.providers
    assert len(ei.value.attempts) == 2
    assert calls == ["openai", "anthropic"]


# ------------------------------------------------------------------
# End-to-end: run_root triggers the cascade on the hot path
# ------------------------------------------------------------------


def test_run_root_cascades_on_primary_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp = FallbackProvider(["openai", "anthropic"])
    session = runtime.AnalysisSession(
        session_id="sid", provider="openai", airgap=False, fallback_provider=fp
    )

    seen_providers: List[str] = []

    def fake_run_root(payload: Any, *, provider: str = "anthropic", airgap: Any = None, **_kw: Any) -> Any:
        seen_providers.append(provider)
        if provider == "openai":
            raise RuntimeError("framework: SDK failed: 429 rate limit")
        return _root_output_stub(provider)

    payload = schemas.RootInput(
        user_query="analyze this trace",
        provider="openai",
        airgap=False,
        session_id="sid",
    )

    with patch.object(runtime.root, "run_root", side_effect=fake_run_root):
        out = session.run_root(payload)

    assert seen_providers == ["openai", "anthropic"]
    assert "anthropic" in out.narrative


def test_run_root_no_fallback_propagates_original_error() -> None:
    """Smoke — no chain configured, old behaviour preserved."""
    session = runtime.AnalysisSession(
        session_id="sid",
        provider="openai",
        airgap=False,
        fallback_provider=None,
    )

    def fake_run_root(payload: Any, *, provider: str = "anthropic", airgap: Any = None, **_kw: Any) -> Any:
        raise RuntimeError("429 rate limit")

    payload = schemas.RootInput(user_query="hi", provider="openai", airgap=False, session_id="sid")

    with patch.object(runtime.root, "run_root", side_effect=fake_run_root):
        with pytest.raises(RuntimeError, match="429"):
            session.run_root(payload)


def test_build_session_from_env_wires_fallback_through_run_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: env var -> build_session -> run_root uses chain.

    This is the exact scenario in the blocker report:
    PERFXPERT_LLM_FALLBACK_CHAIN=anthropic,opencode perfxpert analyze --llm openai
    """
    monkeypatch.setenv("PERFXPERT_LLM_FALLBACK_CHAIN", "anthropic,opencode")

    session = runtime.build_session(provider="openai", airgap=False)
    assert session.fallback_provider is not None
    chain = session._fallback_chain()
    # Primary first (openai via --llm), then env entries, deduped.
    assert chain[0] == "openai"
    assert "anthropic" in chain
    assert "opencode" in chain

    seen: List[str] = []

    def fake_run_root(payload: Any, *, provider: str = "anthropic", airgap: Any = None, **_kw: Any) -> Any:
        seen.append(provider)
        if provider == "openai":
            raise RuntimeError("429 Too Many Requests")
        return _root_output_stub(provider)

    payload = schemas.RootInput(
        user_query="analyze", provider="openai", airgap=False, session_id=session.session_id
    )

    with patch.object(runtime.root, "run_root", side_effect=fake_run_root):
        out = session.run_root(payload)

    # Primary tried, failed, then anthropic — exactly what the user asked for.
    assert seen[0] == "openai"
    assert "anthropic" in seen
    assert "anthropic" in out.narrative
