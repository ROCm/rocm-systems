"""Tests for perfxpert.agents.runtime — session factory + provider wiring."""

import pytest

from perfxpert.agents import runtime as runtime_module
from perfxpert.agents import schemas
from perfxpert.runtime import RecursionGuardViolation


def test_session_builds_with_anthropic(monkeypatch):
    # Stub so provider lookup succeeds without API key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    s = runtime_module.build_session(provider="anthropic")
    assert s.provider == "anthropic"
    assert s.session_id is not None


def test_session_airgap_has_no_provider():
    s = runtime_module.build_session(airgap=True)
    assert s.airgap is True


def test_session_rejects_opencode_recursion(monkeypatch):
    monkeypatch.setenv("PERFXPERT_IN_OPENCODE_SESSION", "1")
    with pytest.raises(RecursionGuardViolation):
        runtime_module.build_session(provider="opencode")


def test_session_honors_PERFXPERT_AIRGAP(monkeypatch):
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    s = runtime_module.build_session()
    assert s.airgap is True


def test_session_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        runtime_module.build_session(provider="my-fake-llm")


def test_session_generates_session_id_if_missing():
    s = runtime_module.build_session(airgap=True)
    assert s.session_id is not None
    assert len(s.session_id) > 5


def test_session_preserves_explicit_session_id():
    s = runtime_module.build_session(airgap=True, session_id="abc-123")
    assert s.session_id == "abc-123"


def test_session_run_root_returns_root_output(monkeypatch):
    """The session.run_root() facade returns a RootOutput."""
    s = runtime_module.build_session(airgap=True)
    result = s.run_root(schemas.RootInput(user_query="why slow?", database_path=None))
    assert isinstance(result, schemas.RootOutput)


def test_build_session_uses_fallback_chain_when_env_set(monkeypatch):
    """Cycle-2 I8: PERFXPERT_LLM_FALLBACK_CHAIN must wire into build_session.

    With the env chain set to ``openai,anthropic`` and the primary provider
    arg defaulting to anthropic, the session's ``fallback_provider`` must
    be a FallbackProvider whose primary entry is ``openai`` (from the env
    chain). The ``provider`` string still reflects the primary name used
    by run_* dispatchers.
    """
    from perfxpert.providers._fallback import FallbackProvider

    monkeypatch.setenv("PERFXPERT_LLM_FALLBACK_CHAIN", "openai,anthropic")
    s = runtime_module.build_session(provider="anthropic")

    assert s.fallback_provider is not None
    assert isinstance(s.fallback_provider, FallbackProvider)
    # Primary == first entry in the env chain (openai).
    first_entry = s.fallback_provider._providers[0]
    assert first_entry == "openai"
    # And anthropic is present as the second entry (explicit primary was
    # already in the chain so no dedup-prepend happened).
    assert "anthropic" in s.fallback_provider._providers


def test_build_session_prepends_explicit_primary_when_not_in_chain(monkeypatch):
    """If ``provider=`` is set AND differs from every chain entry, prepend
    it so the explicit choice is tried first."""
    from perfxpert.providers._fallback import FallbackProvider

    monkeypatch.setenv("PERFXPERT_LLM_FALLBACK_CHAIN", "openai,ollama")
    s = runtime_module.build_session(provider="anthropic")
    assert isinstance(s.fallback_provider, FallbackProvider)
    assert s.fallback_provider._providers[0] == "anthropic"
    assert list(s.fallback_provider._providers) == ["anthropic", "openai", "ollama"]


def test_build_session_airgap_ignores_fallback_chain(monkeypatch):
    """Airgap short-circuits provider resolution — chain must be ignored."""
    monkeypatch.setenv("PERFXPERT_LLM_FALLBACK_CHAIN", "openai,anthropic")
    s = runtime_module.build_session(airgap=True)
    assert s.airgap is True
    assert s.provider is None
    assert s.fallback_provider is None


def test_build_session_no_chain_env_leaves_fallback_unset(monkeypatch):
    """Default path: no chain env → no FallbackProvider built."""
    monkeypatch.delenv("PERFXPERT_LLM_FALLBACK_CHAIN", raising=False)
    s = runtime_module.build_session(provider="anthropic")
    assert s.fallback_provider is None
    assert s.provider == "anthropic"


def test_fallback_provider_complete_refuses_airgap(monkeypatch):
    """Defensive guard: FallbackProvider.complete() MUST NOT run live
    calls under PERFXPERT_AIRGAP=1 (regardless of how the chain was
    built). dry_run=True stays allowed for deterministic cost paths.
    """
    from perfxpert.providers._fallback import FallbackProvider

    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    fp = FallbackProvider(["anthropic"])

    with pytest.raises(RuntimeError, match="PERFXPERT_AIRGAP"):
        fp.complete([{"role": "user", "content": "hi"}])
