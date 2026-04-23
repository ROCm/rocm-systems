"""Tests for perfxpert.agents.runtime — session factory + provider wiring."""

import pytest

from perfxpert.agents import runtime as runtime_module
from perfxpert.agents import schemas
from perfxpert.runtime import RecursionGuardViolation, recursion_guard


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


def test_session_rejects_opencode_recursion_from_local_session_state():
    with recursion_guard.opencode_session():
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
