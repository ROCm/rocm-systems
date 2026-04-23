"""Tests for perfxpert.agents.runtime — session factory + provider wiring."""

import os

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


@pytest.mark.parametrize(
    ("method_name", "module_name", "func_name", "payload"),
    [
        (
            "run_compute_specialist",
            "compute_specialist",
            "run_compute_specialist",
            schemas.ComputeSpecialistInput(gfx_id="gfx942", hot_kernels=[]),
        ),
        (
            "run_memory_specialist",
            "memory_specialist",
            "run_memory_specialist",
            schemas.MemorySpecialistInput(gfx_id="gfx942", hot_kernels=[]),
        ),
        (
            "run_latency_specialist",
            "latency_specialist",
            "run_latency_specialist",
            schemas.LatencySpecialistInput(gfx_id="gfx942", hot_kernels=[]),
        ),
        (
            "run_diff_specialist",
            "diff_specialist",
            "run_diff_specialist",
            schemas.DiffSpecialistInput(
                baseline_db="baseline.db",
                new_db="new.db",
                top_kernels=1,
            ),
        ),
    ],
)
def test_session_restored_specialist_facades_dispatch(
    monkeypatch,
    method_name,
    module_name,
    func_name,
    payload,
):
    session = runtime_module.build_session(airgap=True)
    sentinel = object()

    def _fake_runner(arg, *, airgap=None, provider="anthropic"):
        assert arg == payload
        assert airgap is True
        return sentinel

    monkeypatch.setattr(
        getattr(runtime_module, module_name),
        func_name,
        _fake_runner,
    )

    assert getattr(session, method_name)(payload) is sentinel


def test_session_live_call_scopes_explicit_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-old")
    seen = {}
    expected = schemas.RootOutput(
        narrative="ok",
        recommendations=[],
        primary_bottleneck="mixed",
        warnings=[],
        metadata={},
    )

    def _fake_run_root(payload, *, provider="anthropic", airgap=None):
        assert isinstance(payload, schemas.RootInput)
        assert provider == "openai"
        assert airgap is None
        seen["during"] = os.environ.get("OPENAI_API_KEY")
        return expected

    monkeypatch.setattr(runtime_module.root, "run_root", _fake_run_root)

    session = runtime_module.build_session(provider="openai", api_key="sk-new")
    assert session.run_root(
        schemas.RootInput(user_query="why slow?", database_path=None)
    ) == expected
    assert seen["during"] == "sk-new"
    assert os.environ.get("OPENAI_API_KEY") == "sk-old"
