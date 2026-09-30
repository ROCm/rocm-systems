"""Isolation tests for Correctness decision-maker (Layer 1).

Gates run in middleware (runtime/gate_cascade.py) — Correctness consumes
a GateVerdict. Tests script verdict inputs directly.
"""

import pytest
from unittest.mock import MagicMock

from perfxpert.agents import correctness as cor_module
from perfxpert.agents import schemas
from perfxpert.agents.framework import FakeProviderResponse


def test_correctness_agent_builds():
    agent = cor_module.build_correctness_agent()
    assert agent.name == "Correctness"
    assert agent.layer == 1


def test_correctness_tool_count_at_most_5():
    agent = cor_module.build_correctness_agent()
    assert len(agent.tools) <= 5


def test_correctness_has_no_gate_tools():
    """Spec §5.0: Correctness NEVER owns gates.

    Any execution-class tool in the allowlist = spec violation.
    """
    agent = cor_module.build_correctness_agent()
    forbidden = {
        "compile.build", "sol.sanity_check",
        "patch.verify_output", "regression.compare_runs",
        "anchors.check", "patch.apply", "patch.revert", "profile.run",
    }
    declared = {t.name for t in agent.tools}
    assert not (declared & forbidden), (
        f"Correctness has execution/gate tools: {declared & forbidden}"
    )


def test_correctness_no_allowed_handoffs():
    agent = cor_module.build_correctness_agent()
    assert agent.allowed_handoffs == ()


def test_correctness_pass_verdict_returns_accept(fake_provider):
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "verdict": "pass", "action": "accept",
            "narrative": "All gates passed.",
            "alternative_technique": None,
        },
    )
    v = schemas.GateVerdictModel(status="pass", detail="all 5 gates passed")
    result = cor_module.run_correctness(
        schemas.CorrectnessInput(gate_verdict=v, kernel_name="[K1]"),
        provider="anthropic",
    )
    assert result.action == "accept"
    assert result.verdict == "pass"


def test_correctness_reject_verdict_creates_follow_up_task(monkeypatch):
    """On reject → tasks.create called for manual investigation."""
    created = {}
    def fake_create(**kw):
        created["called"] = True
        created["kw"] = kw
        return "task_123"
    monkeypatch.setattr(cor_module, "_tasks_create", fake_create)
    v = schemas.GateVerdictModel(
        status="reject", failing_gate="bitwise", detail="output diverged 0.1",
    )
    result = cor_module.run_correctness(
        schemas.CorrectnessInput(gate_verdict=v, kernel_name="[K1]"),
        airgap=True,
    )
    assert created.get("called") is True
    assert result.action == "reject_and_log"
    assert result.follow_up_task_id == "task_123"


def test_correctness_regressed_proposes_alternative(monkeypatch):
    """On regression → propose alternative NOT in tasks.query_by_kernel history."""
    # Prior attempt: launch_bounds. Alternative should differ.
    monkeypatch.setattr(cor_module, "_tasks_query_by_kernel",
                        lambda kernel_name, root=None: [{
                            "meta": {
                                "technique": "launch_bounds",
                                "candidate_alternative": "occupancy_tune",
                            }
                        }])
    v = schemas.GateVerdictModel(
        status="regressed", failing_gate="regression",
        detail="total +12%", delta_pct=12.0,
    )
    result = cor_module.run_correctness(
        schemas.CorrectnessInput(
            gate_verdict=v, kernel_name="[K1]", last_technique="launch_bounds",
        ),
        airgap=True,
    )
    assert result.action == "revert"
    assert result.verdict == "regressed"
    # Alternative, if any, must not match the already-tried technique
    if result.alternative_technique:
        assert result.alternative_technique != "launch_bounds"
        assert result.alternative_technique == "occupancy_tune"


def test_correctness_regressed_does_not_create_follow_up_task(monkeypatch):
    created = {}

    def fake_create(**kw):
        created["called"] = True
        return "task_123"

    monkeypatch.setattr(cor_module, "_tasks_create", fake_create)
    monkeypatch.setattr(cor_module, "_tasks_query_by_kernel", lambda kernel_name, root=None: [])
    v = schemas.GateVerdictModel(
        status="regressed", failing_gate="regression",
        detail="total +12%", delta_pct=12.0,
    )
    result = cor_module.run_correctness(
        schemas.CorrectnessInput(gate_verdict=v, kernel_name="[K1]"),
        airgap=True,
    )
    assert created == {}
    assert result.follow_up_task_id is None


def test_correctness_reject_requires_compatible_tasks_create_signature(monkeypatch):
    def bad_create(title):
        return title

    monkeypatch.setattr(cor_module, "_tasks_create", bad_create)
    v = schemas.GateVerdictModel(
        status="reject", failing_gate="bitwise", detail="output diverged 0.1",
    )
    with pytest.raises(TypeError, match="tasks.create binding"):
        cor_module.run_correctness(
            schemas.CorrectnessInput(gate_verdict=v, kernel_name="[K1]"),
            airgap=True,
        )


def test_correctness_llm_alternative_is_filtered_against_history(fake_provider, monkeypatch):
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "narrative": "Try launch_bounds again.",
            "alternative_technique": "launch_bounds",
        },
    )
    monkeypatch.setattr(
        cor_module,
        "_tasks_query_by_kernel",
        lambda kernel_name, root=None: [{"meta": {"technique": "launch_bounds"}}],
    )
    v = schemas.GateVerdictModel(
        status="regressed", failing_gate="regression",
        detail="total +12%", delta_pct=12.0,
    )
    result = cor_module.run_correctness(
        schemas.CorrectnessInput(
            gate_verdict=v, kernel_name="[K1]", last_technique="launch_bounds",
        ),
        provider="anthropic",
        airgap=False,
    )
    assert result.alternative_technique is None


def test_correctness_uses_source_dir_task_store(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    cwd_dir = tmp_path / "cwd"
    project_dir.mkdir()
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    from perfxpert.tools import tasks as tasks_tool

    tasks_tool.create_at(
        str(project_dir),
        "Tried launch_bounds",
        meta={
            "kernel": "[K1]",
            "technique": "launch_bounds",
            "candidate_alternative": "occupancy_tune",
        },
    )

    v = schemas.GateVerdictModel(
        status="regressed", failing_gate="regression",
        detail="total +12%", delta_pct=12.0,
    )
    result = cor_module.run_correctness(
        schemas.CorrectnessInput(
            gate_verdict=v,
            kernel_name="[K1]",
            last_technique="launch_bounds",
            source_dir=str(project_dir),
        ),
        airgap=True,
    )

    assert result.alternative_technique == "occupancy_tune"
    assert not (cwd_dir / ".beads" / "tasks.db").exists()


def test_correctness_airgap_narrative_is_deterministic_template(monkeypatch):
    """Air-gap narrative uses template: "Gate {id} {verdict}: {reason}"."""
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.setattr(cor_module, "_tasks_query_by_kernel",
                        lambda kernel_name, root=None: [])
    v = schemas.GateVerdictModel(
        status="regressed", failing_gate="regression",
        detail="total +12%", delta_pct=12.0,
    )
    result = cor_module.run_correctness(
        schemas.CorrectnessInput(gate_verdict=v, kernel_name="[K1]"),
        airgap=True,
    )
    assert "regression" in result.narrative.lower()
    assert "12" in result.narrative or "regressed" in result.narrative.lower()


def test_correctness_does_not_invoke_gates(monkeypatch):
    """Spec §5.0: Correctness NEVER calls compile/sol/bitwise/regression/anchors."""
    from perfxpert.runtime import gate_cascade
    original_evaluate = gate_cascade.evaluate

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Correctness must not invoke gate_cascade.evaluate")
    monkeypatch.setattr(gate_cascade, "evaluate", fail_if_called)

    v = schemas.GateVerdictModel(status="pass", detail="ok")
    # Should not raise — Correctness consumes the verdict; doesn't evaluate it.
    cor_module.run_correctness(
        schemas.CorrectnessInput(gate_verdict=v, kernel_name="[K1]"),
        airgap=True,
    )
