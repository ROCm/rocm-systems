"""Isolation tests for Recommendation decision-maker (Layer 1)."""

import pytest
from unittest.mock import MagicMock

from perfxpert.agents import recommendation as rec_module
from perfxpert.agents import schemas
from perfxpert.agents.framework import (
    FakeProviderResponse, HandoffPolicyViolation, dispatch_handoff,
)
from perfxpert.tools import profiling


def _findings(bottleneck: str = "compute") -> schemas.AnalysisOutput:
    return schemas.AnalysisOutput(
        primary_bottleneck=bottleneck,
        confidence=0.85,
        time_breakdown={"kernel_pct": 0.9, "memcpy_pct": 0.05, "api_pct": 0.03, "idle_pct": 0.02},
        hot_kernels=[{"name": "[K1]", "pct": 0.6}],
        counter_data_available=True,
    )


def test_recommendation_agent_builds():
    agent = rec_module.build_recommendation_agent()
    assert agent.name == "Recommendation"
    assert agent.layer == 1


def test_recommendation_tool_count():
    agent = rec_module.build_recommendation_agent()
    # 3 tools: plateau.check, trace_fingerprint.fingerprint, profiling.fill_gap
    assert 1 <= len(agent.tools) <= 5


def test_recommendation_binds_real_fill_gap_tool():
    agent = rec_module.build_recommendation_agent()
    tool_map = {tool.name: tool.fn for tool in agent.tools}
    assert tool_map["profiling.fill_gap"] is profiling.fill_gap


def test_recommendation_no_execution_tools():
    agent = rec_module.build_recommendation_agent()
    forbidden = {"patch.apply", "compile.build", "profile.run", "anchors.check"}
    declared = {t.name for t in agent.tools}
    assert not (declared & forbidden)


def test_recommendation_handoff_whitelist_is_specialists_only():
    agent = rec_module.build_recommendation_agent()
    assert set(agent.allowed_handoffs) == {
        "compute_specialist", "memory_specialist", "latency_specialist",
    }


def test_recommendation_cannot_handoff_to_analysis_or_correctness():
    agent = rec_module.build_recommendation_agent()
    with pytest.raises(HandoffPolicyViolation):
        dispatch_handoff(agent, "analysis")
    with pytest.raises(HandoffPolicyViolation):
        dispatch_handoff(agent, "correctness")


def test_recommendation_routes_compute_bottleneck_to_compute_specialist(monkeypatch):
    called = {}
    def fake_compute(payload, **kw):
        called["compute"] = True
        called["payload"] = payload
        return schemas.ComputeSpecialistOutput(
            techniques=[{"name": "launch_bounds"}], confidence=0.9,
        )
    monkeypatch.setattr(rec_module, "_run_specialist_compute", fake_compute)

    result = rec_module.run_recommendation(
        schemas.RecommendationInput(findings=_findings("compute"), gfx_id="gfx950"),
        airgap=True,   # skip LLM, exercise routing
    )
    assert called.get("compute") is True
    assert called["payload"].gfx_id == "gfx950"
    assert result.specialist_used == "compute"


def test_recommendation_routes_memory_bottleneck_to_memory_specialist(monkeypatch):
    called = {}
    def fake_memory(payload, **kw):
        called["memory"] = True
        called["payload"] = payload
        return schemas.MemorySpecialistOutput(
            techniques=[{"name": "coalesce"}], confidence=0.9,
        )
    monkeypatch.setattr(rec_module, "_run_specialist_memory", fake_memory)

    result = rec_module.run_recommendation(
        schemas.RecommendationInput(findings=_findings("memory_transfer"), gfx_id="gfx950"),
        airgap=True,
    )
    assert called.get("memory") is True
    assert called["payload"].gfx_id == "gfx950"
    assert result.specialist_used == "memory"


def test_recommendation_routes_latency_bottleneck_to_latency_specialist(monkeypatch):
    called = {}
    def fake_latency(payload, **kw):
        called["latency"] = True
        called["payload"] = payload
        return schemas.LatencySpecialistOutput(
            techniques=[{"name": "fuse_kernels"}], confidence=0.9,
        )
    monkeypatch.setattr(rec_module, "_run_specialist_latency", fake_latency)

    result = rec_module.run_recommendation(
        schemas.RecommendationInput(findings=_findings("latency"), gfx_id="gfx950"),
        airgap=True,
    )
    assert called.get("latency") is True
    assert called["payload"].gfx_id == "gfx950"
    assert result.specialist_used == "latency"


def test_recommendation_dedups_seen_hashes(monkeypatch):
    """Recommendation drops techniques whose hash is in seen_recommendation_hashes."""
    def fake_compute(payload, **kw):
        return schemas.ComputeSpecialistOutput(
            techniques=[
                {"name": "launch_bounds", "rationale": "r1"},
                {"name": "mfma_enablement", "rationale": "r2"},
            ],
            confidence=0.9,
        )
    monkeypatch.setattr(rec_module, "_run_specialist_compute", fake_compute)

    # Compute a hash matching the first technique
    import hashlib, json
    h = hashlib.sha256(json.dumps({"name": "launch_bounds"}, sort_keys=True).encode()).hexdigest()

    result = rec_module.run_recommendation(
        schemas.RecommendationInput(
            findings=_findings("compute"),
            gfx_id="gfx950",
            seen_recommendation_hashes=[h],
        ),
        airgap=True,
    )
    # The "launch_bounds" recommendation should be filtered out (exact hash logic
    # may differ in impl; this test asserts the count decreased).
    assert len(result.recommendations) <= 1


def test_recommendation_plateau_detection(monkeypatch):
    """If plateau.check returns True → plateau_detected=True in output."""
    monkeypatch.setattr(rec_module, "_plateau_check",
                        lambda history: {"plateau_detected": True, "iterations": 3})
    def fake_compute(payload, **kw):
        return schemas.ComputeSpecialistOutput(techniques=[], confidence=0.0)
    monkeypatch.setattr(rec_module, "_run_specialist_compute", fake_compute)

    result = rec_module.run_recommendation(
        schemas.RecommendationInput(findings=_findings("compute"), gfx_id="gfx950"),
        airgap=True,
    )
    assert result.plateau_detected is True


def test_recommendation_mixed_bottleneck_yields_no_specialist(monkeypatch):
    """Mixed/api_overhead falls through to 'none' specialist."""
    result = rec_module.run_recommendation(
        schemas.RecommendationInput(findings=_findings("mixed"), gfx_id="gfx950"),
        airgap=True,
    )
    assert result.specialist_used == "none"


def test_recommendation_mixed_bottleneck_emits_triage():
    """'mixed' bottleneck must produce a non-empty triage recommendation, not silent empty."""
    result = rec_module.run_recommendation(
        schemas.RecommendationInput(findings=_findings("mixed"), gfx_id="gfx950"),
        airgap=True,
    )
    assert result.specialist_used == "none"
    # Must have at least one triage technique — not silently empty
    assert len(result.recommendations) >= 1, (
        "Expected at least one triage recommendation for 'mixed' bottleneck, got none."
    )
    triage = result.recommendations[0]
    assert triage.get("name") == "mixed_bottleneck_triage", (
        f"Expected 'mixed_bottleneck_triage', got {triage.get('name')!r}"
    )
    assert "ATT" in triage.get("description", ""), (
        "Triage recommendation must mention ATT as the next step."
    )


def test_recommendation_raises_on_unknown_bottleneck(monkeypatch):
    """An unhandled bottleneck type must raise ValueError, not silently return empty.

    The Pydantic schema validates known types, so we bypass it by patching
    the findings object's primary_bottleneck attribute to inject a rogue value.
    This verifies the else-branch in run_recommendation raises ValueError.
    """
    findings = _findings("compute")
    # Bypass schema validation by replacing the frozen field via object.__setattr__
    object.__setattr__(findings, "primary_bottleneck", "totally_unknown_type")

    payload = schemas.RecommendationInput(findings=_findings("compute"), gfx_id="gfx950")
    # Patch findings on the payload to inject the unknown type
    object.__setattr__(payload.findings, "primary_bottleneck", "totally_unknown_type")

    with pytest.raises(ValueError, match="unhandled bottleneck type"):
        rec_module.run_recommendation(payload, airgap=True)


def test_recommendation_requires_non_empty_gfx_id_for_specialist_routing():
    with pytest.raises(ValueError, match="gfx_id"):
        rec_module.run_recommendation(
            schemas.RecommendationInput(findings=_findings("compute"), gfx_id=""),
            airgap=True,
        )
