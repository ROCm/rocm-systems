"""Isolation tests for Latency-Techniques Specialist (Layer 2)."""

import pytest

from perfxpert.agents import latency_specialist as ls_module
from perfxpert.agents import schemas
from perfxpert.agents.framework import FakeProviderResponse, HandoffPolicyViolation, dispatch_handoff


def test_latency_specialist_builds():
    agent = ls_module.build_latency_specialist()
    assert agent.name == "LatencyTechniquesSpecialist"
    assert agent.layer == 2


def test_latency_specialist_tool_count_within_cap():
    agent = ls_module.build_latency_specialist()
    assert len(agent.tools) <= 5


def test_latency_specialist_no_execution_tools():
    agent = ls_module.build_latency_specialist()
    forbidden = {"patch.apply", "compile.build", "profile.run", "anchors.check"}
    declared = {t.name for t in agent.tools}
    assert not (declared & forbidden)


def test_latency_specialist_cannot_handoff_laterally():
    agent = ls_module.build_latency_specialist()
    with pytest.raises(HandoffPolicyViolation):
        dispatch_handoff(agent, "compute_specialist")


def test_latency_specialist_returns_techniques(fake_provider, monkeypatch):
    monkeypatch.setattr(ls_module, "_fetch_catalog", lambda gfx_id: [
        {"name": "fuse_kernels", "expected_impact": 0.4, "effort_factor": 1.0, "risk": "low"},
    ])
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "techniques": [{"name": "fuse_kernels", "expected_impact": 0.4}],
            "confidence": 0.75,
            "citations": ["HIP best practices"],
        },
    )
    result = ls_module.run_latency_specialist(
        schemas.LatencySpecialistInput(
            gfx_id="gfx942",
            hot_kernels=[{"name": "[K1]", "pct": 0.1}],
            api_overhead_pct=0.30,
            avg_kernel_duration_us=4.2,
        ),
        provider="anthropic",
    )
    assert isinstance(result, schemas.LatencySpecialistOutput)


def test_latency_specialist_airgap_sorts_deterministically(monkeypatch):
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.setattr(ls_module, "_fetch_catalog", lambda gfx_id: [
        {"name": "A", "expected_impact": 0.1, "effort_factor": 1.0, "risk": "low"},
        {"name": "B", "expected_impact": 0.7, "effort_factor": 1.0, "risk": "low"},
    ])
    result = ls_module.run_latency_specialist(
        schemas.LatencySpecialistInput(gfx_id="gfx942", hot_kernels=[]),
        airgap=True,
    )
    assert result.techniques[0]["name"] == "B"
