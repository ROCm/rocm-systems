"""Isolation tests for Memory-Techniques Specialist (Layer 2)."""

import pytest

from perfxpert.agents import memory_specialist as ms_module
from perfxpert.agents import schemas
from perfxpert.agents.framework import FakeProviderResponse, HandoffPolicyViolation, dispatch_handoff


def test_memory_specialist_builds():
    agent = ms_module.build_memory_specialist()
    assert agent.name == "MemoryTechniquesSpecialist"
    assert agent.layer == 2


def test_memory_specialist_tool_count_within_cap():
    agent = ms_module.build_memory_specialist()
    assert len(agent.tools) <= 5


def test_memory_specialist_no_execution_tools():
    agent = ms_module.build_memory_specialist()
    forbidden = {"patch.apply", "compile.build", "profile.run", "anchors.check"}
    declared = {t.name for t in agent.tools}
    assert not (declared & forbidden)


def test_memory_specialist_cannot_handoff_laterally():
    agent = ms_module.build_memory_specialist()
    with pytest.raises(HandoffPolicyViolation):
        dispatch_handoff(agent, "compute_specialist")
    with pytest.raises(HandoffPolicyViolation):
        dispatch_handoff(agent, "latency_specialist")


def test_memory_specialist_returns_techniques(fake_provider, monkeypatch):
    monkeypatch.setattr(ms_module, "_fetch_catalog", lambda gfx_id: [
        {"name": "coalesce_loads", "expected_impact": 0.5, "effort_factor": 1.0, "risk": "low"},
    ])
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "techniques": [{"name": "coalesce_loads", "expected_impact": 0.5}],
            "confidence": 0.9,
            "citations": ["AMD opt guide"],
        },
    )
    result = ms_module.run_memory_specialist(
        schemas.MemorySpecialistInput(
            gfx_id="gfx942",
            hot_kernels=[{"name": "[K1]", "pct": 0.4}],
            memcpy_data={"total_gb": 12.0},
        ),
        provider="anthropic",
    )
    assert isinstance(result, schemas.MemorySpecialistOutput)


def test_memory_specialist_airgap_sorts_deterministically(monkeypatch):
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.setattr(ms_module, "_fetch_catalog", lambda gfx_id: [
        {"name": "X", "expected_impact": 0.2, "effort_factor": 1.0, "risk": "low"},
        {"name": "Y", "expected_impact": 0.6, "effort_factor": 1.0, "risk": "low"},
    ])
    result = ms_module.run_memory_specialist(
        schemas.MemorySpecialistInput(gfx_id="gfx942", hot_kernels=[]),
        airgap=True,
    )
    assert result.techniques[0]["name"] == "Y"


def test_memory_specialist_airgap_promotes_stream_overlap(monkeypatch):
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.setattr(ms_module, "_fetch_catalog", lambda gfx_id: [
        {
            "name": "memory_coalescing_stride_fix",
            "expected_impact": 4.0,
            "effort_factor": 2.0,
            "risk": "medium",
        },
        {
            "name": "hip_stream_overlap",
            "expected_impact": 0.7,
            "effort_factor": 2.0,
            "risk": "low",
        },
    ])
    result = ms_module.run_memory_specialist(
        schemas.MemorySpecialistInput(gfx_id="gfx942", hot_kernels=[]),
        airgap=True,
    )
    assert result.techniques[0]["name"] == "hip_stream_overlap"
