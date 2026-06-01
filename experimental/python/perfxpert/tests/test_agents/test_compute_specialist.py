"""Isolation tests for Compute-Techniques Specialist (Layer 2)."""

import pytest
from unittest.mock import MagicMock

from perfxpert.agents import compute_specialist as cs_module
from perfxpert.agents import schemas
from perfxpert.agents.framework import (
    AgentConstructionError, FakeProviderResponse, HandoffPolicyViolation,
    dispatch_handoff,
)


def test_compute_specialist_builds():
    agent = cs_module.build_compute_specialist()
    assert agent.name == "ComputeTechniquesSpecialist"
    assert agent.layer == 2


def test_compute_specialist_tool_count_at_most_5():
    agent = cs_module.build_compute_specialist()
    assert len(agent.tools) <= 5


def test_compute_specialist_has_no_execution_tools():
    agent = cs_module.build_compute_specialist()
    forbidden = {"patch.apply", "patch.revert", "compile.build", "profile.run"}
    declared = {t.name for t in agent.tools}
    assert not (declared & forbidden)


def test_compute_specialist_cannot_handoff_to_layer2(monkeypatch):
    """Spec §2: no Layer-2 → Layer-2 handoffs."""
    agent = cs_module.build_compute_specialist()
    with pytest.raises(HandoffPolicyViolation):
        dispatch_handoff(agent, "memory_specialist")
    with pytest.raises(HandoffPolicyViolation):
        dispatch_handoff(agent, "latency_specialist")


def test_compute_specialist_cannot_handoff_upward():
    agent = cs_module.build_compute_specialist()
    with pytest.raises(HandoffPolicyViolation):
        dispatch_handoff(agent, "recommendation")


def test_compute_specialist_ranks_techniques_llm_mode(fake_provider):
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "techniques": [
                {"name": "launch_bounds", "rationale": "Reduce VGPR", "expected_impact": 0.25, "effort": "low", "risk": "low"},
                {"name": "mfma_enablement", "rationale": "Use MFMA intrinsics", "expected_impact": 0.50, "effort": "high", "risk": "medium"},
            ],
            "confidence": 0.85,
            "citations": ["GEAK+CDNA3 occupancy tables"],
        },
    )
    result = cs_module.run_compute_specialist(
        schemas.ComputeSpecialistInput(
            gfx_id="gfx942",
            hot_kernels=[{"name": "[KERNEL_1]", "pct": 0.60}],
            counter_data={"valu_util_pct": 0.75},
        ),
        provider="anthropic",
    )
    assert isinstance(result, schemas.ComputeSpecialistOutput)
    assert len(result.techniques) >= 1


def test_compute_specialist_airgap_returns_sorted_catalog(monkeypatch):
    """In airgap, specialist returns catalog sorted by impact/effort with
    no LLM filter. Rule-based ranking only."""
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    # Stub the catalog tool
    monkeypatch.setattr(
        cs_module, "_fetch_catalog",
        lambda gfx_id: [
            {"name": "A", "expected_impact": 0.3, "effort_factor": 1.0, "risk": "low"},
            {"name": "B", "expected_impact": 0.5, "effort_factor": 2.0, "risk": "medium"},
            {"name": "C", "expected_impact": 0.2, "effort_factor": 0.5, "risk": "low"},
        ],
    )
    result = cs_module.run_compute_specialist(
        schemas.ComputeSpecialistInput(
            gfx_id="gfx942",
            hot_kernels=[{"name": "[K1]", "pct": 0.7}],
        ),
        airgap=True,
    )
    names = [t["name"] for t in result.techniques]
    # A has score 0.3, B has 0.25, C has 0.4 → expected order: C, A, B
    assert names[0] == "C"


def test_compute_specialist_empty_catalog_returns_empty(monkeypatch):
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.setattr(cs_module, "_fetch_catalog", lambda gfx_id: [])
    result = cs_module.run_compute_specialist(
        schemas.ComputeSpecialistInput(gfx_id="gfx942", hot_kernels=[]),
        airgap=True,
    )
    assert result.techniques == []


def test_compute_specialist_airgap_promotes_mfma_for_matmul(monkeypatch):
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.setattr(
        cs_module,
        "_fetch_catalog",
        lambda gfx_id: [
            {
                "name": "fast_math_compiler_flag",
                "expected_impact": 2.5,
                "effort_factor": 0.5,
                "risk": "medium",
            },
            {
                "name": "mfma_enablement",
                "expected_impact": 9.0,
                "effort_factor": 3.0,
                "risk": "medium",
            },
        ],
    )
    result = cs_module.run_compute_specialist(
        schemas.ComputeSpecialistInput(
            gfx_id="gfx942",
            hot_kernels=[{"name": "matmul", "pct": 1.0}],
        ),
        airgap=True,
    )
    assert result.techniques[0]["name"] == "mfma_enablement"
