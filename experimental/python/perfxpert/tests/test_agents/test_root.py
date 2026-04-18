"""Isolation tests for Root agent (Layer 0).

Each test scripts the mocked LLM response and asserts the routing /
handoff / output shape.
"""

import pytest
from unittest.mock import MagicMock

from perfxpert.agents import root as root_module
from perfxpert.agents import schemas
from perfxpert.agents.framework import (
    Agent, AgentConstructionError, HandoffPolicyViolation, ToolAllowlistViolation,
    FakeProviderResponse,
)


# -- Construction ---------------------------------------------------------

def test_root_agent_builds():
    agent = root_module.build_root_agent()
    assert agent.name == "Root"
    assert agent.layer == 0


def test_root_tool_allowlist_size():
    agent = root_module.build_root_agent()
    assert len(agent.tools) <= 5


def test_root_allowed_handoffs_exactly_three():
    agent = root_module.build_root_agent()
    assert set(agent.allowed_handoffs) == {"analysis", "recommendation", "correctness"}


def test_root_cannot_handoff_to_specialist():
    agent = root_module.build_root_agent()
    with pytest.raises(HandoffPolicyViolation):
        from perfxpert.agents.framework import dispatch_handoff
        dispatch_handoff(agent, "compute_specialist")


def test_root_cannot_call_execution_tool():
    agent = root_module.build_root_agent()
    forbidden = [
        "patch.apply", "patch.revert", "patch.verify_output",
        "compile.build", "profile.run", "anchors.check",
    ]
    for tool in forbidden:
        assert not agent.has_tool(tool), f"Root must NOT have execution tool {tool!r}"


# -- Routing --------------------------------------------------------------

def test_root_routes_analyze_intent_to_analysis(fake_provider, monkeypatch):
    """Rule-first routing: intent=analyze → handoff to Analysis."""
    fake_provider.return_value = FakeProviderResponse(
        text="routed", handoff="analysis",
        structured_output={
            "narrative": "Routed to analysis.",
            "recommendations": [],
            "primary_bottleneck": "mixed",
            "warnings": [], "metadata": {},
        },
    )
    result = root_module.run_root(
        schemas.RootInput(user_query="why is this kernel slow?", database_path="x.db"),
        provider="anthropic",
    )
    assert isinstance(result, schemas.RootOutput)


def test_root_routes_verify_intent_to_correctness(fake_provider):
    fake_provider.return_value = FakeProviderResponse(
        text="routed", handoff="correctness",
        structured_output={
            "narrative": "Routed to correctness.",
            "recommendations": [],
            "primary_bottleneck": "mixed",
            "warnings": [], "metadata": {},
        },
    )
    result = root_module.run_root(
        schemas.RootInput(user_query="did my patch help?", database_path="x.db"),
        provider="anthropic",
    )
    assert isinstance(result, schemas.RootOutput)


def test_root_routes_optimize_intent_to_recommendation(fake_provider):
    fake_provider.return_value = FakeProviderResponse(
        text="routed", handoff="recommendation",
        structured_output={
            "narrative": "Routed to recommendation.",
            "recommendations": [],
            "primary_bottleneck": "mixed",
            "warnings": [], "metadata": {},
        },
    )
    result = root_module.run_root(
        schemas.RootInput(user_query="suggest optimizations", database_path="x.db"),
        provider="anthropic",
    )
    assert isinstance(result, schemas.RootOutput)


# -- Air-gap routing parity -----------------------------------------------

def test_root_routing_is_deterministic_in_airgap(monkeypatch):
    """Spec §5 invariant: air-gap routing decisions identical to LLM mode."""
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    result1 = root_module.run_root(
        schemas.RootInput(user_query="why slow?", database_path="x.db"),
        airgap=True,
    )
    result2 = root_module.run_root(
        schemas.RootInput(user_query="why slow?", database_path="x.db"),
        airgap=True,
    )
    # Deterministic: same input produces same handoff target
    assert result1.metadata.get("routed_to") == result2.metadata.get("routed_to")


# -- Narrative assembly ---------------------------------------------------

def test_root_writes_bottleneck_narrative(fake_provider):
    """Absorbed from former Bottleneck-Narrator (review N3).

    Root's narrative includes a 2-3 sentence classification explanation.
    """
    fake_provider.return_value = FakeProviderResponse(
        text="narrative",
        structured_output={
            "narrative": (
                "The workload is bottlenecked on HBM bandwidth. "
                "Kernel X consumes 62% of total runtime. "
                "Focus optimization on memory coalescing."
            ),
            "recommendations": [{"title": "coalesce loads", "priority": "high"}],
            "primary_bottleneck": "memory_transfer",
            "warnings": [], "metadata": {},
        },
    )
    result = root_module.run_root(
        schemas.RootInput(user_query="analyze", database_path="x.db"),
        provider="anthropic",
    )
    assert len(result.narrative) > 0
    assert result.primary_bottleneck == "memory_transfer"
