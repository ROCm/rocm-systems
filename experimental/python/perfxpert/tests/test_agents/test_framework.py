"""Tests for perfxpert.agents.framework — the SDK facade."""

import pytest

from perfxpert.agents import framework
from perfxpert.agents.framework import (
    Agent,
    AgentConstructionError,
    Handoff,
    ToolBinding,
    run_agent,
)


# -- AgentSpec / Agent construction ----------------------------------------

def test_agent_construction_enforces_tool_cap():
    """Spec §2: ≤ 5 tools per agent."""
    too_many_tools = [ToolBinding(name=f"tool_{i}", fn=lambda x: x) for i in range(6)]
    with pytest.raises(AgentConstructionError, match="tool"):
        Agent(
            name="Overloaded",
            layer=1,
            fence_path="does-not-matter.md",
            input_schema=dict,
            output_schema=dict,
            tools=too_many_tools,
        )


def test_agent_construction_accepts_5_tools():
    tools = [ToolBinding(name=f"tool_{i}", fn=lambda x: x) for i in range(5)]
    a = Agent(
        name="MaxTools",
        layer=1,
        fence_path=None,
        input_schema=dict,
        output_schema=dict,
        tools=tools,
    )
    assert len(a.tools) == 5


def test_agent_normalizes_mutable_collections_to_immutable_sequences():
    tools = [ToolBinding(name="tool_0", fn=lambda x: x)]
    handoffs = ["analysis"]
    agent = Agent(
        name="Immutable",
        layer=1,
        fence_path=None,
        input_schema=dict,
        output_schema=dict,
        tools=tools,
        allowed_handoffs=handoffs,
    )

    tools.append(ToolBinding(name="tool_1", fn=lambda x: x))
    handoffs.append("recommendation")

    assert agent.tools == (ToolBinding(name="tool_0", fn=tools[0].fn),)
    assert agent.allowed_handoffs == ("analysis",)


def test_agent_rejects_post_construction_mutation():
    agent = Agent(
        name="Frozen",
        layer=1,
        fence_path=None,
        input_schema=dict,
        output_schema=dict,
        tools=[],
    )

    with pytest.raises((AttributeError, TypeError)):
        agent.tools += (ToolBinding(name="tool_0", fn=lambda x: x),)


def test_agent_builder_outputs_immutable_sequences():
    from perfxpert.agents import analysis as analysis_module

    agent = analysis_module.build_analysis_agent()
    assert isinstance(agent.tools, tuple)
    assert isinstance(agent.allowed_handoffs, tuple)


def test_agent_construction_enforces_fence_line_cap(tmp_path):
    big_fence = tmp_path / "big.md"
    big_fence.write_text("\n".join(f"line {i}" for i in range(401)))
    with pytest.raises(AgentConstructionError, match="fence"):
        Agent(
            name="Bloated",
            layer=1,
            fence_path=str(big_fence),
            input_schema=dict,
            output_schema=dict,
            tools=[],
        )


def test_agent_construction_accepts_400_line_fence(tmp_path):
    fence = tmp_path / "ok.md"
    fence.write_text("\n".join(f"line {i}" for i in range(400)))
    a = Agent(
        name="OK",
        layer=1,
        fence_path=str(fence),
        input_schema=dict,
        output_schema=dict,
        tools=[],
    )
    assert a.fence_line_count == 400


# -- Handoff whitelist -----------------------------------------------------

def test_handoff_rejects_layer2_to_layer2():
    """Spec §2 rule: no Layer-2 → Layer-2 handoffs."""
    with pytest.raises(AgentConstructionError, match="layer"):
        Handoff(
            source_layer=2,
            target_layer=2,
            source_name="compute_specialist",
            target_name="memory_specialist",
        )


def test_handoff_allows_root_to_layer1():
    h = Handoff(source_layer=0, target_layer=1, source_name="root", target_name="analysis")
    assert h.target_name == "analysis"


def test_handoff_allows_layer1_to_layer2_from_recommendation():
    h = Handoff(
        source_layer=1, target_layer=2,
        source_name="recommendation", target_name="compute_specialist",
    )
    assert h.source_name == "recommendation"


def test_handoff_rejects_upward():
    with pytest.raises(AgentConstructionError, match="downward"):
        Handoff(source_layer=2, target_layer=1,
                source_name="compute_specialist", target_name="recommendation")


def test_handoff_rejects_skip_root_to_layer2():
    with pytest.raises(AgentConstructionError, match="skip"):
        Handoff(source_layer=0, target_layer=2,
                source_name="root", target_name="compute_specialist")


# -- Tool dispatch guard ---------------------------------------------------

def test_tool_dispatch_blocks_out_of_allowlist(fake_provider):
    """Agent cannot call a tool not in its allowlist."""
    allowed = ToolBinding(name="analysis.time_breakdown", fn=lambda db: {})
    agent = Agent(
        name="Test", layer=1, fence_path=None,
        input_schema=dict, output_schema=dict, tools=[allowed],
    )
    # Simulate SDK producing a tool call the agent isn't allowed to make
    with pytest.raises(framework.ToolAllowlistViolation):
        framework.dispatch_tool(agent, "profile.run", {"cmd": "rm -rf /"})


# -- Airgap fallback -------------------------------------------------------

def test_run_agent_airgap_uses_template(monkeypatch, tmp_path):
    """With PERFXPERT_AIRGAP=1, no SDK call is made; templates drive output."""
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")

    fence = tmp_path / "x.md"
    fence.write_text("short fence")
    agent = Agent(
        name="T", layer=1, fence_path=str(fence),
        input_schema=dict, output_schema=dict, tools=[],
    )

    # If run_agent tried to call the SDK we'd get an AttributeError; with
    # airgap the facade must bypass it entirely.
    result = run_agent(agent, input_payload={"user_query": "why slow?"}, airgap=True)
    assert result is not None
    assert "airgap" in result.get("_mode", "").lower() or result.get("airgap") is True


# -- Provider selection pass-through --------------------------------------

def test_run_agent_passes_provider_to_sdk(fake_provider):
    from perfxpert.agents.framework import FakeProviderResponse  # type: ignore

    fence = None
    agent = Agent(
        name="P", layer=1, fence_path=fence,
        input_schema=dict, output_schema=dict, tools=[],
    )
    fake_provider.return_value = FakeProviderResponse(text="ok", structured_output={"x": 1})

    run_agent(agent, input_payload={"user_query": "?"}, provider="anthropic")

    # Assert the facade forwarded "anthropic" to the SDK call
    called_args = fake_provider.call_args
    assert "anthropic" in str(called_args)
