"""Public Python API contract tests."""

from __future__ import annotations


def test_python_api_exports_documented_surface() -> None:
    from perfxpert import api

    assert set(api.__all__) == {
        "agent_root",
        "agent_analysis",
        "agent_recommendation",
        "agent_correctness",
        "agent_compute_specialist",
        "agent_memory_specialist",
        "agent_latency_specialist",
        "agent_diff_specialist",
        "trace_diff_diff_runs",
    }


def test_python_api_mirrors_mcp_tool_implementations() -> None:
    from perfxpert import api
    from perfxpert.tools.agents.analysis import agent_analysis
    from perfxpert.tools.agents.compute import agent_compute_specialist
    from perfxpert.tools.agents.correctness import agent_correctness
    from perfxpert.tools.agents.diff import agent_diff_specialist
    from perfxpert.tools.agents.latency import agent_latency_specialist
    from perfxpert.tools.agents.memory import agent_memory_specialist
    from perfxpert.tools.agents.recommendation import agent_recommendation
    from perfxpert.tools.agents.root import agent_root
    from perfxpert.tools.trace_diff import diff_runs

    assert api.agent_root is agent_root
    assert api.agent_analysis is agent_analysis
    assert api.agent_recommendation is agent_recommendation
    assert api.agent_correctness is agent_correctness
    assert api.agent_compute_specialist is agent_compute_specialist
    assert api.agent_memory_specialist is agent_memory_specialist
    assert api.agent_latency_specialist is agent_latency_specialist
    assert api.agent_diff_specialist is agent_diff_specialist
    assert api.trace_diff_diff_runs is diff_runs
