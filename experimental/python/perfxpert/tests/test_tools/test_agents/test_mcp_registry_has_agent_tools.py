"""The MCP registry exposes the 7 agent tools as READ_ONLY.

Regression guard for the agents-as-MCP-tools surface: the auto-discovery
walker must recurse into ``perfxpert.tools.agents`` so every agent in
the hierarchy is callable from backend TUIs without a forced handoff.
"""

from __future__ import annotations

import pytest

from perfxpert.tools._class import ToolClass


# ---------------------------------------------------------------------------
# Expected wire tool names (keys in the registry, in dot notation)
# ---------------------------------------------------------------------------


_EXPECTED_AGENT_TOOLS = {
    "agents.root.agent_root",
    "agents.analysis.agent_analysis",
    "agents.recommendation.agent_recommendation",
    "agents.correctness.agent_correctness",
    # Specialist module filenames are shortened (``compute.py`` /
    # ``memory.py`` / ``latency.py``) so the MCP wire names fit inside
    # the 64-char cap Claude Code enforces; the function names keep the
    # ``_specialist`` suffix for readability.
    "agents.compute.agent_compute_specialist",
    "agents.memory.agent_memory_specialist",
    "agents.latency.agent_latency_specialist",
}


def test_registry_exposes_all_seven_agent_tools() -> None:
    from mcp_server._registry import discover_read_only_tools

    reg = discover_read_only_tools()
    missing = _EXPECTED_AGENT_TOOLS - set(reg.keys())
    assert not missing, (
        f"agent tools missing from MCP registry: {missing}; "
        f"registry has {sorted(reg.keys())}"
    )


def test_agent_tools_are_read_only_in_registry() -> None:
    """Sanity: every agent tool is READ_ONLY as registered — the
    MCP exposure guard refuses EXECUTION-class tools (§5.8)."""
    from mcp_server._registry import discover_read_only_tools

    reg = discover_read_only_tools()
    for key in _EXPECTED_AGENT_TOOLS:
        fn = reg.get(key)
        assert fn is not None, key
        assert (
            getattr(fn, "__tool_class__", None) is ToolClass.READ_ONLY
        ), f"{key} is not READ_ONLY"


def test_old_run_root_analysis_tool_is_gone() -> None:
    """The retired ``analyze_run.run_root_analysis`` key MUST NOT appear
    anywhere in the registry — its replacement is ``agents.root.agent_root``.
    """
    from mcp_server._registry import discover_read_only_tools

    reg = discover_read_only_tools()
    assert "analyze_run.run_root_analysis" not in reg, (
        "legacy analyze_run.run_root_analysis tool is still registered; "
        "the agents-as-MCP-tools refactor should have removed it"
    )


def test_total_tool_count_is_41() -> None:
    """After the agents-as-MCP-tools refactor the registry should hold
    34 non-agent tools plus 7 agent tools — 41 total."""
    from mcp_server._registry import discover_read_only_tools

    reg = discover_read_only_tools()
    assert len(reg) == 41, (
        f"expected 41 tools (34 non-agent + 7 agent); got {len(reg)}: "
        f"{sorted(reg.keys())}"
    )
