"""The MCP registry exposes the 8 agent tools as READ_ONLY.

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
    "agent_root",
    "agent_analysis",
    "agent_recommendation",
    "agent_correctness",
    "agent_compute_specialist",
    "agent_memory_specialist",
    "agent_latency_specialist",
    "agent_diff_specialist",
}


def test_registry_exposes_all_eight_agent_tools() -> None:
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


def test_total_tool_count_is_56() -> None:
    """After the Phase-10 advanced-specialist work PLUS the RCCL / NIC
    communication-analysis additions PLUS the ``arch.lookup_peaks``
    dedupe (the ``sol.lookup_peaks`` + ``roofline.lookup_peaks`` aliases
    that resolved to the same callable were dropped in favor of the
    canonical ``arch.lookup_peaks`` name), the registry holds 48
    non-agent tools plus 8 agent tools -- 56 total.

    Phase-10 advanced specialists (+9 over the prior baseline):
      +1 kernel_fusion.find_fusion_candidates
      +3 gpu_runtime_monitor.{parse_amd_smi_json, parse_rocm_smi_json,
                              analyze_thermal}
      +1 unified_memory.analyze_paging
      +1 dependency_graph.reconstruct_dag
      +3 predict_impact.{predict_change_impact, list_supported_changes,
                          explain_prediction}

    RCCL / NIC communication analysis (+2 on top of the above):
      +1 rccl_analysis.analyze_collectives
      +1 interconnect.lookup_peaks

    MCP-surface dedupe (-2 after the above):
      -1 sol.lookup_peaks      (was an alias of arch.lookup_peaks)
      -1 roofline.lookup_peaks (was an alias of arch.lookup_peaks)
    """
    from mcp_server._registry import discover_read_only_tools

    reg = discover_read_only_tools()
    assert len(reg) == 56, (
        f"expected 56 tools (48 non-agent + 8 agent); got {len(reg)}: "
        f"{sorted(reg.keys())}"
    )
    # arch.lookup_peaks is the single canonical name; the sol / roofline
    # aliases were dropped in the Phase-10 polish sweep.
    assert "arch.lookup_peaks" in reg
    assert "sol.lookup_peaks" not in reg, (
        "sol.lookup_peaks alias was meant to be removed in favor of "
        "arch.lookup_peaks — underscore-prefixed module-level import."
    )
    assert "roofline.lookup_peaks" not in reg, (
        "roofline.lookup_peaks alias was meant to be removed in favor "
        "of arch.lookup_peaks — underscore-prefixed module-level import."
    )
    # trace_diff lives alongside regression as a READ_ONLY tool.
    assert "trace_diff.diff_runs" in reg
    # The 8th agent tool, exposed under the overridden ``__tool_name__``.
    assert "agent_diff_specialist" in reg
    # Phase-10 advanced-specialist additions must all be present.
    for k in (
        "kernel_fusion.find_fusion_candidates",
        "gpu_runtime_monitor.parse_amd_smi_json",
        "gpu_runtime_monitor.parse_rocm_smi_json",
        "gpu_runtime_monitor.analyze_thermal",
        "unified_memory.analyze_paging",
        "dependency_graph.reconstruct_dag",
        "predict_impact.predict_change_impact",
        "predict_impact.list_supported_changes",
        "predict_impact.explain_prediction",
    ):
        assert k in reg, f"phase-10 tool {k} missing from registry"
    # RCCL / NIC communication-analysis additions.
    for k in (
        "rccl_analysis.analyze_collectives",
        "interconnect.lookup_peaks",
    ):
        assert k in reg, f"phase-10 comm tool {k} missing from registry"
    # Specialist catalog loaders are internal helpers for the layered agent
    # runtime, not standalone MCP tools.
    for k in (
        "compute_techniques.catalog",
        "memory_techniques.catalog",
        "latency_techniques.catalog",
    ):
        assert k not in reg, f"internal helper {k} leaked into MCP registry"
