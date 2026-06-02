"""Phase-10 specialist allowlist tests.

Ensures that:
- Compute specialist binds ``kernel_fusion.find_fusion_candidates`` (Feature A).
- Memory specialist binds ``unified_memory.analyze_paging`` (Feature C).
- Latency specialist binds ``dependency_graph.reconstruct_dag`` (Feature D).
- All three stay within the 5-tool cap.
- ``gpu_runtime_monitor.*`` is NOT in the latency allowlist — it is
  available via the MCP surface only (Feature B policy).
"""

import pytest

from perfxpert.agents import (
    build_compute_specialist,
    build_memory_specialist,
    build_latency_specialist,
)


def _names(agent):
    return {t.name for t in agent.tools}


def test_compute_binds_kernel_fusion():
    agent = build_compute_specialist()
    assert "kernel_fusion.find_fusion_candidates" in _names(agent)
    assert len(agent.tools) <= 5


def test_memory_binds_unified_memory():
    agent = build_memory_specialist()
    assert "unified_memory.analyze_paging" in _names(agent)
    assert len(agent.tools) <= 5


def test_latency_binds_dependency_graph():
    agent = build_latency_specialist()
    assert "dependency_graph.reconstruct_dag" in _names(agent)
    assert len(agent.tools) <= 5


def test_latency_does_not_bind_gpu_runtime_monitor():
    """gpu_runtime_monitor.* lives on the MCP surface, not the cap-5 allowlist.

    Rationale: thermal envelope analysis is diagnostic / out-of-band.
    """
    agent = build_latency_specialist()
    names = _names(agent)
    forbidden_on_allowlist = {
        "gpu_runtime_monitor.parse_amd_smi_json",
        "gpu_runtime_monitor.parse_rocm_smi_json",
        "gpu_runtime_monitor.analyze_thermal",
    }
    assert not (names & forbidden_on_allowlist)


def test_all_phase10_tools_discoverable_via_mcp():
    """Sanity check — the new modules are visible to the MCP registry."""
    from mcp_server._registry import discover_read_only_tools
    tools = discover_read_only_tools()
    expected = {
        "kernel_fusion.find_fusion_candidates",
        "gpu_runtime_monitor.parse_amd_smi_json",
        "gpu_runtime_monitor.parse_rocm_smi_json",
        "gpu_runtime_monitor.analyze_thermal",
        "unified_memory.analyze_paging",
        "dependency_graph.reconstruct_dag",
    }
    missing = expected - set(tools.keys())
    assert not missing, f"missing from MCP registry: {missing}"
