"""CI guard — no EXECUTION-class tool is in the MCP registry.

This test failing is a security-critical incident. If you're reading this
message because you moved an EXECUTION tool into the MCP-exposed list,
STOP and open an architectural RFC per spec §5.8.
"""

import pytest

from perfxpert.tools._class import ToolClass
from mcp_server._registry import discover_read_only_tools


def test_no_execution_tools_in_registry():
    registered = discover_read_only_tools()
    for name, fn in registered.items():
        assert getattr(fn, "__tool_class__", None) == ToolClass.READ_ONLY, (
            f"MCP registry contains non-READ_ONLY tool: {name}"
        )


def test_registry_discovers_expected_tools():
    """Sanity: at least 30 read-only tools are discovered (baseline floor ~25)."""
    registered = discover_read_only_tools()
    assert len(registered) >= 30, (
        f"registry has only {len(registered)} tools — did tool discovery break?"
    )


def test_known_read_only_tools_present():
    """Canary: these specific tools MUST be exposed."""
    registered = discover_read_only_tools()
    required = {
        "arch.lookup_peaks",
        "bottleneck.classify_from_metrics",
        "roofline.classify",
        "sol.sanity_check",
        "regression.compare_runs",
        "trace_fingerprint.fingerprint",
        "plateau.check",
        "intent.classify",
    }
    missing = required - set(registered.keys())
    assert not missing, f"expected tools not registered: {missing}"


def test_known_execution_tools_absent():
    """Canary: these EXECUTION tools MUST NOT be exposed."""
    registered = discover_read_only_tools()
    forbidden = {
        "patch_mgr.apply",
        "patch_mgr.revert",
        "patch_mgr.verify_output",
        "compile_runner.build",
        "profile_runner.run",
        "anchors.check",
    }
    leaked = forbidden & set(registered.keys())
    assert not leaked, (
        f"CRITICAL: EXECUTION tools leaked into MCP registry: {leaked}. "
        "This is a §5.8 violation. Revert immediately."
    )
