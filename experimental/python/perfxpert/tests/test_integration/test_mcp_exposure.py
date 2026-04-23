"""CI guardrail: no execution-class tool registered with the MCP server (spec §5.8).

The MCP server lives in `mcp_server/server.py` (Phase 4). This test scaffolds
the assertion now; full MCP wiring lands in Phase 4 and this test then runs
against the live registry.
"""

import pytest

try:
    from mcp_server.server import MCP_TOOL_REGISTRY  # type: ignore
    _MCP_PRESENT = True
except ImportError:
    _MCP_PRESENT = False


EXECUTION_TOOLS = frozenset({
    "patch.apply", "patch.revert", "patch.verify_output",
    "compile.build", "profile.run", "anchors.check",
})


@pytest.mark.skipif(not _MCP_PRESENT, reason="MCP server not yet present (Phase 4)")
def test_mcp_registry_has_no_execution_tools():
    exposed = set(MCP_TOOL_REGISTRY.keys())
    violators = exposed & EXECUTION_TOOLS
    assert not violators, (
        f"MCP server exposes execution-class tool(s): {violators}. "
        f"Execution tools must remain in-process only (spec §5.8)."
    )
