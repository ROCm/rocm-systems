"""MCP tool descriptions lead with the perfxpert-first priority hint.

See issue 1 in docs/superpowers/plans/2026-04-18-perfxpert-phase8-pr2-user-issues.md:
opencode-hosted LLMs ignored perfxpert MCP tools in favor of read/glob/grep for
GPU-performance queries. Reinforcing tool priority at the schema layer is a
belt-and-suspenders counterpart to the default-prompt patch.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")


def test_every_tool_description_leads_with_priority_hint() -> None:
    from mcp_server.server import _fn_to_tool_schema  # type: ignore[import-not-found]
    from mcp_server._registry import discover_read_only_tools

    tools = discover_read_only_tools()
    assert tools, "no READ_ONLY tools discovered"
    for name, fn in tools.items():
        schema = _fn_to_tool_schema(name, fn)
        desc = schema.description  # type: ignore[attr-defined]
        # Cycle-4 B1: the leading imperative is an ALL-CAPS bracketed tag
        # which surfaces the priority rule at the very start of the tool
        # listing — LLMs respect this form more reliably than a plain
        # sentence (observed during live-user validation).
        assert desc.startswith(
            "[MUST BE CALLED FIRST FOR GPU-PERF QUERIES]"
        ), f"{name!r} description does not lead with priority bracket: {desc[:80]!r}"
        assert (
            "call before file-search tools" in desc.lower()
        ), f"{name!r} description lost the file-search-tools hint: {desc[:120]!r}"
