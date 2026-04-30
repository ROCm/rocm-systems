# Contributing: new MCP tool

## What you're adding

A `READ_ONLY` PerfXpert tool that the MCP server can auto-discover and
expose to external clients (Claude Desktop, Cursor, Codex, Gemini,
opencode, etc.) via the Model Context Protocol. There is no separate
`mcp_server/tools/` wrapper directory in the current code path: the MCP
registry walks `perfxpert.tools.*` and registers public callables
annotated with `@tool_class(ToolClass.READ_ONLY)`.

## File locations

- Tool implementation: `perfxpert/tools/<module>.py`
- Tool classification: `@tool_class(ToolClass.READ_ONLY)` from
  `perfxpert.tools._class`
- MCP auto-discovery: `mcp_server/_registry.py`
- MCP protocol server: `mcp_server/server.py`
- Tests: `tests/test_tools/test_<module>.py` plus integration coverage
  in `tests/test_integration/test_mcp_exposure.py`

## Key constraint

**Only `READ_ONLY` tools are exposed via MCP.** `EXECUTION` tools must remain
in-process only. CI enforces this via `tests/test_integration/test_mcp_exposure.py`.

## Template

```python
# SKIP-SAMPLE — template: <name> is a placeholder
"""<name> — read-only analysis helper."""

from typing import Any, Dict

from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def lookup_name(metric: str) -> Dict[str, Any]:
    """Return read-only metadata for a metric."""
    return {"metric": metric, "description": "..."}
```

## Registration

No manual server registration is needed for ordinary tools. The MCP
server imports `mcp_server._registry.discover_read_only_tools()`, walks
`perfxpert.tools.*` recursively, skips private helpers and packages, and
exports each public `READ_ONLY` callable. Dots in the internal tool name
are converted to underscores on the MCP wire.

## Schema constraints (CI-enforced)

- Tool must have `ToolClass.READ_ONLY` set
- MCP exposure test (`test_mcp_exposure.py`) verifies no `EXECUTION` tools leak
- Type hints on tool args + return so the MCP schema has useful input
  types

## Tests you must add

- `test_<name>_returns_expected_shape()` — pure function output
- `test_<name>_is_read_only()` — class invariant
- Schema/registry coverage if the tool has unusual parameters or a
  custom exposed name

## Review requirements

- 1 security-focused reviewer
- Exposure test green (no EXECUTION tools)
- CI green (unit + MCP policy)

## Common pitfalls

- Don't mark EXECUTION tools as READ_ONLY — they must stay in-process
- Return JSON-serializable data; `mcp_server/server.py` serializes the
  result into MCP `TextContent`
- Error messages should not leak internal details

## Related docs

- `mcp_server/_registry.py` — auto-discovery details
- `mcp_server/server.py` — MCP schema and JSON response wrapping
- MCP spec: https://modelcontextprotocol.io/
- Exposure test: `tests/test_integration/test_mcp_exposure.py`
