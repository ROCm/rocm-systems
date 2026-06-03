# Contributing: new MCP tool

## What you're adding

A deterministic PerfXpert tool that is safe for external clients
(Claude Desktop, Cursor, Codex, Gemini, opencode, etc.) to call through
the Model Context Protocol. The MCP server does not have hand-written
per-tool wrappers; it auto-discovers READ_ONLY callables at startup.

Source of truth:

- Discovery: `mcp_server/_registry.py::discover_read_only_tools`
- Server: `mcp_server/server.py`
- Tool class marker: `perfxpert/tools/_class.py`

## File locations

- Implementation: `perfxpert/tools/<module>.py`
- Tests: `tests/test_tools/test_<module>.py`
- MCP exposure policy: `tests/test_integration/test_mcp_exposure.py`
- Public inventory docs: `docs/integration/mcp-server.md`

There is no `mcp_server/tools/<name>.py` wrapper directory in the current
implementation. Adding a public MCP tool means adding or updating a
`perfxpert.tools.*` function and marking it READ_ONLY.

## Key constraint

**Only `READ_ONLY` tools are exposed via MCP.** `EXECUTION` tools must
remain in-process only. CI enforces this via
`tests/test_integration/test_mcp_exposure.py`.

The registry walks `perfxpert.tools.*` recursively, skips private /
internal modules listed in `_SKIP_MODULES`, and registers public
functions whose `__tool_class__` is `ToolClass.READ_ONLY`. If a function
sets `__tool_name__`, that value becomes the MCP registry key;
otherwise the key is `<module_path>.<function>`.

## Template

```python
# SKIP-SAMPLE — template: <module> and fields are placeholders
"""<module> — <one-line purpose>.

Tool class: READ_ONLY (MCP-safe).
"""

from typing import Any, Dict

from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def my_tool(arg: str) -> Dict[str, Any]:
    """Return deterministic analysis for arg.

    Args:
        arg: Input identifier to analyze.

    Returns:
        JSON-serializable analysis dictionary.
    """
    return {"arg": arg, "status": "ok"}
```

## Registration

No manual registration step is required. On every `perfxpert-mcp` boot,
`discover_read_only_tools()` imports `perfxpert.tools.*`, finds public
READ_ONLY functions, and exposes them on the MCP wire with dots replaced
by underscores. For example:

```text
gpu_discovery.discover_runtime_gpu_specs
```

appears to MCP clients as:

```text
gpu_discovery_discover_runtime_gpu_specs
```

Some clients additionally prefix the server name in their UI or tool-call
syntax (for example `mcp__perfxpert__...`).

## Tests you must add

- `test_<module>_returns_expected_shape()` — happy path
- `test_<module>_handles_bad_input()` — error path
- `test_<module>_is_read_only_class()` — MCP exposure policy
- Update / rely on `test_mcp_exposure.py` so no EXECUTION tools leak
- If the public inventory changed, update
  `docs/integration/mcp-server.md`

## Review requirements

- 1 security-focused reviewer
- Exposure test green (no EXECUTION tools)
- CI green (unit + MCP policy)

## Common pitfalls

- Do not wrap EXECUTION tools. Profiling, patching, compiling, and test
  anchors stay in-process.
- Return JSON-serializable data. The MCP server serializes the result
  into text content.
- Do not add side effects to a READ_ONLY tool; SQL reads and knowledge
  YAML loads are fine, filesystem writes and live profiling are not.
- Keep the docs snapshot in `docs/integration/mcp-server.md` current if
  the tool count or public inventory changes.

## Related docs

- [MCP server](../integration/mcp-server.md)
- [New tool guide](tools.md)
- MCP spec: https://modelcontextprotocol.io/
- Exposure test: `tests/test_integration/test_mcp_exposure.py`
