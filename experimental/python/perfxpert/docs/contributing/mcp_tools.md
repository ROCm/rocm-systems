# Contributing: new MCP tool

## What you're adding

A wrapper that exposes a perfxpert tool to external clients (Claude Desktop,
Cursor, etc.) via the Model Context Protocol. MCP tools must preserve the
`READ_ONLY` class invariant — no `EXECUTION` tools are exposed.

## File locations

- MCP server: `mcp_server/`
- Tool wrapper: `mcp_server/tools/<name>.py`
- Tests: `tests/test_mcp/test_<name>.py`

## Key constraint

**Only `READ_ONLY` tools are exposed via MCP.** `EXECUTION` tools must remain
in-process only. CI enforces this via `tests/test_integration/test_mcp_exposure.py`.

## Template

```python
# SKIP-SAMPLE — template: <name> is a placeholder
"""<name> MCP tool wrapper."""

from typing import Any, Dict

import mcp.types as types
from mcp.server import RequestContext

from perfxpert.tools import my_read_only_tool


async def handle_name(
    context: RequestContext,
    **kwargs: Any
) -> list[types.TextContent | types.ImageContent]:
    """MCP handler for <name>."""
    try:
        result = my_read_only_tool(**kwargs)
        return [types.TextContent(type="text", text=str(result))]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {e}")]
```

## Registration

Add to MCP server tool list:

```python
# SKIP-SAMPLE — template: `server` and `handle_tool_call` are placeholders
server.set_request_handler(
    types.messages.CallToolRequest,
    lambda ctx, req: handle_tool_call(ctx, req)
)
```

## Schema constraints (CI-enforced)

- Wrapping tool must have `ToolClass.READ_ONLY` set
- MCP exposure test (`test_mcp_exposure.py`) verifies no `EXECUTION` tools leak
- Type hints on wrapper args + return

## Tests you must add

- `test_<name>_mcp_wraps_read_only_tool()` — class invariant
- `test_<name>_returns_valid_mcp_content()` — output shape
- `test_<name>_handles_error()` — error path

## Review requirements

- 1 security-focused reviewer
- Exposure test green (no EXECUTION tools)
- CI green (unit + MCP policy)

## Common pitfalls

- Don't wrap EXECUTION tools — they must stay in-process
- MCP responses are text + images only — complex types must serialize to JSON
- Error messages should not leak internal details

## Related docs

- Design spec: Appendix A (tool-class split)
- MCP spec: https://modelcontextprotocol.io/
- Exposure test: `tests/test_integration/test_mcp_exposure.py`
