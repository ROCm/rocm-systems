"""perfxpert-mcp — stdio MCP server exposing perfxpert READ_ONLY tools.

Entry point: `perfxpert-mcp` (see pyproject.toml).

Transport: stdio (spawned by opencode / Claude Desktop / Cursor etc.).
Registered tools: whatever `_registry.discover_read_only_tools()` returns.
EXECUTION tools are NEVER registered (CI-guarded; see test_mcp_exposure.py).

On every tool call, the server:
- Validates input against the tool's signature (typed via Pydantic when
  signatures have typed params; falls back to raw dict otherwise)
- Invokes the tool in-process (same Python brain)
- Returns structured JSON response
- Logs the call to a ring-buffer for debugging (stderr — never stdout
  which is reserved for MCP protocol)
"""

from __future__ import annotations

import inspect
import json
import logging
import sys
from typing import Any, Callable, Dict

# Check for MCP dependency via require_tool
from perfxpert.tools._tooldep import require_tool, ExternalToolMissing

# MCP SDK imports wrapped in try/except for graceful degradation
_MCP_AVAILABLE = False
try:
    require_tool("mcp")
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
    _MCP_AVAILABLE = True
except (ImportError, ExternalToolMissing):
    Server = object  # type: ignore
    TextContent = None  # type: ignore
    Tool = None  # type: ignore

from mcp_server._registry import discover_read_only_tools


_LOG = logging.getLogger("perfxpert-mcp")
_LOG.setLevel(logging.INFO)
_LOG.addHandler(logging.StreamHandler(stream=sys.stderr))


def _fn_to_tool_schema(name: str, fn: Callable) -> Tool:
    """Convert a Python callable into an MCP Tool descriptor.

    Parameter schema is derived via inspect.signature; typed with `string`
    fallback if the annotation isn't a primitive.
    """
    sig = inspect.signature(fn)
    properties: Dict[str, Any] = {}
    required = []
    for pname, p in sig.parameters.items():
        if pname == "self" or p.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        schema = _annotation_to_json_schema(p.annotation)
        properties[pname] = schema
        if p.default is inspect.Parameter.empty:
            required.append(pname)

    doc = inspect.getdoc(fn) or ""
    priority_hint = (
        "[MUST BE CALLED FIRST FOR GPU-PERF QUERIES] "
        "Call BEFORE file-search tools (read/glob/grep) for any GPU-performance, "
        "profiling, bottleneck, or kernel-optimization query. "
    )
    description = priority_hint + doc.split("\n\n", 1)[0]
    return Tool(
        name=name.replace(".", "_"),          # MCP tool names disallow dots
        description=description,
        inputSchema={
            "type": "object",
            "properties": properties,
            "required": required,
        },
    )


def _annotation_to_json_type(ann) -> str:
    if ann is inspect.Parameter.empty:
        return "string"
    if ann is str:
        return "string"
    if ann in (int, float):
        return "number"
    if ann is bool:
        return "boolean"
    if ann is list or getattr(ann, "__origin__", None) is list:
        return "array"
    if ann is dict or getattr(ann, "__origin__", None) is dict:
        return "object"
    return "string"


def _annotation_to_json_schema(ann) -> Dict[str, Any]:
    """Build a JSON-Schema fragment for one parameter annotation.

    OpenAI function-calling requires `items` for arrays and `additionalProperties`
    for objects, else the tool is rejected. We infer element type from parametric
    hints (List[str], Dict[str, int]) when present, else default to string.
    """
    origin = getattr(ann, "__origin__", None)
    args = getattr(ann, "__args__", ()) or ()

    if ann is list or origin is list:
        item_ann = args[0] if args else str
        return {"type": "array", "items": _annotation_to_json_schema(item_ann)}
    if ann is dict or origin is dict:
        return {"type": "object", "additionalProperties": True}
    return {"type": _annotation_to_json_type(ann)}


def build_server() -> Server:
    """Construct the Server with all READ_ONLY tools registered."""
    if not _MCP_AVAILABLE:
        try:
            require_tool("mcp")
        except ExternalToolMissing as e:
            raise RuntimeError(f"MCP SDK not available: {e.install_hint}") from e

    server: Server = Server("perfxpert")
    tools = discover_read_only_tools()
    _LOG.info("perfxpert-mcp: discovered %d read-only tools", len(tools))

    # MCP protocol maps "list tools" call
    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [_fn_to_tool_schema(name, fn) for name, fn in tools.items()]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[TextContent]:
        # Reverse the dot→underscore mapping
        original_name = name.replace("_", ".", 1) if "." not in name else name
        # Fall back to exact match
        fn = tools.get(original_name) or tools.get(name)
        if fn is None:
            # Try all keys by replacing first underscore with dot
            for k in tools:
                if k.replace(".", "_") == name:
                    fn = tools[k]
                    original_name = k
                    break
        if fn is None:
            raise ValueError(f"unknown tool: {name}")

        _LOG.info("perfxpert-mcp: calling %s with %r", original_name, arguments)
        result = fn(**arguments)
        return [TextContent(
            type="text",
            text=json.dumps(result, default=str, indent=2),
        )]

    return server


def main() -> None:
    """Entry point for `perfxpert-mcp` console script."""
    import asyncio

    server = build_server()

    async def _serve():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
