# Integration docs index

How to connect external tools and clients to PerfXpert without
importing the Python package directly.

| Topic | Doc | Audience |
|-------|-----|----------|
| MCP server (`perfxpert-mcp`) — 34 READ_ONLY tools re-exposed to any MCP-compatible client (opencode, Claude Desktop, Cursor, …) over stdio. Also documents the dot→underscore tool-name mangling. | [mcp-server.md](mcp-server.md) | External MCP client authors; integrators that need read-only access to PerfXpert's knowledge and classifiers |

## See also

- [../architecture/agent-hierarchy.md](../architecture/agent-hierarchy.md)
  — the agents that call the same tools in-process.
- [../guides/agentic-mode.md](../guides/agentic-mode.md) — note that
  MCP clients are always air-gap-safe because only READ_ONLY tools
  are exposed.
- [../contributing/mcp_tools.md](../contributing/mcp_tools.md) — how
  to add a new tool to the MCP surface.
