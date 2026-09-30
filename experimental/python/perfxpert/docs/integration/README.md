# Integration docs index

How to connect external tools and clients to PerfXpert without
importing the Python package directly.

| Topic | Doc | Audience |
|-------|-----|----------|
| MCP server (`perfxpert-mcp`) — 56 READ_ONLY tools re-exposed to any MCP-compatible client (opencode, Claude Desktop, Cursor, …) over stdio. **8 of the 56 are agent-hierarchy tools** (`perfxpert_agent_root`, `perfxpert_agent_analysis`, `perfxpert_agent_recommendation`, `perfxpert_agent_correctness`, `perfxpert_agent_compute_specialist`, `perfxpert_agent_memory_specialist`, `perfxpert_agent_latency_specialist`, `perfxpert_agent_diff_specialist`) — 1:1 mirrored in `perfxpert.api`. The other 48 are classifier / knowledge lookups + `trace_diff.diff_runs`. Also documents the dot→underscore tool-name mangling. | [mcp-server.md](mcp-server.md) | External MCP client authors; integrators that need read-only access to PerfXpert's knowledge and classifiers |
| Python API (`perfxpert.api`) — 1:1 mirror of the 8 agent MCP tools for in-process embedding without an MCP server. Same schemas, same return shapes. | [../guides/python-api.md](../guides/python-api.md) | Library callers embedding PerfXpert's analysis brain in their own scripts / tools |

## See also

- [../architecture/agent-hierarchy.md](../architecture/agent-hierarchy.md)
  — the agents that call the same tools in-process.
- [../guides/agentic-mode.md](../guides/agentic-mode.md) — note that
  MCP clients are always air-gap-safe because only READ_ONLY tools
  are exposed.
- [../contributing/mcp_tools.md](../contributing/mcp_tools.md) — how
  to add a new tool to the MCP surface.
