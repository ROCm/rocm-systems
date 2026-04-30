# Architecture: MCP Boundary

This layer corresponds to the `56 READ_ONLY MCP tools` node in the main README
architecture diagram. It explains the boundary between external MCP clients and
the shared PerfXpert brain.

## Responsibilities

- Expose `perfxpert-mcp` as a standalone stdio MCP server for external clients.
- Keep MCP tools side-effect free: they analyze traces, look up deterministic
  facts, classify bottlenecks, route to agents, or compare runs.
- Route agent MCP tools into the same shared runtime used by `perfxpert analyze`,
  `perfxpert-code`, and `perfxpert.api`.
- Keep compile, edit, profiling, and filesystem mutation outside the MCP tool
  contract; those belong to the calling backend or user workflow.

## Guides

| Topic | Doc |
|-------|-----|
| Running `perfxpert-mcp` directly from an MCP client | [../../integration/mcp-server.md](../../integration/mcp-server.md) |
| Contributor rules for MCP tool definitions | [../../contributing/perfxpert-mcp/mcp_tools.md](../../contributing/perfxpert-mcp/mcp_tools.md) |
| Product-surface contribution map for `perfxpert-mcp` | [../../contributing/perfxpert-mcp/](../../contributing/perfxpert-mcp/) |

## Related Docs

- [../entry-surfaces/](../entry-surfaces/) — `perfxpert-mcp` is one of the
  external entry surfaces.
- [../agent-runtime/](../agent-runtime/) — agent MCP tools enter the same shared
  runtime/session as the CLI, Python API, and TUI backends.
- [../deterministic-foundation/](../deterministic-foundation/) — non-agent MCP
  tools expose deterministic classifiers, counter metadata, hardware facts, and
  trace comparison helpers.
