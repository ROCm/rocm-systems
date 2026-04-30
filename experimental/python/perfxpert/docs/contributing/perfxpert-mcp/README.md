# Contributing: PerfXpert MCP

This folder covers the standalone `perfxpert-mcp` server and the rules
for exposing read-only tools to external MCP clients.

## Layers

| Layer | Guide |
|-------|-------|
| Add or expose a READ_ONLY MCP tool | [mcp_tools.md](mcp_tools.md) |
| Implement the underlying deterministic helper | [../agent-brain/tools.md](../agent-brain/tools.md) |
| Evolve schemas returned through MCP | [../agent-brain/schemas.md](../agent-brain/schemas.md) |

## Related Runtime Docs

- [../../integration/mcp-server.md](../../integration/mcp-server.md) —
  standalone server, client snippets, and tool inventory.
- [../../guides/python-api.md](../../guides/python-api.md) — in-process
  mirror for the agent entry points.
