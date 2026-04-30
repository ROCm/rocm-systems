# Architecture: Entry Surfaces

This layer explains how users and host backends enter PerfXpert before
control reaches the shared agent runtime.

## Guides

| Topic | Doc |
|-------|-----|
| Backend adapter lifecycle for `perfxpert-code claude|codex|gemini|opencode` | [backend-adapter.md](backend-adapter.md) |

## Related Docs

- [../../guides/backends.md](../../guides/backends.md) — user-facing
  backend setup and troubleshooting.
- [../../integration/mcp-server.md](../../integration/mcp-server.md) —
  standalone MCP server entry path.
- [../agent-runtime/](../agent-runtime/) — what happens after a surface
  enters the shared brain.
