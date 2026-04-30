# Contributing: PerfXpert-Code

This folder covers the interactive launcher surface: `perfxpert-code`,
backend dispatch, backend-specific installation, and the native TUI path.

## Layers

| Layer | Where to look |
|-------|---------------|
| Backend dispatch and install flow | [../../guides/backends.md](../../guides/backends.md) |
| Interactive optimization workflow | [../../guides/getting-started.md §9](../../guides/getting-started.md#9-agentic-tui-workflow-the-star-feature) |
| Provider behavior used by scripted runs | [../perfxpert/providers.md](../perfxpert/providers.md) |
| Required binaries and dependency checks | [../perfxpert/external-tools.md](../perfxpert/external-tools.md) |
| Agent prompt and MCP behavior behind the launcher | [../agent-brain/agents.md](../agent-brain/agents.md), [../perfxpert-mcp/](../perfxpert-mcp/) |

## Rule of Thumb

If the change affects how users launch or wire a backend, start here. If
the change affects the analysis itself after the backend is running, move
to [../agent-brain/](../agent-brain/).
