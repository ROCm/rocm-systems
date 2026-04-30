# Architecture Docs Index

Reference material for how PerfXpert's agentic runtime is put together.
Start with [../architecture.md](../architecture.md) for the system-level
overview, then drill into the layer you need.

## Folder Map

```text
docs/architecture/
├── entry-surfaces/     CLI, perfxpert-code, backend adapters, API/MCP entry paths
├── agent-runtime/      Root, Layer-1 agents, specialists, fence-slice pattern
└── correctness/        gate cascade, validation, rollback, anti-reward-hacking rules
```

## Start Here

| You need to understand... | Go to |
|---------------------------|-------|
| How users and backend TUIs enter the system | [entry-surfaces/](entry-surfaces/) |
| How the agent brain is layered and routed | [agent-runtime/](agent-runtime/) |
| How proposed changes are validated or rejected | [correctness/](correctness/) |

## See Also

- [../architecture.md](../architecture.md) — top-level system diagram and
  design rationale.
- [../integration/mcp-server.md](../integration/mcp-server.md) — how
  READ_ONLY tools are exposed to external MCP clients.
- [../guides/agentic-mode.md](../guides/agentic-mode.md) — end-user view
  of air-gap vs LLM runtime modes.
