# Architecture Docs Index

Reference material for how PerfXpert's agentic runtime is put together.
Start with [../architecture.md](../architecture.md) for the system-level
overview, then drill into the layer you need.

## Correlation to the README Diagram

The architecture diagram in [../../README.md](../../README.md#architecture-v020)
is the canonical top-level view. This directory mirrors that diagram by layer:

| README diagram node | Architecture docs layer | What it explains |
|---------------------|-------------------------|------------------|
| `Entry surfaces`: `perfxpert analyze`, `perfxpert-code`, `perfxpert-mcp`, `perfxpert.api` | [entry-surfaces/](entry-surfaces/) | How users, backend TUIs, external MCP clients, and Python embedding enter PerfXpert |
| `56 READ_ONLY MCP tools` | [mcp-boundary/](mcp-boundary/) | The side-effect-free MCP contract between external clients and the shared runtime |
| `Shared agent runtime/session` and `8-agent hierarchy` | [agent-runtime/](agent-runtime/) | The shared brain, routing model, and Root / Analysis / Recommendation / Correctness / specialist agents |
| `Deterministic tools + validated knowledge YAMLs` | [deterministic-foundation/](deterministic-foundation/) | Classifiers, counters, hardware facts, trace diff, schemas, fixtures, and curated knowledge |
| `Correctness middleware` | [correctness/](correctness/) | Gate cascade, validation, rollback, and anti-reward-hacking rules |

## Folder Map

```text
docs/architecture/
├── entry-surfaces/              CLI, perfxpert-code, backend adapters, API/MCP entry paths
├── mcp-boundary/                READ_ONLY external MCP contract
├── agent-runtime/               Root, Layer-1 agents, specialists, fence-slice pattern
├── deterministic-foundation/    classifiers, counters, hardware facts, schemas, knowledge
└── correctness/                 gate cascade, validation, rollback, anti-reward-hacking rules
```

## Start Here

| You need to understand... | Go to |
|---------------------------|-------|
| How users and backend TUIs enter the system | [entry-surfaces/](entry-surfaces/) |
| How external MCP clients reach the shared brain safely | [mcp-boundary/](mcp-boundary/) |
| How the agent brain is layered and routed | [agent-runtime/](agent-runtime/) |
| Where deterministic tools and curated knowledge live | [deterministic-foundation/](deterministic-foundation/) |
| How proposed changes are validated or rejected | [correctness/](correctness/) |

## See Also

- [../architecture.md](../architecture.md) — top-level system diagram and
  design rationale.
- [../integration/mcp-server.md](../integration/mcp-server.md) — how
  READ_ONLY tools are exposed to external MCP clients.
- [../guides/agentic-mode.md](../guides/agentic-mode.md) — end-user view
  of air-gap vs LLM runtime modes.
