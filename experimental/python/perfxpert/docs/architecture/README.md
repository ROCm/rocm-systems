# Architecture docs index

Reference material for how PerfXpert's agentic runtime (v0.2.0+) is
put together. Start with [../architecture.md](../architecture.md) for
the system-level overview; the docs below zoom in on specific
sub-systems.

| Topic | Doc | Audience |
|-------|-----|----------|
| Agent tier map — Root / Analysis / Recommendation / Correctness / specialists, plus the fence-slice prompt pattern. Each agent is callable via MCP (`perfxpert_agent_*`) + Python API (`perfxpert.api.agent_*`) — 1:1 mirror. | [agent-hierarchy.md](agent-hierarchy.md) | Contributors adding or modifying agents; integrators reading the code path |
| 5-gate correctness cascade (middleware) — Compile, SOL, Bitwise, Regression, Test Anchors | [gate-cascade.md](gate-cascade.md) | Contributors touching correctness logic; reviewers validating anti-reward-hack invariants |
| `BackendAdapter` Protocol — per-backend install / plan / spawn / verify_mcp_live / uninstall lifecycle for the multi-backend `perfxpert-code` launcher | [backend-adapter.md](backend-adapter.md) | Contributors adding a new backend (Claude / Gemini / Codex / opencode / future); integrators debugging the MCP + gate-hook install flow |

## See also

- [../architecture.md](../architecture.md) — top-level system diagram
  and design rationale.
- [../integration/mcp-server.md](../integration/mcp-server.md) — how
  the READ_ONLY tools used by the agents are re-exposed to external
  MCP clients.
- [../guides/agentic-mode.md](../guides/agentic-mode.md) — end-user
  view of the two runtime modes (air-gap vs LLM).
