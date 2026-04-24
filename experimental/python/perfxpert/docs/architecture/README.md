# Architecture docs index

Reference material for how PerfXpert's agentic runtime (v0.2.0+) is
put together. Start with [../architecture.md](../architecture.md) for
the system-level overview; the docs below zoom in on specific
sub-systems.

| Topic | Doc | Audience |
|-------|-----|----------|
| Agent tier map — Root / Analysis / Recommendation / Correctness / specialists, plus the fence-slice prompt pattern | [agent-hierarchy.md](agent-hierarchy.md) | Contributors adding or modifying agents; integrators reading the code path |
| 5-gate correctness cascade (middleware) — Compile, SOL, Bitwise, Regression, Test Anchors | [gate-cascade.md](gate-cascade.md) | Contributors touching correctness logic; reviewers validating anti-reward-hack invariants |
| Multi-backend launcher contract — shared install/verify/spawn lifecycle for opencode, Claude Code, Gemini CLI, and Codex CLI | [backend-adapter.md](backend-adapter.md) | Contributors extending `perfxpert-code` or validating backend-specific behavior |

## See also

- [../architecture.md](../architecture.md) — top-level system diagram
  and design rationale.
- [../integration/mcp-server.md](../integration/mcp-server.md) — how
  the READ_ONLY tools used by the agents are re-exposed to external
  MCP clients.
- [../guides/agentic-mode.md](../guides/agentic-mode.md) — end-user
  view of the two runtime modes (air-gap vs LLM).
