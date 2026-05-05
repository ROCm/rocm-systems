# Architecture: Agent Runtime

This layer explains the shared PerfXpert brain used by the CLI, Python
API, MCP server, and `perfxpert-code` backends.

## Guides

| Topic | Doc |
|-------|-----|
| Root, Tier-1 agents, specialists, routing, and fence-slice pattern | [agent-hierarchy.md](agent-hierarchy.md) |

## Related Docs

- [../correctness/](../correctness/) — gate cascade used after proposed
  edits.
- [../../contributing/agent-brain/](../../contributing/agent-brain/) —
  contributor guides for agents, tools, knowledge, schemas, and fixtures.
- [../../guides/python-api.md](../../guides/python-api.md) — embedding
  the same agent entry points from Python.
