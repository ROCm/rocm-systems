# Contributing Docs Index

Start with the product surface or architecture layer you are changing,
then drill down. Start at the repo-level [CONTRIBUTING.md](../../CONTRIBUTING.md)
for project-wide process and review expectations.

## Folder Map

```
docs/contributing/
├── perfxpert/          batch CLI, report output, providers, dependency checks
├── perfxpert-code/     interactive launcher and backend dispatch surface
├── perfxpert-mcp/      standalone MCP server exposure rules
├── perfxpert-diff/     diff, CI comparison, and baseline-regression surface
├── agent-brain/        agents, deterministic tools, knowledge, schemas, fixtures
└── process/            governance, escalation, RFC boundaries
```

## Start Here

| You are changing... | Go to |
|---------------------|-------|
| `perfxpert analyze`, output files, report formats, LLM provider setup, or required external tools | [perfxpert/](perfxpert/) |
| `perfxpert-code`, backend dispatch, native TUI setup, or launcher behavior | [perfxpert-code/](perfxpert-code/) |
| `perfxpert-mcp`, MCP tool registration, or external MCP client exposure | [perfxpert-mcp/](perfxpert-mcp/) |
| `perfxpert diff`, `perfxpert ci`, `--baseline`, or run-to-run comparison contracts | [perfxpert-diff/](perfxpert-diff/) |
| Agent routing, specialists, deterministic tools, knowledge YAMLs, schemas, GPU facts, fixtures, or new bottleneck classes | [agent-brain/](agent-brain/) |
| Review policy, RFC requirements, or escalation rules | [process/](process/) |

## Architecture Cross-Links

- [../architecture/agent-runtime/agent-hierarchy.md](../architecture/agent-runtime/agent-hierarchy.md) —
  Root, Layer-1 agents, specialists, and source locations.
- [../architecture/correctness/gate-cascade.md](../architecture/correctness/gate-cascade.md) —
  correctness gates and revert flow.
- [../integration/mcp-server.md](../integration/mcp-server.md) —
  standalone MCP server behavior and client integration.
