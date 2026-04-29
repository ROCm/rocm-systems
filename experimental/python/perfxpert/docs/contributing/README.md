# Contributing docs index

Per-surface contributor guides. Start at the repo-level
[CONTRIBUTING.md](../../CONTRIBUTING.md) for the master surface matrix
and governance; this directory holds the deep-dive guides referenced
from there.

## Extension surfaces

| Surface | Guide |
|---------|-------|
| New tool | [tools.md](tools.md) |
| New knowledge entry | [knowledge.md](knowledge.md) |
| New proven optimization | [proven_optimizations.md](proven_optimizations.md) |
| New agent | [agents.md](agents.md) |
| New LLM provider | [providers.md](providers.md) |
| New MCP tool | [mcp_tools.md](mcp_tools.md) |
| New test fixture | [fixtures.md](fixtures.md) |
| New GPU arch | [gpu_arch.md](gpu_arch.md) |
| External-tool dependency registration (`require_tool`, binaries / pylibs / shared libs) | [external-tools.md](external-tools.md) |

## Process and walkthroughs

| Topic | Doc |
|-------|-----|
| Governance — reviewers, escalation, RFC-required changes | [governance.md](governance.md) |
| End-to-end walkthrough: add a new bottleneck class in ≤ 7 files, no RFC | [walkthrough_new_bottleneck_class.md](walkthrough_new_bottleneck_class.md) |

## See also

- [../architecture/](../architecture/) — know the architecture before
  extending it; agent additions in particular require familiarity with
  [agent-hierarchy.md](../architecture/agent-hierarchy.md) and
  [gate-cascade.md](../architecture/gate-cascade.md).
- [../rfcs/README.md](../rfcs/README.md) — for architectural changes.
