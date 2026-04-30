# Contributing docs index

Per-surface contributor guides. Start at the repo-level
[CONTRIBUTING.md](../../CONTRIBUTING.md) for the master surface matrix
and governance; this directory holds the deep-dive guides referenced
from there.

## Layer-first map

PerfXpert is layered. Start from the layer you are changing, then use the
surface guide in the right column.

```
docs/contributing/
├── entry surfaces
│   └── providers.md, external-tools.md
├── agent brain
│   └── agents.md, schemas.md
├── deterministic tools and MCP exposure
│   └── tools.md, mcp_tools.md, gpu_arch.md
├── knowledge and optimization evidence
│   └── knowledge.md, proven_optimizations.md
├── report contracts
│   └── output_formats.md, schemas.md
├── fixtures and validation
│   └── fixtures.md, walkthrough_new_bottleneck_class.md
└── process
    └── governance.md
```

| Architecture layer | Typical change | Start here |
|--------------------|----------------|------------|
| Entry surfaces | CLI/backend/provider behavior, required binaries, pylibs, shared libs | [providers.md](providers.md), [external-tools.md](external-tools.md) |
| Agent brain | Root / Analysis / Recommendation / Correctness / specialist behavior | [agents.md](agents.md), [schemas.md](schemas.md) |
| Deterministic tool layer | Classifiers, SQL readers, hardware lookups, pure analysis helpers | [tools.md](tools.md) |
| MCP exposure | Make a READ_ONLY tool available to external MCP clients | [mcp_tools.md](mcp_tools.md) |
| Hardware facts | Add or update GPU architecture metadata | [gpu_arch.md](gpu_arch.md) |
| Knowledge layer | Add bottleneck signatures, thresholds, facts, or provenance | [knowledge.md](knowledge.md) |
| Proven optimizations | Add measured optimization evidence and impact ranges | [proven_optimizations.md](proven_optimizations.md) |
| Report/API contracts | Change text/json/markdown/webview output or schema fields | [output_formats.md](output_formats.md), [schemas.md](schemas.md) |
| Fixtures/tests | Add trace fixtures or validate a new scenario | [fixtures.md](fixtures.md), [walkthrough_new_bottleneck_class.md](walkthrough_new_bottleneck_class.md) |
| Governance | Decide whether the change needs an RFC or extra reviewers | [governance.md](governance.md) |

For the runtime layer diagram, read
[../architecture/agent-hierarchy.md](../architecture/agent-hierarchy.md)
before editing agent or specialist behavior.

## Alphabetical surface index

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
