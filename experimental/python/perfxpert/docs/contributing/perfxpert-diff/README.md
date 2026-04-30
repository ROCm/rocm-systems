# Contributing: PerfXpert Diff

This folder covers run-to-run comparison surfaces: `perfxpert diff`,
`perfxpert ci`, `perfxpert analyze --baseline`, and the Diff Specialist.

## Layers

| Layer | Where to look |
|-------|---------------|
| Diff schema and payload fields | [../agent-brain/schemas.md](../agent-brain/schemas.md) |
| Diff specialist behavior | [../agent-brain/agents.md](../agent-brain/agents.md) |
| Deterministic regression/diff tools | [../agent-brain/tools.md](../agent-brain/tools.md) |
| Report rendering for diff sections | [../perfxpert/output_formats.md](../perfxpert/output_formats.md) |
| Fixtures for comparison scenarios | [../agent-brain/fixtures.md](../agent-brain/fixtures.md) |

## Related Runtime Docs

- [../../guides/python-api.md](../../guides/python-api.md) — Python
  examples for `trace_diff_diff_runs`.
- [../../architecture/agent-runtime/agent-hierarchy.md](../../architecture/agent-runtime/agent-hierarchy.md) —
  where the Diff Specialist fits in the brain.
