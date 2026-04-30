# Contributing: PerfXpert Batch CLI

This folder covers the user-facing `perfxpert` command: analysis reports,
provider selection, output contracts, and dependency checks used by the
batch CLI.

## Layers

| Layer | Guide |
|-------|-------|
| Provider registry and `--llm` behavior | [providers.md](providers.md) |
| Report formats and rendered output contracts | [output_formats.md](output_formats.md) |
| External dependency registration and validation | [external-tools.md](external-tools.md) |

## Related Runtime Docs

- [../../guides/getting-started.md](../../guides/getting-started.md) —
  user-facing CLI workflow.
- [../../guides/agentic-mode.md](../../guides/agentic-mode.md) —
  deterministic vs LLM-enabled analysis.
