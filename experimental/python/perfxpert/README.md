# PerfXpert

AI-powered AMD ROCm GPU trace analysis.

## Quickstart

```bash
pip install perfxpert

# One-shot analysis (batch mode)
perfxpert analyze -i trace.db --llm anthropic --format webview -o report.html

# Interactive agentic TUI (AMD-themed opencode, replaces the old --interactive flag)
perfxpert-code

# Air-gap mode (no LLM; deterministic rule-based analysis only)
perfxpert analyze -i trace.db --format markdown -o report.md

# Health check
perfxpert doctor
```

## Architecture (v0.2.0+)

```
┌────────────────────────────────────────────────────────────┐
│  User shell                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ perfxpert   │  │ perfxpert-   │  │ Library API       │  │
│  │ analyze     │  │ code (TUI)   │  │ (Python)          │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬──────────┘  │
│         │                │                    │             │
│         └────────────────┴────────────────────┘             │
│                          │                                   │
│                          ▼                                   │
│             OpenAI Agents SDK hierarchy                      │
│   (Root → Analysis → Recommendation → Specialists)           │
│                          │                                   │
│                          ▼                                   │
│    Deterministic middleware (gate_cascade, intent router)    │
│                          │                                   │
│                          ▼                                   │
│            ~45 pure-Python tools + ~22 knowledge YAMLs       │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

Core analysis is self-contained — `pip install perfxpert` is sufficient for all
profiling and recommendation features. `perfxpert-code` (`perfxpert-code`
sub-command) requires **opencode** as a system dependency (bundling into the
wheel is a Phase 8 deliverable). Install opencode separately:

```bash
curl -fsSL https://opencode.ai/install | bash
```

Or point `PERFXPERT_OPENCODE_PATH` at an existing opencode binary.

## Contributing

perfxpert welcomes contributions. Start with [CONTRIBUTING.md](CONTRIBUTING.md)
for the extension-surface matrix + governance. Per-surface guides under
[docs/contributing/](docs/contributing/). Architectural changes go through
an [RFC](docs/rfcs/README.md).

## Feature flags

| Flag | Default | Purpose |
|------|---------|---------|
| `PERFXPERT_USE_AGENTS=1` | noop (kept for back-compat) | Was the Phase-4 opt-in; agentic is now default |
| `PERFXPERT_LEGACY=1` | unset | One-version safety net — routes through the pre-v0.2.0 path. Removed in vX.Y+1. |
| `ROCINSIGHT_LLM_REFERENCE_GUIDE` | unset | Legacy-only — path to user-supplied monolithic guide |
| `PERFXPERT_OPENCODE_PATH` | system PATH | Override path to opencode binary; must point to an existing file or launcher raises immediately |

## Supported GPUs

| Arch | GPU | CDNA/RDNA |
|------|-----|-----------|
| gfx908 | MI100 | CDNA1 |
| gfx90a | MI210/MI250/MI250X | CDNA2 |
| gfx942 | MI300A/MI300X/MI325X | CDNA3 |
| gfx950 | MI350X/MI355X | CDNA4 |
| gfx1030 | RX 6900 XT | RDNA2 |
| gfx1100 | RX 7900 XTX | RDNA3 |

## Output formats

`--format` accepts: `text` (default), `json`, `markdown`, `webview` (AMD-themed HTML).

## Documentation

- [Python API](perfxpert/ai_analysis/docs/AI_ANALYSIS_API.md)
- [Analysis output JSON schema](perfxpert/ai_analysis/docs/analysis-output.schema.json)
- [Migration guide from legacy path](docs/migration-to-agentic.md)
- [`PERFXPERT_LEGACY` deprecation note](docs/deprecation/PERFXPERT_LEGACY.md)
- [Contributing](CONTRIBUTING.md)

## Licensing

MIT. opencode (system dependency, not bundled in this release) is also MIT.
