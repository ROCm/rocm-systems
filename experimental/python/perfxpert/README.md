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

Everything ships inside one `pip install perfxpert` — including the bundled opencode binary for `perfxpert-code`.

## Feature flags

| Flag | Default | Purpose |
|------|---------|---------|
| `PERFXPERT_USE_AGENTS=1` | noop (kept for back-compat) | Was the Phase-4 opt-in; agentic is now default |
| `PERFXPERT_LEGACY=1` | unset | One-version safety net — routes through the deprecated local-only path. LLM enhancement is unavailable. Removed in vX.Y+1. |
| `ROCINSIGHT_LLM_REFERENCE_GUIDE` | unset | Legacy compatibility knob for direct reference-guide loading; not used by the default analyze path |
| `PERFXPERT_OPENCODE_PATH` | bundled | Override path to opencode binary |

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

MIT. Bundled opencode binary is also MIT (permissive redistribution).
