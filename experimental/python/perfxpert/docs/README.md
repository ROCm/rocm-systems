# PerfXpert documentation

Top-level index for everything under `docs/`. The repo-level
[README.md](../README.md) is the user-facing entry point; this index is
the map for everything that follows.

## Start here

PerfXpert's user workflow is:

1. Capture a rocprofiler-sdk `.db` trace with `rocprofv3` on a ROCm
   machine.
2. Analyze the `.db` with `perfxpert analyze`. Existing `.db` traces can
   be analyzed on a machine without ROCm when the Python dependencies are
   installed.
3. Use `perfxpert-code` for interactive profile → analyze → edit →
   rebuild → reprofile loops.

| You want to… | Read |
|--------------|------|
| Install, capture a trace, and run your first analysis | [guides/getting-started.md](guides/getting-started.md) |
| Choose `text`, `json`, `markdown`, or `webview` output and control report files | [guides/getting-started.md §4](guides/getting-started.md#4-first-analysis-60-seconds) |
| Use an existing `.db` on a non-ROCm analysis host | [guides/getting-started.md §0](guides/getting-started.md#0-workflow-at-a-glance) |
| Understand deterministic air-gap vs LLM-enabled analysis | [guides/agentic-mode.md](guides/agentic-mode.md) |
| Drive the interactive TUI | [guides/getting-started.md §9](guides/getting-started.md#9-agentic-tui-workflow-the-star-feature) |
| Embed PerfXpert in your own tool | [guides/python-api.md](guides/python-api.md) |
| Drive PerfXpert from a backend (opencode / claude / codex / gemini) | [guides/backends.md](guides/backends.md) |

## By area

| Area | Index |
|------|-------|
| User guides (install, formats, providers, MCP) | [guides/README.md](guides/README.md) |
| Architecture (agent hierarchy, gate cascade, backend adapter) | [architecture/README.md](architecture/README.md) |
| Integration (MCP server, Python API embedding) | [integration/README.md](integration/README.md) |
| Contributing (extension surfaces, governance, audit gate) | [contributing/README.md](contributing/README.md) |
| RFCs (architectural change proposals) | [rfcs/README.md](rfcs/README.md) |
| Benchmarks (proven optimizations baselines) | [benchmarks/](benchmarks/) |
| Archive (pre-agentic migration notes) | [archive/](archive/) |

## File quick-reference

```
docs/
├── README.md                ← you are here
├── architecture.md          ← top-level system overview
├── architecture/
│   ├── README.md            ← architecture sub-index
│   ├── agent-hierarchy.md   ← Root → Analysis → Recommendation → specialists
│   ├── gate-cascade.md      ← correctness gate + revert path
│   └── backend-adapter.md   ← per-backend installer / verifier surface
├── guides/
│   ├── README.md            ← guides sub-index
│   ├── getting-started.md   ← install, run, formats, providers, troubleshooting
│   ├── agentic-mode.md      ← how the agent brain works at runtime
│   ├── python-api.md        ← `perfxpert.api.*` — 1:1 mirror of the agent MCP tools
│   ├── backends.md          ← perfxpert-code dispatch matrix
│   └── assets/gifs/         ← checked-in guide demo recordings
├── integration/
│   ├── README.md
│   └── mcp-server.md        ← the MCP wire surface (56 tools)
├── contributing/
│   ├── README.md            ← per-surface extension guides
│   ├── tools.md, knowledge.md, providers.md, agents.md, mcp_tools.md,
│   │   output_formats.md, schemas.md, fixtures.md, gpu_arch.md,
│   │   external-tools.md, proven_optimizations.md
│   ├── governance.md
│   └── walkthrough_new_bottleneck_class.md
├── audit_gate_runbook.md    ← parity / red-team / go-no-go commands
├── rfcs/
│   ├── README.md
│   ├── TEMPLATE.md
│   └── 0000-example-rfc.md
├── benchmarks/
│   └── mi300x_baseline.md
└── archive/
    └── migration-to-agentic.md
```

## Conventions

- Every Python / bash code block intended as illustrative (not executable
  in CI) is prefixed with `# SKIP-SAMPLE — <reason>` so
  `scripts/test-samples.py --strict` skips it.
- All cross-doc links are relative paths within this tree; external links
  are flagged out-of-scope by `scripts/link-checker.py`.
- The user-facing brand is **PerfXpert**. `rocinsight` / `ROCINSIGHT` are
  retired and should not appear in any new doc.
