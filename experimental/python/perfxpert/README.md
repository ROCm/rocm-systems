# PerfXpert

AI-powered AMD ROCm GPU trace analysis.

## Quickstart

### Prerequisites

- Python 3.10+
- (Optional) `claude`, `codex`, or `gemini` CLI on PATH if you plan to
  use the multi-backend dispatch.

The bundled AMD-branded opencode binary is compiled during `pip install`
by the setup.py build hook, which requires `bun`. If `bun` isn't on
PATH, the hook **auto-downloads** a prebuilt bun release into
`~/.cache/perfxpert/bun/bin/bun` (Linux x64/arm64 + macOS x64/arm64
supported; musl / Windows / offline installs fall back to a warn-skip
and `perfxpert-code` will print a helpful error at first launch).
Opt out with `PERFXPERT_SKIP_BUN_DOWNLOAD=1` if you'd rather manage bun
yourself, or `PERFXPERT_SKIP_BUNDLED_BUILD=1` to skip the whole build.

### Install

```bash
# SKIP-SAMPLE — install from PyPI (when published)
pip install "perfxpert[all]"

# OR install the latest development build direct from GitHub:
# SKIP-SAMPLE — install latest dev from source
pip install "perfxpert[all] @ git+https://github.com/ROCm/rocm-systems.git#subdirectory=experimental/python/perfxpert"
```

`[all]` pulls in the optional LLM providers (anthropic, openai,
claude-agent-sdk) plus rich for pretty terminal output.

### Run

```bash
# SKIP-SAMPLE — requires an existing rocprofv3 trace DB
# One-shot analysis (batch mode)
perfxpert analyze -i trace.db --llm anthropic --format webview -o report.html

# SKIP-SAMPLE — launches interactive TUI
# Interactive agentic TUI (AMD-branded bundled opencode)
perfxpert-code

# SKIP-SAMPLE — multi-backend dispatch (requires the native CLI installed)
# Claude / Codex / Gemini native CLIs with perfxpert MCP wired in:
perfxpert-code claude   # installs perfxpert MCP + gate into Claude Code, execs claude
perfxpert-code codex    # same for Codex CLI (trust-gate workflow)
perfxpert-code gemini   # same for Gemini CLI
perfxpert-code claude --dry-run "analyze this trace"   # preview, write nothing
perfxpert-code uninstall claude   # reverses install (refuses on marker drift)

# SKIP-SAMPLE — requires an existing rocprofv3 trace DB
# Air-gap mode (no LLM; deterministic rule-based analysis only)
PERFXPERT_AIRGAP=1 perfxpert analyze -i trace.db --format markdown -o report.md

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

Core analysis is self-contained — `pip install perfxpert` handles all
profiling + recommendation features. `perfxpert-code` ships an
AMD-branded opencode binary bundled directly into the wheel: the
pip-install build hook compiles it from the pinned `sst/opencode`
submodule + our patch series (`experimental/python/perfxpert/.patches/`)
during install, provided `bun` is on PATH.

If `bun` is missing, the install still succeeds (library + analyze + MCP
paths all work); `perfxpert-code` itself won't launch until you install
bun and rerun the install. As a last resort, set
`PERFXPERT_OPENCODE_PATH` to point at a pre-built opencode binary.

Advanced: for tightly-sandboxed CI where neither bun nor network access
is available, set `PERFXPERT_SKIP_BUNDLED_BUILD=1` before
`pip install` to suppress the build attempt entirely:

```bash
# SKIP-SAMPLE — actual installer; scripts/test-samples.py must not execute this
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
| `PERFXPERT_OPENCODE_PATH` | system PATH | Override path to opencode binary; must point to an existing file or launcher raises immediately |

The agentic runtime is the sole execution path; no feature flag
toggles it. Setting any of the following has no effect:

- `PERFXPERT_USE_AGENTS` — removed in Phase 7.1.
- `PERFXPERT_LEGACY` — removed in Phase 7.1.

See [CHANGELOG.md](CHANGELOG.md) for the removal history.

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

- **Getting started**
  - [Getting started guide](docs/guides/getting-started.md)
  - [Agentic mode: air-gap vs LLM, provider ladder](docs/guides/agentic-mode.md)
  - [Multi-backend launcher (`perfxpert-code claude|gemini|codex`)](docs/guides/backends.md)
    — register perfxpert with your native Claude Code / Gemini CLI
    TUI while keeping the perfxpert tool-priority gate.
- **Architecture (v0.2.0+)**
  - [Architecture overview](docs/architecture.md)
  - [Architecture index](docs/architecture/README.md)
    - [Agent hierarchy (Root / Analysis / Recommendation / specialists)](docs/architecture/agent-hierarchy.md)
    - [Gate cascade (5 correctness gates as middleware)](docs/architecture/gate-cascade.md)
    - [BackendAdapter protocol (multi-backend launcher)](docs/architecture/backend-adapter.md)
- **Integration**
  - [Integration index](docs/integration/README.md)
    - [MCP server (`perfxpert-mcp`) — 34 READ_ONLY tools](docs/integration/mcp-server.md)
- **Contributing**
  - [CONTRIBUTING.md](CONTRIBUTING.md)
  - [Contributing index](docs/contributing/README.md)
    - [External-tool dependencies (`require_tool`)](docs/contributing/external-tools.md)
- **Other**
  - [Historical migration notes](docs/archive/migration-to-agentic.md)
  - [Known issues and scanner scope limitations](docs/known-issues.md)

## Licensing

MIT. opencode is also MIT — bundled into the wheel via the build hook (`setup.py`) from the pinned upstream submodule.
