# PerfXpert

AI-powered AMD ROCm GPU trace analysis.

## Caution

> **Experimental software.** PerfXpert is still evolving and is provided
> without warranties or guarantees. AI-generated analysis, explanations, and
> recommendations can be incomplete or incorrect, so verify important results
> before relying on them in production or performance-critical workflows.



## Quickstart

### Install

```bash
# SKIP-SAMPLE — package install + venv setup are host-specific
# Ubuntu 22/24 example. Use the package-manager equivalent on RHEL/SLES.
apt install -y curl git unzip python3-venv python3-pip
python3 -m venv .venv
. .venv/bin/activate

# Latest development build from ROCm/rocm-systems.
REF=develop; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"
```

Pin a tag or commit by changing `REF`:

```bash
# SKIP-SAMPLE — replace <SHA> with a real tag or commit
REF=<SHA>; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"
```

The wrapper installs from GitHub, scopes submodule init to the pinned
PerfXpert `opencode` submodule, and bootstraps bun when needed. It
builds the patched bundled `perfxpert-code` binary and verifies it before exiting.
No separate `opencode` install is needed for the default `perfxpert-code`
TUI. See [docs/guides/getting-started.md](docs/guides/getting-started.md)
for the Ubuntu/RHEL/SLES package matrix, direct-pip equivalent, editable
installs, and troubleshooting.

### LLM Providers

| Provider | Source | Typical use |
|----------|--------|-------------|
| `anthropic` | Claude API | Production default; requires `ANTHROPIC_API_KEY` |
| `openai` | OpenAI API | Alternative hosted; requires `OPENAI_API_KEY` |
| `ollama` | Local Ollama | Fully local; requires a running `ollama serve` |
| `private` | Any OpenAI-compatible endpoint | Internal deployments; requires `PERFXPERT_LLM_PRIVATE_URL` + `PERFXPERT_LLM_PRIVATE_MODEL`; CLI preflight also needs `PERFXPERT_LLM_PRIVATE_API_KEY` or `--llm-api-key` |
| `opencode` | Bundled opencode CLI | Used by `perfxpert-code`; not callable from inside opencode itself (recursion-guarded) |

Private endpoint example:

```bash
# SKIP-SAMPLE — requires a real trace.db and reachable private endpoint
export PERFXPERT_LLM_PRIVATE_URL="https://llm-api.iexample.com/OpenAI"
export PERFXPERT_LLM_PRIVATE_MODEL="gpt-5.3-codex"
export PERFXPERT_LLM_PRIVATE_API_KEY="..."
export PERFXPERT_LLM_PRIVATE_HEADERS='{"Ocp-Apim-Subscription-Key":".......","user":".....","api-version":"preview"}'
perfxpert analyze -i trace.db --llm private
```

### Analyze

```bash
# SKIP-SAMPLE — requires a real trace.db and provider credentials
export ANTHROPIC_API_KEY="sk-ant-..."
perfxpert analyze -i trace.db --llm anthropic --format webview -o report.html

export OPENAI_API_KEY="sk-..."
perfxpert analyze -i trace.db --llm openai --llm-model gpt-4o-mini --format markdown -o report.md

# Air-gap mode: no LLM calls, deterministic local analysis only.
PERFXPERT_AIRGAP=1 perfxpert analyze -i trace.db --format markdown -o report.md
```

### Interactive TUI

```bash
# SKIP-SAMPLE — launches interactive CLIs and may write backend config
# Default first-class TUI: bundled patched opencode built during install.
perfxpert-code

# Use native shells with PerfXpert MCP and context installed for that backend.
perfxpert-code claude
perfxpert-code codex
perfxpert-code gemini

# Explicit upstream-opencode escape hatch. The default TUI does not need this.
PERFXPERT_OPENCODE_PATH="$(command -v opencode)" perfxpert-code opencode
```

Preview or uninstall a backend integration:

```bash
# SKIP-SAMPLE — preview/uninstall commands mutate or inspect backend setup
perfxpert-code claude --dry-run "analyze this trace"
perfxpert-code uninstall claude
```

### Other Commands

```bash
# SKIP-SAMPLE — requires real trace DB paths
perfxpert diff baseline.db candidate.db --format markdown
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
│   — all 8 agents callable via MCP + perfxpert.api            │
│                          │                                   │
│                          ▼                                   │
│    Deterministic middleware (gate_cascade, intent router)    │
│                          │                                   │
│                          ▼                                   │
│  56 READ_ONLY MCP tools (8 agent + 47 classifier + 1 diff)   │
│            + ~22 knowledge YAMLs                             │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

Core analysis is self-contained — `pip install perfxpert` handles all
profiling + recommendation features. The default `perfxpert-code` path
uses only the bundled AMD-branded opencode binary built from the pinned
`experimental/python/perfxpert/opencode` submodule plus the local patch
series. If the build prerequisites are missing, pip fails with the
missing package-manager pieces instead of falling back to an arbitrary
opencode binary.

Advanced: `PERFXPERT_SKIP_BUNDLED_BUILD=1` is only for tightly-sandboxed
CI that intentionally skips the interactive TUI build. Users who
explicitly want their own upstream opencode can run
`perfxpert-code opencode ...`; the default `perfxpert-code` command
continues to require the bundled submodule-built binary.

## Contributing

perfxpert welcomes contributions. Start with [CONTRIBUTING.md](CONTRIBUTING.md)
for the extension-surface matrix + governance. Per-surface guides under
[docs/contributing/](docs/contributing/). Architectural changes go through
an [RFC](docs/rfcs/README.md).

## Feature flags

| Flag | Default | Purpose |
|------|---------|---------|
| `PERFXPERT_OPENCODE_PATH` | unset | Explicit upstream-opencode escape hatch used only by `perfxpert-code opencode ...`; the default TUI ignores it |

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
    - [MCP server (`perfxpert-mcp`) — 56 READ_ONLY tools (8 agent-hierarchy + 47 knowledge/classifier + 1 trace_diff)](docs/integration/mcp-server.md)
    - [Python API (`perfxpert.api`) — 1:1 mirror of the 8 agent MCP tools](docs/guides/python-api.md)
- **Contributing**
  - [CONTRIBUTING.md](CONTRIBUTING.md)
  - [Contributing index](docs/contributing/README.md)
    - [External-tool dependencies (`require_tool`)](docs/contributing/external-tools.md)
- **Other**
  - [Historical migration notes](docs/archive/migration-to-agentic.md)

## Licensing

MIT. opencode is also MIT — the packaged build bundles a patched binary
from the pinned upstream submodule, and source/editable checkouts use
that same fork via the local submodule build path.
