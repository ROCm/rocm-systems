# PerfXpert

AI-powered AMD ROCm GPU trace analysis.

## Quickstart

### Prerequisites

- Python 3.10+
- `bun` on PATH if you want the patched `opencode` build produced during
  `pip install`. Packaged installs bundle that build; source/editable
  checkouts can rebuild the same pinned fork locally. The build hook
  does not auto-download bun; if bun is missing, install still succeeds
  and only the patched launcher build is skipped. See
  [docs/guides/getting-started.md](docs/guides/getting-started.md)
  for the exact install contract and opt-out envs.
- (Optional) `claude`, `codex`, or `gemini` CLI on PATH for multi-backend dispatch.

### Install

> **First-time tip (stock Ubuntu / `rocm/dev-ubuntu-22.04` images):** the
> default pip (22.x) misreads perfxpert's PEP-621 metadata and aborts the
> wheel build. Upgrade pip once before installing:
>
> ```bash
> pip install -U pip setuptools wheel
> ```

```bash
# SKIP-SAMPLE — install from PyPI (when published)
pip install "perfxpert[all]"
```

To install the latest development build direct from the rocm-systems
monorepo on GitHub, use the wrapper script (~5 sec instead of
~5 min — details below):

```bash
# SKIP-SAMPLE — first clone the repo (no recursive submodules), then run
git clone --depth 1 --no-recurse-submodules https://github.com/ROCm/rocm-systems.git
bash rocm-systems/experimental/python/perfxpert/scripts/pip-install-from-git.sh
# Pass a ref / pip flags as positional args:
#   scripts/pip-install-from-git.sh v0.2.0
#   scripts/pip-install-from-git.sh <SHA> --user
```

Or invoke pip directly with the submodule-scope env vars set
(equivalent to the wrapper, just verbose):

```bash
# SKIP-SAMPLE — pip directly, scoped submodule init
GIT_CONFIG_COUNT=1 \
GIT_CONFIG_KEY_0=submodule.active \
GIT_CONFIG_VALUE_0=experimental/python/perfxpert/opencode \
  pip install "perfxpert[all] @ git+https://github.com/ROCm/rocm-systems.git#subdirectory=experimental/python/perfxpert"
```

The plain `pip install "perfxpert @ git+…"` one-liner still works but
triggers pip's full recursive submodule init over the ~34 unrelated
submodules declared at the rocm-systems repo root — 3-6 min of wasted
network on stock `rocm/dev-ubuntu-22.04`. The wrapper script scopes
the init down to just the opencode submodule perfxpert's build hook
actually needs; see `docs/guides/getting-started.md` §1.2 for the
measurements.

`[all]` pulls in the optional LLM providers (`anthropic`, `openai`,
`litellm`) plus `rich` for pretty terminal output. That covers the
hosted/local SDK-backed provider paths; the default patched `opencode`
path is validated separately through the launcher/build flow. Pick a
provider with `--llm <name>`.

### Run

```bash
# SKIP-SAMPLE — requires an existing rocprofv3 trace DB
# One-shot analysis (batch mode)
perfxpert analyze -i trace.db --llm anthropic --format webview -o report.html

# SKIP-SAMPLE — launches interactive TUI
# Interactive agentic TUI (default patched opencode path)
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

# SKIP-SAMPLE — requires two rocprofv3 trace DBs (baseline + candidate)
# Diff two runs (schema-0.3.1 trace_diff block — per-kernel deltas + verdict)
perfxpert diff baseline.db candidate.db --format markdown

# SKIP-SAMPLE — CI wrapper; rc=1 on regression, rc=0 otherwise
# Regression gate for CI pipelines (wraps `diff` + fails the build if slower)
perfxpert ci baseline.db candidate.db --threshold 3.0

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
profiling + recommendation features. `perfxpert-code` defaults to the
patched opencode path: packaged installs use the bundled AMD-branded
binary, while source/editable checkouts can rebuild the same pinned
`sst/opencode` fork locally from
`experimental/python/perfxpert/opencode` plus our patch series
(`experimental/python/perfxpert/.patches/`) when `bun` is on PATH.

If `bun` is missing, the install still succeeds (library + analyze + MCP
paths all work). Packaged installs then miss the bundled patched build,
and source/editable checkouts cannot rebuild the repo-local patched
binary until bun is available. As a last resort, set
`PERFXPERT_OPENCODE_PATH` to point at a patched opencode binary.

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
| `PERFXPERT_OPENCODE_PATH` | system PATH | Override path to opencode binary; point this at a patched build if you need the full perfxpert gate behavior |

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

## Caution

> **Experimental software.** PerfXpert is still evolving and is provided
> without warranties or guarantees. AI-generated analysis, explanations, and
> recommendations can be incomplete or incorrect, so verify important results
> before relying on them in production or performance-critical workflows.

## Licensing

MIT. opencode is also MIT — the packaged build bundles a patched binary
from the pinned upstream submodule, and source/editable checkouts use
that same fork via the local submodule build path.
