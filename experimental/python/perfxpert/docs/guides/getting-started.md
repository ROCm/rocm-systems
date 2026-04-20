# PerfXpert — Getting Started

This guide walks through
install, the three entry points, the first analysis, and the top
troubleshooting cases. For deeper dives follow the cross-links at the
end of each section; nothing here duplicates reference material that
lives elsewhere.

## What is PerfXpert

PerfXpert is an AI-augmented GPU profiling and optimization tool for
AMD ROCm. It reads `rocprofv3` trace databases (`.db`), classifies
bottlenecks against hardware SoL bounds, and produces targeted
recommendations — either deterministically (air-gap) or with an LLM
specialist agent. Supported GPUs: MI100 / MI200 / MI300 / MI350 and
RDNA2 / RDNA3.

## 1. Install

PerfXpert ships as a single Python wheel. The `setuptools` build hook
in `setup.py` automatically compiles the AMD-branded bundled opencode
binary during `pip install` — no separate step to remember.

### Prerequisites

- Python 3.10+
- Network access during `pip install` (the build hook downloads
  opencode source + the bun runtime if not already on PATH).

The setup.py build hook compiles the bundled AMD-branded opencode
binary from the pinned `sst/opencode` submodule. This requires `bun`;
if `bun` isn't on PATH the hook **auto-downloads** a prebuilt bun
release into `~/.cache/perfxpert/bun/bin/` (or
`%USERPROFILE%\.cache\perfxpert\bun\bin\` on Windows).

#### Supported hosts for auto-bun-download

| Host | bun asset downloaded |
|------|---------------------|
| Linux x64 glibc | `bun-linux-x64.zip` |
| Linux x64 musl (Alpine) | `bun-linux-x64-musl.zip` |
| Linux aarch64 glibc | `bun-linux-aarch64.zip` |
| Linux aarch64 musl | `bun-linux-aarch64-musl.zip` |
| macOS x64 | `bun-darwin-x64.zip` |
| macOS arm64 | `bun-darwin-aarch64.zip` |
| Windows x64 | `bun-windows-x64.zip` |
| Windows arm64 | `bun-windows-aarch64.zip` |

Offline installs fall back to a warn-skip; `perfxpert-code` will then
print a helpful message on first launch explaining how to install bun
manually.

#### Opt-out env vars

- `PERFXPERT_SKIP_BUN_DOWNLOAD=1` — don't auto-fetch bun; let the
  hook warn-skip if bun isn't already on PATH.
- `PERFXPERT_SKIP_BUNDLED_BUILD=1` — skip the entire opencode build.
  Library + analyze + MCP all still work; `perfxpert-code` will
  fall back to whatever `opencode` is on PATH or exit with a helpful
  error.
- `PERFXPERT_OPENCODE_PATH=/path/to/opencode` — last-resort override
  pointing the launcher at a pre-built binary.

#### Note on Windows

Bun itself is supported and auto-downloaded, but the subsequent
`bun build` of opencode inherits opencode's own platform-support
matrix. If the build step fails on Windows, install `perfxpert`
without the bundled launcher (`PERFXPERT_SKIP_BUNDLED_BUILD=1 pip
install …`) and use the multi-backend launcher
(`perfxpert-code claude` / `codex` / `gemini`) — those route to
native backend CLIs and work anywhere.

### Pip install

> **Run this FIRST on stock Ubuntu / rocm/dev-ubuntu images**: those
> images ship with pip 22.x and pre-PEP-621 setuptools, which fails the
> perfxpert wheel build with `filename has 'perfxpert', but metadata has
> 'unknown'`. The fix is a one-liner:
>
> ```bash
> pip install -U pip setuptools wheel
> ```
>
> The build requires `setuptools>=61` (declared in `pyproject.toml`'s
> `[build-system] requires`). Recent pip (23+) honors that declaration
> automatically; older pip does not.

```bash
# SKIP-SAMPLE — install from a local checkout (editable mode)
pip install -e "experimental/python/perfxpert[all]"
```

…or pull the latest build straight from GitHub using the fast-install
wrapper (see §1.2 for why):

```bash
# SKIP-SAMPLE — clone root without recursing submodules, then run wrapper
git clone --depth 1 --no-recurse-submodules https://github.com/ROCm/rocm-systems.git
bash rocm-systems/experimental/python/perfxpert/scripts/pip-install-from-git.sh
```

The `[all]` extra pulls in `anthropic`, `openai`, `rich`, and
`litellm` so every LLM provider works out of the box. Omit it if you
only want deterministic air-gap analysis (wrapper: pass
`--extras ''` to skip extras entirely).

### 1.2 Why the wrapper: scoped submodule init

`pip install "perfxpert @ git+https://...rocm-systems.git#subdirectory=..."`
triggers pip's built-in `git submodule update --init --recursive -q`
on the cloned work-tree BEFORE the perfxpert build hook ever runs.
The rocm-systems root `.gitmodules` declares ~34 submodules (mscclpp,
perfetto, glog, fmt, gtest, dyninst, sqlite, …) shared by the other
projects in the monorepo; pip dutifully fetches every one of them.
PerfXpert itself only needs ONE of them — the `opencode` submodule at
`experimental/python/perfxpert/opencode`, used by the build hook to
compile the bundled AMD-branded opencode binary.

Measured on a fast host against the live rocm-systems repo:

| Step                                                  | Time      |
|-------------------------------------------------------|-----------|
| `git clone --filter=blob:none --depth 1` of rocm-systems | ~15 sec   |
| `git submodule update --init --recursive -q` (pip default) | **141 sec** |
| `git -c submodule.active=…/opencode submodule update --init --recursive -q` | **0.03 sec** |

On stock `rocm/dev-ubuntu-22.04` with corporate-grade bandwidth the
default step regularly runs 3-6 minutes — and the first 99% of that
time is downloading dependencies of projects the PerfXpert wheel
never touches.

`scripts/pip-install-from-git.sh` wraps pip with these env vars set:

```bash
GIT_CONFIG_COUNT=1
GIT_CONFIG_KEY_0=submodule.active
GIT_CONFIG_VALUE_0=experimental/python/perfxpert/opencode
```

pip uses `os.environ.copy()` when it spawns git subprocesses, so
those env vars propagate into `git submodule update --init
--recursive -q` and the `submodule.active` config restricts init to
the single path listed. All other submodules stay at zero bytes on
disk. Documented under `git-config(1) "GIT_CONFIG_COUNT"` and
`gitmodules(5) "submodule.<name>.active"`.

If the user insists on the plain one-liner without the wrapper, pip
still works — they just pay the 3-6 min submodule-init penalty once.
The `setup.py` build hook notices if the opencode submodule is still
empty after pip's checkout (i.e. the user scoped submodule init out
manually without including opencode) and falls back to a direct
shallow clone of `sst/opencode` at the pinned tag — so the install
always completes regardless of how the user configured git.

Opt-outs:

- `PERFXPERT_SKIP_OPENCODE_FETCH=1` — don't attempt the fallback
  clone; air-gap CI that intentionally skips the bundled opencode
  build should set this AND `PERFXPERT_SKIP_BUNDLED_BUILD=1`.

### What the build hook does

It applies all 26 patches in `.patches/` (AMD branding, color palette,
per-model system-prompt preambles with the STRICT-TOOL-DISCIPLINE
stanza, the tool-priority gate, and the deep-rebrand session UI) to the
pinned `opencode` submodule, then runs `bun build` to produce the
bundled binary at `perfxpert/_bundled/opencode`. Subsequent `pip install`
invocations skip the rebuild if the binary is already newer than every
patch file.

**Opt-out:** set `PERFXPERT_SKIP_BUNDLED_BUILD=1` to suppress the build
(useful in offline / sandboxed CI). `perfxpert-code` will then fall back
to whatever `opencode` is on `PATH`, which will NOT include our tool-
priority gate or system prompt — library + analyze + MCP paths still
work fine. A last-resort override `PERFXPERT_OPENCODE_PATH` points the
launcher at an explicit binary.

**Bun missing:** the install completes with a clear warning; only
`perfxpert-code` (the interactive TUI) is affected.

## 2. Verify

```bash
# SKIP-SAMPLE — requires perfxpert on PATH from the pip install above
perfxpert doctor
```

Expected output ends with `ALL CLEAN` when everything is wired. The
doctor checks:

- `perfxpert` version + Python ≥ 3.10
- openai-agents SDK
- MCP server reachable (`perfxpert-mcp` boots + 41 tools registered — 7 agent-hierarchy + 34 classifier/knowledge)
- task store (`~/.perfxpert` or `$PERFXPERT_TASK_ROOT`)
- bundled opencode binary + bundled opencode config dir
- LLM providers configured (counts `N/5` against
  `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OLLAMA_HOST`,
  `PRIVATE_LLM_ENDPOINT`, plus always-present `opencode`)

If `opencode binary` reports missing, re-run the patch script
(§1 step 2) or install upstream opencode:
`curl -fsSL https://opencode.ai/install | bash`.

## 3. Three entry points

PerfXpert ships three command-line surfaces, all driving the same
agent runtime.

- **`perfxpert analyze -i trace.db`** — non-interactive CLI. Consumes
  a rocprofv3 `.db`, emits a single report (text / JSON / markdown /
  webview HTML). Deterministic with `--llm` omitted; LLM-augmented
  with `--llm {anthropic,openai}`.
- **`perfxpert-mcp`** — stdio MCP server that re-exposes the 41
  READ-ONLY analysis tools over JSON-RPC (7 agent-hierarchy entry
  points — Root, Analysis, Recommendation, Correctness, +3
  specialists — plus 34 classifier / knowledge tools). Meant to be
  spawned by an MCP client (Claude Desktop, Claude Code, Codex CLI,
  Gemini CLI, opencode). See `../integration/mcp-server.md`.
- **`perfxpert-code`** — interactive TUI (the patched opencode
  bundle). Conversational optimization loop with the Root → Analysis
  → Recommendation → Specialist agent hierarchy and gate-cascade
  correctness middleware.

## 3.1 Choosing a backend

`perfxpert-code` is multi-backend: the same AMD-branded bundled
opencode is the default, but it can also wrap the user's native
Claude Code, Gemini CLI, or (soon) Codex TUI while still enforcing
the perfxpert tool-priority gate and registering the `perfxpert-mcp`
server for free. Pick whichever matches your existing LLM workflow.

- **Default (no subcommand)** — AMD-branded bundled opencode, the
  recommended entry point. Ships with patched prompt + MCP
  pre-wired, no extra install.
- **`perfxpert-code claude`** — registers perfxpert as an MCP
  server in the Claude Code project config, installs the native
  `PreToolUse` gate hook, then execs the user's `claude` CLI.
- **`perfxpert-code gemini`** — appends `perfxpert-mcp` to
  `~/.gemini/settings.json` and list-appends a project-cache
  prompt file to `context.fileName` (never touches user's
  `GEMINI.md`), then execs `gemini`.
- **`perfxpert-code codex`** — writes the perfxpert MCP stanza into
  `~/.codex/config.toml` (TOML, not JSON) and execs `codex`. Gate
  enforcement is prompt-layer-only because Codex's native
  `PreToolUse` hook is Bash-only as of April 2026; the trust gate
  runs before MCP registration and either prompts or honors
  `PERFXPERT_AUTO_TRUST=1` (see §3.2 below).

Short recipe per backend:

```bash
# SKIP-SAMPLE — requires the named backend binary on PATH
perfxpert-code                              # default (bundled opencode)
perfxpert-code claude                       # Claude Code (native TUI)
perfxpert-code gemini                       # Gemini CLI
perfxpert-code codex                        # Codex CLI
perfxpert-code uninstall claude             # reverse the Claude install
```

See [backends.md](backends.md) for the full per-backend install /
uninstall recipes, the gate-hook event-based lift semantics, the
consent model (per-backend × cwd × file-set), and the env-var
reference (`PERFXPERT_MCP_WARMUP_TIMEOUT_S`,
`PERFXPERT_MCP_RETRY_BUDGET_S`, `PERFXPERT_SKIP_LIVE_CHECK`,
`PERFXPERT_ASSUME_CONSENT`). The gate-probe coverage table in
[backends.md §Gate-probe](backends.md) documents which small-model
probe each backend uses to verify mechanical gate enforcement at
`install()` time — Codex is not probed because its gate is
prompt-layer-only.

## 3.2 Codex trust gate

Codex refuses to run agents in a project directory that isn't marked
`trusted` in `~/.codex/config.toml`. `perfxpert-code codex` handles
this as a separate step BEFORE MCP registration:

- **Interactive sessions** — the adapter prompts "Trust <cwd> for
  Codex? [y/N]" and writes `[projects."<abs-cwd>"].trust_level =
  "trusted"` on confirmation.
- **CI / non-interactive** — set `PERFXPERT_AUTO_TRUST=1` to
  auto-trust the cwd without prompting. An always-on stderr warning
  ("perfxpert-code codex: auto-trusted <cwd> because
  PERFXPERT_AUTO_TRUST=1") is emitted even under `--quiet` — this is
  the security-warning contract.
- Without either path, the adapter raises `TrustRequired` and exits
  non-zero. `ConfigClobber` is raised if `~/.codex/config.toml` is
  git-tracked or fails to parse.

## 4. First analysis (60 seconds)

Profile a trivial HIP app, then analyze.

```bash
# SKIP-SAMPLE — requires ./my_app built against ROCm and rocprofv3 on PATH
rocprofv3 --sys-trace --summary -d ./out -o results -- ./my_app
perfxpert analyze -i ./out/results_results.db --format text
```

`rocprofv3` writes `./out/results_results.db` (the `_results` suffix
is the rocprofv3 default); `perfxpert analyze` reads it, classifies
the bottleneck (compute-bound / memory-bound / latency-bound /
idle-bound), ranks hot kernels by Amdahl weight, and emits
recommendations.

Output formats:

```bash
# SKIP-SAMPLE — requires a real trace.db
perfxpert analyze -i trace.db --format json -d ./out -o report
perfxpert analyze -i trace.db --format markdown -d ./out -o report
perfxpert analyze -i trace.db --format webview -d ./out -o report
```

The webview format produces a self-contained HTML file (AMD dark
theme, SVG gauges, collapsible recommendation cards) suitable for
sharing over email.

### Report contents (every format)

Every format — text, JSON, markdown, webview — carries the same
dataset. The deterministic pass runs unconditionally (even with
`--llm <prov>` or under airgap); the LLM supplies narrative +
primary-bottleneck + recommendation prose on top. The four formats
differ only in rendering, not in completeness. Every report
contains:

- **Agent narrative** — the LLM prose (or the airgap template in
  deterministic mode), spliced at the top of the report under a
  "Summary" heading.
- **Primary bottleneck** — explicit label (compute / memory_transfer
  / latency / mixed / data_insufficient).
- **Time breakdown** — kernel / memcpy / API-overhead percentages
  plus total runtime and kernel count.
- **Hotspot list** — top-N kernels ranked by total duration, with
  call count, avg / min / max duration, percent of total.
- **Memory analysis** — H2D / D2H / D2D volumes, total duration, and
  average bandwidth per direction.
- **Hardware counters** — Tier-2 derived metrics (GPU utilization,
  avg waves, HBM utilization) plus the raw collected counter table.
  Shows a graceful "not collected" placeholder when the `.db` was
  captured without `--pmc`.
- **Kernel resources / occupancy** — VGPR / SGPR / LDS / scratch and
  theoretical occupancy for each hotspot kernel when the trace
  carries kernel-symbol metadata.
- **API overhead breakdown** — top HIP / HSA API calls by total time
  plus warmup outliers when detected.
- **Thread-trace (Tier 3)** — included when `--att-dir` is set;
  per-instruction stall ratio and bottleneck category (VMEM,
  LDS-bank, dep-chain, branch-divergence).
- **Tier-0 source findings** — included when `--source-dir` is set;
  detected kernels, anti-patterns, suggested counters, and the
  suggested first-profiling command.
- **Recommendations (merged)** — LLM + deterministic recommendations
  merged and deduped by target, each with category / issue /
  suggestion / citation / code-snippet-before / code-snippet-after
  (where available).
- **Warnings** — alert block (empty when no warnings fire).
- **Metadata** — GPU arch, DB path, kernel count, total runtime,
  provider, model (footer).

The JSON format exposes each section under a flat top-level key
(`.time_breakdown`, `.hotspots`, `.memory_analysis`,
`.hardware_counters`, `.kernel_resources`, `.api_overhead`,
`.tier0_findings`, `.recommendations`, `.narrative`,
`.primary_bottleneck`, `.warnings`, `.metadata`) — so `jq` pipelines
can slice the report without reaching into nested schema shapes.

### Report structure

Every report renders the same top-level sections in a fixed order. The
main take-away: **the Tier-0 source scan always lives in its own
section — NEVER folded into the main recommendations table**. Listed
top-to-bottom:

1. **Summary** — agent narrative (findings-derived: dominant kernel +
   metric evidence + primary-bottleneck verdict). Never routing prose.
2. **Time breakdown** — kernel / memcpy / API-overhead percentages.
3. **Hotspots** — top-N kernels by total duration.
4. **Memory analysis** — per-direction volume, duration, bandwidth.
5. **Hardware counters** — Tier-2 derived metrics + raw table.
6. **Kernel resources / occupancy** — VGPR / SGPR / LDS / scratch.
7. **API overhead** — top HIP / HSA calls + warmup outliers.
8. **Thread trace (Tier 3)** — only when `--att-dir` is set.
9. **Recommendations** — merged LLM + deterministic recommendations,
   deduped by target. **Only real perf-issue items** (e.g. hot-kernel
   triage, cache-unfriendly access, synchronous hipMemcpy patterns).
   Instrumentation advice (what counters to collect, what rocprofv3
   command to run first) is **NOT** in this list.
10. **Tier-0: Source Scan** — only when `--source-dir` is set. Split
    into two clearly-labelled subsections:
    - **Profiling Plan** — suggested first `rocprofv3 --sys-trace …`
      command, suggested counters, other instrumentation actions.
    - **Detected Code Patterns** — actual code-level perf issues found
      by the scanner (these ALSO appear in the main Recommendations
      list so they don't get overlooked).
11. **Warnings** — block (empty when no warnings fire).
12. **Metadata** — GPU arch, DB path, provider, model (footer).

In the JSON format the Tier-0 block lives under
`.tier0_findings.profiling_plan` +
`.tier0_findings.code_patterns`. The webview carries it under
`<section id="tier0">` with `<h3>Profiling Plan</h3>` and a Detected
Code Patterns table, inserted AFTER the main recommendations so the
two groups remain visually distinct.

## 5. Multi-GPU / MPI workflows

**Correct pattern — MPI OUTSIDE, rocprofv3 INSIDE, per rank.** Each
rank gets its own rocprofv3 instance and writes its own `.db` file:

```bash
# SKIP-SAMPLE — requires mpirun + ./app built against ROCm
mpirun -n 8 rocprofv3 --sys-trace -d ./out -o results_%q{MPI_RANK} -- ./app
perfxpert analyze -i ./out/merged_processes.db
```

**Wrong — do NOT emit this form.** `rocprofv3` attaches to `mpirun`,
not to the ranks, so the `.db` is empty or contains only mpirun's
no-GPU runtime:

```bash
# SKIP-SAMPLE — shown for contrast; DO NOT RUN
rocprofv3 [flags] -- mpirun -n 8 ./app   # WRONG
```

Additional rules (enforced in `_bundled/opencode_config/AGENTS.md`):

- Use `-o results_%q{MPI_RANK}` (or `%nid%` on Slurm) so ranks don't
  race on the same file.
- Do NOT use `--process-sync` with OpenMPI — it strips `LD_PRELOAD`
  and breaks tracer injection.
- Slurm / jsrun wrappers go OUTSIDE rocprofv3 too:
  `srun rocprofv3 ... -- ./app`.

The full canonical rule — including FETCH_SIZE / WRITE_SIZE PMC
isolation — lives in `../../perfxpert/_bundled/opencode_config/AGENTS.md`.

## 6. LLM modes

Two modes, one code path. The agent hierarchy and gate cascade behave
identically in both; the difference is whether an LLM rewrites the
narrative and recommendation titles.

### Air-gap (default, deterministic)

```bash
# SKIP-SAMPLE — requires a real trace.db
PERFXPERT_AIRGAP=1 perfxpert analyze -i trace.db
# or simply omit --llm — omission defaults to deterministic mode
perfxpert analyze -i trace.db
```

Air-gap mode: no outbound calls, rule-based classification against
the knowledge YAMLs. `primary_bottleneck` is still set.
`recommendations[].name` populates only with LLM mode; air-gap
returns bottleneck + narrative only (verbatim from the rule tables).

### LLM-enabled

All five supported providers are selectable from the CLI with `--llm
<name>`: `anthropic`, `openai`, `ollama`, `private`, `opencode`. The
same five are also reachable from Python via
`perfxpert.api.agent_root(..., provider=<name>)` — the CLI is a thin
wrapper over the public Python API.

```bash
# SKIP-SAMPLE — requires a real trace.db and an LLM credential
export ANTHROPIC_API_KEY="sk-ant-..."
perfxpert analyze -i trace.db --llm anthropic
# or:
export OPENAI_API_KEY="sk-..."
perfxpert analyze -i trace.db --llm openai
```

An LLM analysis can take 1-5 minutes per call — PerfXpert draws a live
progress spinner on stderr so you can see each agent phase as it
enters / exits (`entering root`, `entering analysis`, etc.) and if the
fallback chain cascades across providers. The spinner is stderr-only,
so piping stdout to a file (e.g. `--format json > out.json`) still
captures clean output. Two opt-outs:

- `--no-progress` — silent (useful for CI and log capture).
- `--verbose` — full log lines instead of the compact spinner (unchanged
  from prior releases).

When stderr is not a TTY (piped / redirected / under a CI runner) the
spinner degrades automatically to plain `[perfxpert] <phase>` status
lines, with no ANSI escapes.

For the full provider matrix, model-selection ladder, and fallback
chain, see §10 LLM Providers below.

### Credentials

Each provider has a canonical env var PerfXpert reads at session
boot; the `--llm-api-key` CLI flag is an equivalent one-shot
override. If both are set and differ the flag wins (PerfXpert emits
a one-line stderr WARNING so you know which credential is active):

| `--llm` | Primary env var | PerfXpert alias | Notes |
|---------|-----------------|-----------------|-------|
| `anthropic` | `ANTHROPIC_API_KEY` | `PERFXPERT_LLM_ANTHROPIC_KEY` | Either works; alias kept for migration parity |
| `openai` | `OPENAI_API_KEY` | `PERFXPERT_LLM_OPENAI_KEY` | Either works |
| `private` | `PERFXPERT_LLM_PRIVATE_API_KEY` | — | Plus `PERFXPERT_LLM_PRIVATE_URL` (required) |
| `ollama` | — (no key) | — | Plus `PERFXPERT_LLM_LOCAL_URL` (default `http://localhost:11434`) |
| `opencode` | — (no key) | — | Plus `PERFXPERT_OPENCODE_PATH` if not on `PATH` |

```bash
# SKIP-SAMPLE — requires a real trace.db and a live ANTHROPIC_API_KEY
# CLI flag — equivalent to exporting ANTHROPIC_API_KEY
perfxpert analyze -i trace.db --llm anthropic --llm-api-key sk-ant-...

# Env var — survives across invocations
export ANTHROPIC_API_KEY="sk-ant-..."
perfxpert analyze -i trace.db --llm anthropic
```

**Missing credentials surface a clean pre-flight error.** Starting
with Phase 8, running `--llm anthropic` with no `ANTHROPIC_API_KEY`
and no `--llm-api-key` raises a one-line stderr message like:

```
⚠ LLM auth failed for anthropic. Check ANTHROPIC_API_KEY is set correctly.
```

and exits with rc=2 BEFORE any network call or formatter pass runs.
No empty HTML / markdown file is left behind — the previous
"silently produces a blank report" failure mode is gone.

---

## 7. Tier 0: Source Code Scanning

Analyze your source code before profiling — no GPU or trace database needed.

```bash
# SKIP-SAMPLE — requires a ./my_app source tree and/or trace.db
# Scan source directory
perfxpert analyze --source-dir ./my_app

# Combined: source scan + trace analysis
perfxpert analyze -i trace.db --source-dir ./my_app
```

PerfXpert scans `.hip`, `.cpp`, `.cu`, `.cl`, `.py`, `.h`, `.hpp` files and detects:
- GPU kernel definitions and launch patterns
- Memory operations (hipMemcpy, hipMemcpyAsync)
- Synchronization points (hipDeviceSynchronize)
- Stream usage (or lack thereof)
- Framework usage (PyTorch, JAX, TensorFlow)
- ROCTx markers

The output includes a profiling plan with the exact `rocprofv3` command to run, with counters pre-selected based on what was found in the source.


---

## 8. Agentic TUI Workflow (The Star Feature)

The agentic TUI automates the full optimization loop: profile, analyze,
AI-edit code, recompile, re-profile, compare. As of v0.2.0 this is the
`perfxpert-code` command (AMD-themed bundled opencode TUI) — it wraps the
same agent runtime the batch-mode `analyze` CLI uses.

```bash
# SKIP-SAMPLE — requires bundled opencode binary on PATH
perfxpert-code
```

Inside the TUI you describe your workload in natural language
(e.g. "profile ./my_app and suggest optimizations"). The Root agent then
drives the Analysis → Recommendation → Specialist hierarchy behind the
scenes.

### What happens:

1. **Workload detection** — identifies your binary type (HIP, Python ML,
   MPI) and selects optimal profiling flags
2. **Profiling plan** — shows the generated `rocprofv3` command for your
   approval
3. **Profile run** — runs the profiler with real-time output streaming
4. **Analysis** — analyzes the trace, shows findings with AI-refined
   recommendations
5. **Recommendations menu** — address with AI, skip, or re-profile
6. **AI edit** — AI edits your source files using precise SEARCH/REPLACE
   blocks (see Gate Cascade doc for correctness guarantees)
7. **Re-profile** — re-profile with the optimized code and compare


### AI Code Editing

When the agent applies an optimization, the LLM generates targeted code
changes as SEARCH/REPLACE blocks — not full-file rewrites. This prevents
truncation on large files and makes the diff easy to review:

```
<<<<<<< SEARCH
    for(int i = 0; i < CHUNKS; i++) {
        HIP_CHECK(hipMemcpy(d_in + i * chunk, ...));
    }
=======
    HIP_CHECK(hipMemcpyAsync(d_in, h_in.data(),
                 N * sizeof(float),
                 hipMemcpyHostToDevice, stream1));
>>>>>>> REPLACE
```

If the edit causes compilation errors, the Correctness agent reverts the
change automatically; see `docs/architecture/gate-cascade.md` for the full
5-gate correctness/regression contract.

---

## 9. MPI Multi-GPU Profiling (detailed)

PerfXpert auto-detects MPI launchers and restructures the profiling command so each rank gets its own profiler instance.

```bash
# SKIP-SAMPLE — requires a built MPI application + openmpi
# Batch-mode analyze that profiles an MPI workload end-to-end
perfxpert analyze \
  --llm anthropic \
  --source-dir ./src \
  --run "mpirun -n 8 ./multi_gpu_demo"
```

Inside `perfxpert-code` you can describe the same workload in natural
language ("profile mpirun -n 8 ./multi_gpu_demo") and the Analysis agent
will drive the same detection logic.

The tool:
- Detects `mpirun`/`srun`/`jsrun` and wraps each rank: `mpirun -n 8 rocprofv3 <flags> -- ./binary`
- Uses `%nid%` per-rank output naming to avoid SQLite collisions
- Auto-merges all per-rank databases into a single `merged_processes.db`
- Analyzes the unified trace across all GPUs


> **Note**: `--process-sync` (used for Python DDP/torchrun) is NOT used for MPI because OpenMPI strips LD_PRELOAD from child processes. PerfXpert handles this automatically.

---

## 10. LLM Providers

All five LLM providers are selectable via `--llm <name>` on the CLI
**and** via `provider=<name>` on `perfxpert.api.agent_root(...)` — the
same registry backs both surfaces. LLM is optional; all analysis runs
locally without internet when you omit `--llm` (or set
`PERFXPERT_AIRGAP=1`).

| `--llm` | Env vars | Purpose |
|---------|----------|---------|
| `anthropic` | `ANTHROPIC_API_KEY` | Claude API (production default) |
| `openai` | `OPENAI_API_KEY` | OpenAI hosted API |
| `ollama` | `OLLAMA_HOST` (default `http://localhost:11434`) | Local Ollama daemon — fully offline once the model is pulled |
| `private` | `PERFXPERT_LLM_PRIVATE_URL`, `PERFXPERT_LLM_PRIVATE_MODEL`, optional `PERFXPERT_LLM_PRIVATE_API_KEY` | Any OpenAI-compatible endpoint (enterprise / self-hosted) |
| `opencode` | none required (bundled) | Bundled opencode CLI — subprocess wrapper |

```bash
# SKIP-SAMPLE — requires a real trace.db and an LLM credential
export ANTHROPIC_API_KEY="sk-ant-..."
perfxpert analyze -i trace.db --llm anthropic --llm-model claude-sonnet-4-5
# or:
export OPENAI_API_KEY="sk-..."
perfxpert analyze -i trace.db --llm openai --llm-model gpt-4o

# Private endpoint (any OpenAI-compatible server)
export PERFXPERT_LLM_PRIVATE_URL="https://llm.corp.internal/v1"
export PERFXPERT_LLM_PRIVATE_MODEL="llama-3-70b"
perfxpert analyze -i trace.db --llm private

# Local Ollama (no network after model pull)
export OLLAMA_HOST="http://localhost:11434"
perfxpert analyze -i trace.db --llm ollama --llm-model llama3:70b

# Bundled opencode (no credential — used internally by perfxpert-code)
perfxpert analyze -i trace.db --llm opencode
```

Model selection ladder (first hit wins, resolved at session boot):

1. `PERFXPERT_AGENTS_MODEL_<PROVIDER>` — per-provider pin, e.g.
   `PERFXPERT_AGENTS_MODEL_OPENAI=gpt-4o-mini`
2. `PERFXPERT_LLM_MODEL` — cross-provider override
3. Built-in default (anthropic: `claude-sonnet-4-20250514`, openai:
   `gpt-4o-mini`)

### Fallback chain (recommended for interactive use)

`PERFXPERT_LLM_FALLBACK_CHAIN` (new in cycle-2) cascades to the next
provider when the primary raises `RateLimitError`. Survives most
transient 429s without forcing a rerun:

```bash
# SKIP-SAMPLE — requires live credentials for each provider in the chain
export PERFXPERT_LLM_FALLBACK_CHAIN="openai,anthropic"
perfxpert analyze -i trace.db --llm openai
```

`PERFXPERT_DISABLE_RATE_LIMIT_RETRY=1` short-circuits client-side
retry so the fallback chain takes over on the first 429. See
`../guides/agentic-mode.md` for the full provider-resolution contract
and the recursion-guard rules.

## 11. Interactive sessions via `perfxpert-code`

One-shot non-interactive:

```bash
# SKIP-SAMPLE — requires bundled opencode + an LLM credential
perfxpert-code run -m anthropic/claude-haiku-4-5 "optimize ./app.cpp"
```

Fully interactive (drop into the TUI):

```bash
# SKIP-SAMPLE — requires the bundled opencode binary on PATH
perfxpert-code
```

### Tool-priority gate + lift semantics

The bundled opencode enforces a two-stage gate for every
GPU-performance request:

1. **Gate stage.** The LLM MUST call `perfxpert_intent_classify`
   first, then `perfxpert_workflow_next_step`. Calls to `bash`,
   `edit`, `write`, `read`, `glob`, `grep` BEFORE those two are
   refused by the system-prompt discipline (patches 0010, 0012-0017;
   applied to all 8 prompt families so Anthropic / GPT / Gemini /
   Kimi / Trinity / Codex / Beast / default all get identical
   framing).
2. **Lift stage.** Once `perfxpert_workflow_next_step` returns a phase
   (`profile` / `optimize` / `reprofile` / `analyze` / `build`), the
   gate is LIFTED. The agent then MUST use `bash` to run the profiler
   (rocprofv3 / rocprof-compute), `edit` / `write` to apply the patch,
   and `bash` to rebuild. Refusing to execute is the wrong move — the
   workflow asked for it.

Session state is auto-saved. List / resume with `perfxpert-code
session list` and `perfxpert-code session <id>` (pass-through to the
underlying opencode `session` subcommand).

## 12. Connecting other MCP clients

Any MCP-compatible client can consume the 41 READ-ONLY tools exposed
by `perfxpert-mcp` (7 agent-hierarchy entry points + 34
classifier/knowledge tools). Configuration snippets for Claude Desktop,
Claude Code, Codex CLI, Gemini CLI, and generic stdio clients live in
`../integration/mcp-server.md` under §"Client integration". The
bundled opencode inside `perfxpert-code` wires `perfxpert-mcp`
automatically — no client-side setup required.

## 13. Troubleshooting

**`perfxpert-code` launches the unpatched upstream binary.**
Check that `PERFXPERT_OPENCODE_PATH` is either unset or points at the
bundled copy (usually under
`experimental/python/perfxpert/perfxpert/_bundled/opencode`). If it's
unset and you still get the upstream binary, re-run
`bash experimental/python/perfxpert/scripts/apply-opencode-patches.sh`
and rebuild the bundled opencode. The launcher warns on fallback.

**LLM quota exhausted (429 / `insufficient_quota`).**
Set `PERFXPERT_LLM_FALLBACK_CHAIN` to a comma-separated provider
list, e.g. `export PERFXPERT_LLM_FALLBACK_CHAIN="openai,anthropic"`.
Combine with `PERFXPERT_DISABLE_RATE_LIMIT_RETRY=1` to skip the
client-side retry entirely and fall through on first 429.

**PMC counter collection fails with a per-block limit error.**
`FETCH_SIZE` and `WRITE_SIZE` each consume the full TCC budget — each
MUST own its own `--pmc` pass, alone. Any other TCC-derived counter
combined with either of those two will exceed the hardware limit.
Consult `perfxpert.counter_list_by_block` / `counter_lookup_info`
before emitting a counter set.

**`opencode binary not found` on `perfxpert doctor`.**
Either run the patch script (§1 step 2) or install upstream opencode
(`curl -fsSL https://opencode.ai/install | bash`) and retry. The
launcher searches `$PERFXPERT_OPENCODE_PATH` → bundled wheel →
`~/.opencode/bin/opencode` → `~/.local/bin/opencode` → `PATH`.

**`claude mcp add perfxpert perfxpert-mcp` not finding the binary.**
The client's `PATH` is narrower than your login shell. Use the
absolute path returned by `which perfxpert-mcp` when registering the
server.

## 14. Next steps

- `../architecture/agent-hierarchy.md` — 7-agent tier map + fence
  slices
- `../architecture/gate-cascade.md` — 5-gate correctness/regression
  middleware
- `../integration/mcp-server.md` — full MCP tool list + client
  integration
- `../guides/agentic-mode.md` — air-gap vs LLM + provider ladder +
  fallback chain
- `../guides/python-api.md` — `perfxpert.api` (1:1 mirror of the 7
  agent MCP tools) for embedding PerfXpert's analysis brain in your
  own tooling

<!-- footer marker: perfxpert v0.2.0 — getting-started.md audit 2026-04-20 -->
<!-- The footer marker above is moved in lockstep with the version bump in
     pyproject.toml / perfxpert/__init__.py / CMakeLists.txt (see
     CHANGELOG.md under [0.2.0]). Do not remove without bumping the
     corresponding version entries. -->
