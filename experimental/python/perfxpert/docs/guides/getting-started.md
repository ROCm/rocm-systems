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

PerfXpert is a pure-Python package plus a bundled AMD-themed opencode
binary. The install is two steps: pip-install the Python wheel, then
build the bundled patched opencode (one-time, requires `bun`).

### Step 1: pip install

```bash
# SKIP-SAMPLE — destructive pattern filtered by scripts/test-samples.py
pip install -e "experimental/python/perfxpert[all]"
```

The `[all]` extra pulls in `anthropic`, `openai`, `rich`, and
`claude-agent-sdk` so every LLM provider works out of the box. Omit it
if you only want deterministic air-gap analysis.

### Step 2: build the bundled patched opencode (one-time)

The interactive TUI (`perfxpert-code`) wraps upstream opencode with 17
AMD patches: branding, the AMD red color palette, the 7-agent prompt
preamble, and the STRICT-TOOL-DISCIPLINE stanza that forces
`perfxpert_intent_classify` → `perfxpert_workflow_next_step` as the
first two tool calls. Apply them with:

```bash
# SKIP-SAMPLE — requires the opencode submodule and bun toolchain
cd experimental/python/perfxpert
git submodule update --init --recursive
bash scripts/apply-opencode-patches.sh
```

See `.patches/README.md` for the patch catalogue. Skip this step and
`perfxpert-code` will still launch — it falls back to whatever
`opencode` is on `PATH` — but you will NOT get the PerfXpert system
prompt or the tool-priority gate.

## 2. Verify

```bash
# SKIP-SAMPLE — requires perfxpert on PATH from the pip install above
perfxpert doctor
```

Expected output ends with `ALL CLEAN` when everything is wired. The
doctor checks:

- `perfxpert` version + Python ≥ 3.10
- openai-agents SDK
- MCP server reachable (`perfxpert-mcp` boots + 34 tools registered)
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
  with `--llm {anthropic,openai,claude-code}`.
- **`perfxpert-mcp`** — stdio MCP server that re-exposes the 34
  READ-ONLY analysis tools over JSON-RPC. Meant to be spawned by an
  MCP client (Claude Desktop, Claude Code, Codex CLI, Gemini CLI,
  opencode). See `../integration/mcp-server.md`.
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

```bash
# SKIP-SAMPLE — requires a real trace.db and an LLM credential
export ANTHROPIC_API_KEY="sk-ant-..."
perfxpert analyze -i trace.db --llm anthropic
# or:
export OPENAI_API_KEY="sk-..."
perfxpert analyze -i trace.db --llm openai
# or (uses stored Claude Code credentials — no API key needed):
perfxpert analyze -i trace.db --llm claude-code
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
export PERFXPERT_LLM_FALLBACK_CHAIN="openai,anthropic,claude-code"
perfxpert analyze -i trace.db --llm openai
```

`PERFXPERT_DISABLE_RATE_LIMIT_RETRY=1` short-circuits client-side
retry so the fallback chain takes over on the first 429. See
`../guides/agentic-mode.md` for the full provider-resolution contract
and the recursion-guard rules.

## 7. Interactive sessions via `perfxpert-code`

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

## 8. Connecting other MCP clients

Any MCP-compatible client can consume the 34 READ-ONLY tools exposed
by `perfxpert-mcp`. Configuration snippets for Claude Desktop, Claude
Code, Codex CLI, Gemini CLI, and generic stdio clients live in
`../integration/mcp-server.md` under §"Client integration". The
bundled opencode inside `perfxpert-code` wires `perfxpert-mcp`
automatically — no client-side setup required.

## 9. Troubleshooting

**`perfxpert-code` launches the unpatched upstream binary.**
Check that `PERFXPERT_OPENCODE_PATH` is either unset or points at the
bundled copy (usually under
`experimental/python/perfxpert/perfxpert/_bundled/opencode`). If it's
unset and you still get the upstream binary, re-run
`bash experimental/python/perfxpert/scripts/apply-opencode-patches.sh`
and rebuild the bundled opencode. The launcher warns on fallback.

**LLM quota exhausted (429 / `insufficient_quota`).**
Set `PERFXPERT_LLM_FALLBACK_CHAIN` to a comma-separated provider
list, e.g. `export PERFXPERT_LLM_FALLBACK_CHAIN="openai,anthropic,claude-code"`.
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

## 10. Next steps

- `../architecture/agent-hierarchy.md` — 7-agent tier map + fence
  slices
- `../architecture/gate-cascade.md` — 5-gate correctness/regression
  middleware
- `../integration/mcp-server.md` — full MCP tool list + client
  integration
- `../guides/agentic-mode.md` — air-gap vs LLM + provider ladder +
  fallback chain
