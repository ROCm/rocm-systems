# MCP Server (`perfxpert-mcp`)

**If you use plain `perfxpert-code` or `perfxpert-code claude|codex|gemini`,
the MCP registration is handled automatically — see
[../guides/backends.md](../guides/backends.md). The manual integration
snippets below remain for users running `claude`, `codex`, or `gemini`
directly.**

PerfXpert ships a stdio-transport MCP server (`perfxpert-mcp`) that
re-exposes every `READ_ONLY` tool in `perfxpert.tools.*` to any
MCP-compatible client (opencode, Claude Desktop, Cursor, etc.). The
server never exposes `EXECUTION` tools; that split is enforced at
startup and by CI (`tests/test_integration/test_mcp_exposure.py`).

Cross-links:
- [Agent hierarchy](../architecture/agent-hierarchy.md) — what the tools
  are used for inside PerfXpert itself
- [Gate cascade](../architecture/gate-cascade.md) — the correctness
  guarantees MCP clients inherit for free when they call agent tools

## Starting the server

```bash
# SKIP-SAMPLE — server is intended to be spawned by an MCP client, not run directly
perfxpert-mcp
```

The entry point is registered in `pyproject.toml` (`perfxpert-mcp`).
Current wheels/install-from-source builds already depend on the Python
`mcp` package, so no separate extra is required:

```bash
# SKIP-SAMPLE — pip install is in the destructive-skip list
pip install perfxpert
```

Under `PERFXPERT_AIRGAP=1`, the server still serves cached READ_ONLY
tools — they're pure-Python lookups against knowledge YAMLs and
structured DB queries and don't need network.

## Tool classes (READ_ONLY vs EXECUTION)

Every tool function is tagged with one of two classes via
`@tool_class(ToolClass.READ_ONLY | ToolClass.EXECUTION)`:

| Class | Purpose | Examples | MCP? |
|-------|---------|----------|------|
| READ_ONLY | Pure queries against fixed artifacts / knowledge | `bottleneck.classify_from_metrics`, `counters.lookup_info`, `regression.extract_kernel_runtimes_from_db` | Yes |
| EXECUTION | Side-effecting (compile, profile, patch, anchor) | `compile.build`, `profile.run`, `patch.apply` | **No** |

The `READ_ONLY` / `EXECUTION` split is the core of spec §5.8: an
external client that connects to PerfXpert cannot cause side effects
even if its prompt asks it to; the agent would need to drive an
EXECUTION tool, and those stay in-process.

## Naming convention (dot→underscore on the wire)

Internally, PerfXpert tool names use a dotted namespace
(`bottleneck.classify_from_metrics`). The MCP specification disallows
dots in tool names, so the server rewrites every dotted internal name
to an underscored wire name at registration time (see
`mcp_server/server.py:69`, `name=name.replace(".", "_")`).

Summary:

- **Internal / Python API:** dotted — `bottleneck.classify_from_metrics`.
- **MCP wire (`tools/list` response, `tools/call` request):** underscored
  — `bottleneck_classify_from_metrics`.

The server handles the reverse mapping on `tools/call` (it accepts both
forms and looks the function up in its dotted registry), but external
clients should always send the underscored form for forward
compatibility. The tools-list snapshot in the next section is shown
with **dotted** names because that's what `discover_read_only_tools()`
returns; the equivalent wire names have the dots replaced by
underscores.

## Tools exposed (56)

Auto-discovered by `mcp_server._registry.discover_read_only_tools()`
every boot. The registry walks `perfxpert.tools.*` modules but skips
a small private `_`-prefixed skip list (currently `_class`; hardcoded
in `_registry.py:21`) because those entries are machinery, not
tools. Current set (enumerate yourself at any time by running the
Python snippet below):

```python
# Print every READ_ONLY tool the MCP server exposes at boot time.
from mcp_server._registry import discover_read_only_tools
for name in sorted(discover_read_only_tools()):
    print(name)
```

### Snapshot: agent-hierarchy tools (8)

These are the eight agent entry points. Each is a READ_ONLY MCP tool
AND a 1:1-mirrored callable on [`perfxpert.api`](../guides/python-api.md)
— invoking `mcp__perfxpert__agent_root` from a client and calling
`perfxpert.api.agent_root(...)` from Python produce the same output.

```
agents.root.agent_root
agents.analysis.agent_analysis
agents.recommendation.agent_recommendation
agents.correctness.agent_correctness
agents.compute.agent_compute_specialist
agents.memory.agent_memory_specialist
agents.latency.agent_latency_specialist
agents.diff.agent_diff_specialist
```

On the MCP wire these become `perfxpert_agent_root`,
`perfxpert_agent_analysis`, … (dot→underscore, see §"Naming
convention"). Input/output schemas for each agent are defined in
`perfxpert/agents/schemas.py` — see
[../guides/python-api.md](../guides/python-api.md) for field-level
examples.

`perfxpert_agent_diff_specialist` is the 8th agent (Confluence row
#7 follow-on): it wraps `trace_diff.diff_runs` + `regression.compare_runs`
+ `roofline.classify` and returns a structured run-to-run verdict
(`improved` / `regressed` / `neutral`) + per-kernel deltas + narrative.
Call it conversationally from any TUI backend ("diff this run against
baseline.db", "what got slower since yesterday's trace?") instead of
running `analyze` twice.

### Snapshot: classifier / knowledge tools (48)

Lower-level building blocks the agents themselves compose. External
clients can call these directly when they want the raw classifier
output without routing through the agent hierarchy.

```
arch.lookup_peaks
att.classify_stall_ratio
att.classify_stall_reason
bottleneck.classify_from_metrics
bottleneck.lookup_signatures
bottleneck.prioritize_by_amdahl
compiler.explain_flag
compiler.lookup_flags
counters.lookup_info
counters.validate_for_gpu
dependency_graph.reconstruct_dag            # DAG critical-path / bubble finder
gpu_runtime_monitor.analyze_thermal         # Thermal envelope
gpu_runtime_monitor.parse_amd_smi_json      # amd-smi monitor log parser
gpu_runtime_monitor.parse_rocm_smi_json     # rocm-smi log parser
intent.classify
interconnect.lookup_peaks
kernel_fusion.find_fusion_candidates        # Adjacent-short-kernel fusion
memory.classify_cache_performance
metrics.compute_gpu_utilization
metrics.compute_hbm_bandwidth
metrics.compute_l1_miss_rate
metrics.compute_l2_hit_rate
metrics.compute_latency
occupancy.lookup_waves_per_eu
occupancy.suggest_vgpr_reduction
plateau.check
pragma.explain_pragma
pragma.lookup_pragmas
pragma.suggest_pragmas_for_kernel
predict_impact.explain_prediction
predict_impact.list_supported_changes
predict_impact.predict_change_impact
profiling.fill_gap
rccl_analysis.analyze_collectives
regression.compare_runs
regression.extract_kernel_runtimes_from_db
regression.identify_hot_kernels
roofline.classify
roofline.plot_points                        # Live-roofline per-kernel points
sol.classify_utilization
sol.sanity_check
topdown.classify_overhead
trace_diff.diff_runs
trace_fingerprint.fingerprint
tracelens.classify_overhead
tracelens.lookup_metrics
unified_memory.analyze_paging               # MI300X paging / XCD penalty
workflow.next_step
```

`trace_diff.diff_runs` is the newest READ_ONLY tool (Confluence row
#7): compares two rocpd databases and returns a schema-0.3.1 diff dict.
Same engine powers the `perfxpert diff` + `perfxpert ci` CLI
subcommands, the `perfxpert analyze --baseline <db>` splice, and the
gate-cascade `trace_diff_regression_rule` — one brain, one number.

The `pragma.*` trio (advanced-recommendations tier): `pragma.lookup_pragmas`
enumerates the 3 allowlisted LLVM loop-hint pragmas (+ 7 rejected
entries for fence visibility), `pragma.explain_pragma` returns the
full catalog entry for a given pragma id, and
`pragma.suggest_pragmas_for_kernel` applies the Amdahl gate + trigger
rules to return 0-N candidates for a given kernel. Rendering of
pragma recs in `perfxpert analyze` output is gated behind the
`--advanced` CLI flag (or `PERFXPERT_ADVANCED_RECS=1` env var) — see
the getting-started guide "Advanced recommendations" section.

Advanced-specialist additions (+13 tools over the prior 43-tool
baseline, of which 8 were agent-hierarchy and 35 were classifier /
knowledge at that time):

- `kernel_fusion.find_fusion_candidates` (Feature A) — scans the kernel
  timeline for adjacent short kernels (< 10 us each, gap <= 500 ns
  default) with matching tensor-shape signatures and returns a ranked
  list of fusion candidates with `(est_speedup_lo, est_speedup_hi)`
  brackets. Bound into the Compute Specialist allowlist.
- `gpu_runtime_monitor.parse_amd_smi_json` / `parse_rocm_smi_json` /
  `analyze_thermal` (Feature B) — ingest a pre-captured `amd-smi
  monitor --json` or `rocm-smi --json` log (set
  `PERFXPERT_GPU_MONITOR_LOG=<path>`) and return a thermal envelope +
  throttle-event summary. Available via the MCP surface only — not
  bound to any specialist allowlist because thermal analysis is
  diagnostic / out-of-band.
- `unified_memory.analyze_paging` (Feature C) — detects CPU-resident
  GPU-accessed pages (HtoD/DtoH spikes + page-fault events) and
  quantifies MI300X cross-die (XCD-to-XCD) traffic totals. Bound into
  the Memory Specialist allowlist.
- `dependency_graph.reconstruct_dag` (Feature D) — builds a coarse DAG
  from stream-local kernel ordering + sync events, then returns
  `critical_path`, `bubbles`, `total_bubble_ns`, and `sync_event_count`
  so the Latency Specialist can distinguish over-synchronisation from
  inherent dependencies. Bound into the Latency Specialist allowlist.
- `predict_impact.predict_change_impact` / `list_supported_changes` /
  `explain_prediction` (Feature E — Change-Impact Prediction) — for a
  given baseline DB + kernel + named change_type, return a
  conservative speedup bracket (`hi × 0.85`), confidence, rationale,
  and `source_citation` back to the seed entry in
  `knowledge/proven_optimizations.yaml`. Seeded with 5 techniques
  (`vgpr_reduction`, `lds_tiling`, `mfma_enablement`, `fast_math_flag`,
  `hip_stream_overlap`); Amdahl (< 5%) + tier-2 (no counters) gates
  enforced internally. Bound into the Memory Specialist allowlist; the
  Compute + Latency Specialists attach predictions via a post-hook
  helper after the agent tool loop returns. Rendering on rec cards is
  always-on (no CLI gate) and bumps the JSON `schema_version` to
  `0.3.3`.

Features F-H (matrix meter, attention scope, live roofline) did not
add new MCP tools — they extend specialist fences + pipeline with
signatures drawn from existing `metrics.*` + `tracelens_port` +
`roofline.plot_points` helpers.

### Live Roofline (+1 tool)

The Live Roofline work adds one more READ_ONLY tool on top of
the advanced-specialist additions above:

- `roofline.plot_points` — reads a rocpd database's `pmc_events` view,
  aggregates per-kernel `SQ_INSTS_VALU`, the rocprofv3 v1.2.1
  `SQ_INSTS_VALU_MFMA_MOPS_*` counters when present (falling back to
  legacy `SQ_INSTS_VALU_MFMA`), plus `FETCH_SIZE` / `WRITE_SIZE`, and
  returns a per-kernel `(ai,
  achieved_flops_per_s, bottleneck_class)` list plus the arch ridge
  point. The webview formatter uses this payload to draw the inline
  SVG roofline (no external JS / CSS). Also surfaced in the JSON
  format under the top-level `roofline` key, bumping `schema_version`
  to `0.3.4`. See `perfxpert/tools/roofline.py::plot_points` and
  the python-api guide entry for the full shape.

## Protocol examples

### `tools/list`

JSON-RPC request:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

Response (truncated; names are **underscored** on the wire — see the
"Naming convention" section above):

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "bottleneck_classify_from_metrics",
        "description": "Classify bottleneck from raw hardware metrics.",
        "inputSchema": {
          "type": "object",
          "properties": {"metrics": {"type": "object"}},
          "required": ["metrics"]
        }
      },
      {
        "name": "regression_extract_kernel_runtimes_from_db",
        "description": "Extract per-kernel runtimes from a rocpd .db file.",
        "inputSchema": {
          "type": "object",
          "properties": {"db_path": {"type": "string"}},
          "required": ["db_path"]
        }
      }
    ]
  }
}
```

### `tools/call`

JSON-RPC request:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "bottleneck_classify_from_metrics",
    "arguments": {
      "metrics": {
        "gpu_utilization_percent": 94,
        "memory_controller_percent_busy": 22
      }
    }
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"class\": \"compute_bound\", \"confidence\": 0.87}"
      }
    ]
  }
}
```

### Worked example: external client driving `get_hotspots`

An external MCP client (e.g. Claude Desktop configured via
`claude_desktop_config.json`) can compose the READ_ONLY tools to
reproduce PerfXpert's hotspot view without ever importing the Python
package. The sequence (using on-the-wire underscored names) is:

1. `tools/call` → `regression_extract_kernel_runtimes_from_db` with
   `db_path=/tmp/trace.db`
2. `tools/call` → `regression_identify_hot_kernels` with the runtimes
   returned in step 1
3. `tools/call` → `bottleneck_classify_from_metrics` per hot kernel
4. `tools/call` → `bottleneck_prioritize_by_amdahl` to rank results

This sequence mirrors what the in-process Analysis agent runs during
`session.run_root(...)` — external clients get the same answer because
the underlying tools are identical (the agent calls the dotted
in-process names; MCP clients call the underscored wire names; both
resolve to the same Python function).

## Adding a tool to the MCP surface

See [contributing/mcp_tools.md](../contributing/mcp_tools.md) for the
full procedure. The short version:

1. Decorate the function with `@tool_class(ToolClass.READ_ONLY)`.
2. Keep side effects out — the CI exposure test will reject the PR
   otherwise.
3. Add a unit test under `tests/test_mcp/test_<name>.py`.

Adding an EXECUTION tool requires an RFC; the exposure invariant is
load-bearing for the security posture in spec §5.8.

## Client integration

`perfxpert-mcp` speaks stdio MCP (JSON-RPC, protocol `2024-11-05`), so
any MCP-compatible client can consume the 56 READ_ONLY tools. The
`command` field in every example below must resolve on the client's
`PATH` — run `which perfxpert-mcp` to get an absolute path if your
client launches with a narrower env than your login shell.

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows) and
add a `perfxpert` entry under `mcpServers`:

```json
// SKIP-SAMPLE — client config, not executable
{
  "mcpServers": {
    "perfxpert": {
      "command": "perfxpert-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

Restart Claude Desktop. The 56 tools appear under the 🔌 panel with
`perfxpert_` name prefixes (underscored-on-the-wire — see §"Naming
convention").

### Claude Code (`claude` CLI)

Register the server with the `claude mcp add` command:

```bash
# SKIP-SAMPLE — requires a live claude CLI install
claude mcp add perfxpert perfxpert-mcp
```

This writes to `~/.claude.json`. Verify with `claude mcp list`; the
tools become available in any subsequent `claude` session under the
`perfxpert_*` namespace.

### Codex CLI (OpenAI)

For a trusted project, add `[mcp_servers.perfxpert]` to
`<cwd>/.codex/config.toml`. If project scope is unavailable, the
fallback location is `~/.codex/config.toml`. Trust itself still lives
in `~/.codex/config.toml` under `[projects."<abs-cwd>"]`.

```toml
# SKIP-SAMPLE — client config, not executable
[mcp_servers.perfxpert]
command = "perfxpert-mcp"
args = []
```

Codex 0.7+ auto-discovers MCP servers from this file on startup. In
an interactive session, `/mcp` lists the loaded servers and `/tools`
shows the exposed tools.

### Gemini CLI (Google)

`perfxpert-code gemini` installs MCP wiring into the **project-local**
`.gemini/settings.json` so hooks and prompt state stay scoped to the
current checkout. Add the server under `mcpServers`:

```json
// SKIP-SAMPLE — client config, not executable
{
  "mcpServers": {
    "perfxpert": {
      "command": "perfxpert-mcp",
      "args": [],
      "env": {},
      "timeout": 30000
    }
  }
}
```

Verify with `gemini mcp list` (Gemini CLI ≥ 0.2). For project-local
installs this probe is advisory: PerfXpert treats
`.gemini/settings.json` as the source of truth and does not fail setup
just because `gemini mcp list` omits the project MCP. Any tool named
`mcp_perfxpert_*` is exposed; unprefixed names are reserved for built-in
Gemini capabilities.

Older perfxpert releases may have left user-global entries in
`~/.gemini/settings.json`; current installs do not mutate that file.

### opencode (bundled with `perfxpert-code`)

No manual step required — the bundled `opencode.json` already wires
`perfxpert-mcp` as a local MCP server:

```json
// SKIP-SAMPLE — bundled config shown for reference
{
  "mcp": {
    "perfxpert": {
      "type": "local",
      "command": ["perfxpert-mcp"],
      "enabled": true
    }
  }
}
```

If you run the upstream `opencode` binary directly (outside
`perfxpert-code`), copy the `opencode.json` from
`~/.cache/perfxpert/opencode/opencode.json` into your project or into
`~/.config/opencode/opencode.json` so `perfxpert-mcp` is picked up.

### Generic MCP clients (stdio JSON-RPC)

Any client that spawns stdio subprocesses and speaks MCP can consume
`perfxpert-mcp` directly. Minimum spawn spec:

| Field | Value |
|-------|-------|
| command | `perfxpert-mcp` |
| args | `[]` |
| protocol | MCP stdio JSON-RPC, `2024-11-05` |
| auth | none (READ_ONLY, no mutations) |
| working dir | inherited from caller (server writes nothing) |
| env needed | none (inherits caller's `PATH` to find `rocprofv3` if queried) |

Send `initialize` → `tools/list` → `tools/call` in that order. See the
`Protocol examples` section above for the exact JSON payloads.

### Troubleshooting

- **"command not found: perfxpert-mcp"** — the client's `PATH` is
  narrower than your login shell. Use the absolute path returned by
  `which perfxpert-mcp` in the `command` field.
- **Tools list empty** — verify the server started correctly by
  running `perfxpert-mcp </dev/null 2>&1 | head` manually; you should
  see "discovered N read-only tools".
- **"unknown tool: bottleneck.classify_from_metrics"** — you sent the
  dotted in-process name. External clients must use the underscored
  wire name (`bottleneck_classify_from_metrics`). See §"Naming
  convention".
- **EXECUTION tools missing** — by design. The MCP surface is
  READ_ONLY-only per spec §5.8. If you need to run a profiler from a
  client, use that client's own shell tool (opencode's `bash`, Claude
  Desktop's `run_shell_command`, etc.), NOT an MCP call.

## Launching via `perfxpert-code <backend>` (multi-backend dispatch)

The `perfxpert-code` launcher supports three third-party agent
backends in addition to the default patched opencode path. Each
subcommand
installs perfxpert MCP + AGENTS.md plus the backend's gate layer,
then execs the backend binary for an interactive
session:

| Subcommand | Backend | Config file written | Tool-name wire format |
|---|---|---|---|
| `perfxpert-code claude`  | Claude Code | `.mcp.json` + `CLAUDE.local.md` + `.claude/settings.json` | `mcp__perfxpert__<tool>` |
| `perfxpert-code gemini`  | Gemini CLI  | `.gemini/settings.json` (project-local list-append, never touches `GEMINI.md`) | `mcp_perfxpert_<tool>` |
| `perfxpert-code codex`   | Codex CLI   | `~/.codex/config.toml` (trust) + `<cwd>/.codex/config.toml` when trusted or fallback `~/.codex/config.toml` for MCP + `AGENTS.override.md` | `mcp_perfxpert_<tool>` |

All three subcommands accept the same dispatcher-owned flags:

- `--dry-run` — run plan(), print intended actions, write nothing,
  skip the backend spawn. Useful for code review.
- `--quiet` — suppress banner + per-step progress.
- `--force` — bypass the recursion guard
  (`PERFXPERT_IN_AGENT_SESSION`) and any clobber refusals.
- `--allow-agents-md-append` — opt-in for merging into a
  git-tracked `CLAUDE.md` / `AGENTS.md` (default: refuse, stage at
  `CLAUDE.local.md` / dedicated pointer instead — I3).

Example:

```bash
# SKIP-SAMPLE — requires the claude CLI installed and on PATH
perfxpert-code claude --dry-run "analyze this trace"
perfxpert-code claude "analyze this trace"
```

### First-run consent

On first install for a given `(backend, cwd, file-set)` triple,
the launcher lists every planned file edit and asks `[y/N]`. Set
`PERFXPERT_ASSUME_CONSENT=1` to auto-grant (CI / automated
bootstrapping). Consent invalidates automatically when the
file set changes (e.g. the user newly tracks `CLAUDE.md` in
git — the security impact of "yes" changed).

### Tool-priority gate (event-based)

Each backend receives a gate layer that rejects any
non-PerfXpert-prefixed tool call UNTIL
`perfxpert_intent_classify` has returned in the current session.
There is no turn counter — a legitimate `bash` on turn 2 AFTER
`intent_classify` on turn 1 passes through. Gate escape hatch:
`export PERFXPERT_GATE_HOOK=0`.

Surfaces per backend:

- **Claude Code** — native `PreToolUse` hook inside
  `.claude/settings.json` with `permissionDecision: deny` +
  `permissionDecisionReason`. The decision record lives locally in
  the contributor's working copy.
- **Gemini** — native `BeforeTool` / `AfterTool` hook entries in
  `.gemini/settings.json` + runtime sidecar at
  `.gemini/runtime/perfxpert-gate-<session_id>.json`.
- **Codex** — prompt-layer-only (rejection-language stanza in the
  perfxpert-managed `AGENTS.override.md` compatibility file). Codex's
  native `PreToolUse` hook currently intercepts Bash only, not MCP /
  Write / other tools, so it cannot satisfy the event-based gate
  contract. Rationale is captured in the local Codex hook-surface
  decision record.
- **opencode** — `{block, retryWith}` from patched
  `tool.execute.before` plugin (fork-only — patch 0020).

### Known issue: first-run exit-124

Sporadically the first `perfxpert-code <backend>` invocation
after a stale server state hangs with an MCP handshake timeout.
The launcher mitigates this in three layers:

1. **Warmup during `install()`** — boots `perfxpert-mcp` once to
   prime sqlite + tool registry. Bounded by
   `PERFXPERT_MCP_WARMUP_TIMEOUT_S` (default `10`).
   Disable via `PERFXPERT_MCP_WARMUP=0`.
2. **Handshake retry in `verify_mcp_live()`** — 3 attempts with
   exponential backoff 2s / 4s / 8s, early-exit on
   `PERFXPERT_MCP_RETRY_BUDGET_S` (default `6`).
3. **Graceful degradation** — warmup failure is logged but
   non-fatal; install continues.

### `perfxpert-code install-patches` (deprecated)

The `install-patches` subcommand is kept as a backward-compat
alias for the patched opencode build step. It now prints a
yellow-stderr deprecation notice. Silence with
`export PERFXPERT_SILENCE_DEPRECATION=1`.

### `.gitignore` hints

Project-scope MCP config lives in `.mcp.json`; if a `.gitignore`
is present, `perfxpert-code claude install` appends `.mcp.json`
to it so multi-collaborator repos don't leak per-user paths. No
action taken when `.gitignore` is absent (installer won't
create one).

### Platform boundary

POSIX only (Linux + macOS + WSL). Windows-native is explicitly
out of scope; a tracking issue follows.
