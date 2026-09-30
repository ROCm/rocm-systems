# Python API (`perfxpert.api`)

`perfxpert.api` is a **1:1 mirror** of the 8 agent-hierarchy MCP
tools. Every callable here IS the same function the MCP server wraps
— the Python API and the MCP surface share a single implementation.
Use this module to embed PerfXpert's analysis brain in your own
tooling without running the MCP server.

Cross-links:

- [MCP server](../integration/mcp-server.md) — the same 8 agent tools
  + 48 classifier / knowledge tools re-exposed over stdio JSON-RPC.
  **This API is the same surface as the MCP tools.**
- [Agent hierarchy](../architecture/agent-hierarchy.md) — tier map,
  fence-slice pattern, and where each agent lives in source.
- [Agentic mode guide](agentic-mode.md) — air-gap vs LLM,
  `PERFXPERT_LLM_FALLBACK_CHAIN`, typed-error taxonomy.

## Exported callables

The mirror drops the `perfxpert_` prefix (which is only applied at
MCP registration):

| MCP tool name | Python API |
|---------------|------------|
| `perfxpert_agent_root` | `perfxpert.api.agent_root` |
| `perfxpert_agent_analysis` | `perfxpert.api.agent_analysis` |
| `perfxpert_agent_recommendation` | `perfxpert.api.agent_recommendation` |
| `perfxpert_agent_correctness` | `perfxpert.api.agent_correctness` |
| `perfxpert_agent_compute_specialist` | `perfxpert.api.agent_compute_specialist` |
| `perfxpert_agent_memory_specialist` | `perfxpert.api.agent_memory_specialist` |
| `perfxpert_agent_latency_specialist` | `perfxpert.api.agent_latency_specialist` |
| `perfxpert_agent_diff_specialist` | `perfxpert.api.agent_diff_specialist` |
| `perfxpert_trace_diff_diff_runs` | `perfxpert.api.trace_diff_diff_runs` |
| `perfxpert_rccl_analysis_analyze_collectives` | `perfxpert.tools.rccl_analysis.analyze_collectives` |
| `perfxpert_interconnect_lookup_peaks` | `perfxpert.tools.interconnect.lookup_peaks` |

Every agent callable honors `PERFXPERT_AIRGAP=1` and the full provider /
fallback-chain ladder (`PERFXPERT_LLM_FALLBACK_CHAIN`) because the
agent tools defer to `perfxpert.agents.runtime.build_session`.

`trace_diff_diff_runs` is deterministic and credential-free — it reads
two rocpd databases and produces a schema-0.3.1 diff dict (see
[schemas.md][schemas] for the full contract). This is the same engine
`perfxpert diff`, `perfxpert ci`, and `perfxpert analyze --baseline`
use.

The two communication-analysis tools are deterministic (no
LLM, no credentials) and map 1:1 onto their module functions:

```python
# SKIP-SAMPLE — illustrative RCCL analysis call (requires a rocpd .db)
from perfxpert.tools import rccl_analysis, interconnect

peaks = interconnect.lookup_peaks("gfx942")       # MI300X XGMI spec
peaks["achievable_gbps"]                           # -> 340.0

result = rccl_analysis.analyze_collectives(
    "merged.db", gfx_id="gfx942"
)
result["summary"]["dominant_op"]                   # -> "AllReduce"
result["summary"]["overlap_pct"]                   # comm/compute overlap %
```

[schemas]: ../contributing/schemas.md

## Quickstart

```python
# SKIP-SAMPLE — illustrative Python API call (requires real trace.db / running session)
from perfxpert import api

out = api.agent_root(
    database_path="trace.db",
    user_query="why slow?",
    airgap=True,
)
print(out["primary_bottleneck"])
print(out["narrative"])
for rec in out["recommendations"]:
    print(rec["name"], rec["title"])
```

`api.agent_root(...)` is the same call `perfxpert analyze` makes
under the hood — passing `database_path=<path>` + `user_query=<str>`
gives you the equivalent of the CLI's end-to-end report in a Python
dict.

### Hotspot output shape

Each entry in `out["hotspots"]` is a dict with `name`, `calls`,
`total_duration`, `avg_duration`, `min_duration`, `max_duration`,
and `percent_of_total`. When the caller supplied a `source_dir` to
the root agent, every hotspot **also** carries a
`source_locations: list[dict]` field:

```python
# SKIP-SAMPLE — illustrative output shape for hotspot source correlation
{
    "name": "ns::matmul<float>(float*, float*)",
    "calls": 128,
    "total_duration": 92_000_000,
    "percent_of_total": 48.5,
    "source_locations": [
        {"file": "src/ops.hip",  "line": 42, "kind": "definition"},
        {"file": "src/main.hip", "line": 88, "kind": "launch"},
    ],
}
```

`kind` is `"definition"` (the `__global__` symbol declaration) or
`"launch"` (a detected `hipLaunchKernelGGL` / triple-angle dispatch
site). The list is empty when the Tier-0 scanner ran but found no
matching basename, and the field is absent entirely when no source
scan was performed. The `source_locations` field is the same shape
as `hotspots[i].source_locations` in the JSON report (schema
`0.3.1`).

## Input schemas (one example per agent)

Field names come from `perfxpert/agents/schemas.py` (Pydantic
models; frozen to prevent mutation during handoff). Schemas cap at
≤10 input fields and ≤5 output fields per agent — CI-enforced in
`tests/test_agents/test_schema_field_caps.py`.

### `agent_root` — Layer-0 entry point

```python
# SKIP-SAMPLE — illustrative Python API call (requires real trace.db / running session)
out = api.agent_root(
    user_query="Propose the first optimization for my hot kernel.",
    database_path="/tmp/trace.db",        # optional
    source_dir="./my_app",                 # optional — Tier 0 source scan
    provider="anthropic",                  # anthropic|openai|ollama|private|opencode
    airgap=False,                          # True ⇒ deterministic path, no LLM
    session_id=None,                       # optional — for session persistence
    progress_callback=print,               # optional — live phase updates
)
```

Output: `narrative: str`, `recommendations: list[dict]`,
`primary_bottleneck: str`, `warnings: list[str]`, `metadata: dict`.

#### Live progress feedback

Every agent tool (`agent_root`, `agent_analysis`, `agent_recommendation`,
`agent_correctness`, `agent_compute_specialist`, `agent_memory_specialist`,
`agent_latency_specialist`, `agent_diff_specialist`) accepts an optional
`progress_callback: Callable[[str], None]`. When set, the runtime fires
short status strings as each agent phase enters / exits and when the
fallback chain cascades across providers — useful for driving a
spinner, streaming to a web UI, or piping into a log aggregator:

```python
# SKIP-SAMPLE — illustrative Python API call
events = []
out = api.agent_root(
    database_path="/tmp/trace.db",
    provider="anthropic",
    progress_callback=events.append,
)
# events == ["entering root", ..., "exit root"]
```

`progress_callback=None` (the default) is zero-overhead — the runtime
short-circuits all emission when no callback is registered. This is the
same mechanism the `perfxpert analyze` CLI uses to draw its Rich
spinner (see the Getting Started guide §6).

### `agent_analysis` — Layer-1 bottleneck classifier

```python
# SKIP-SAMPLE — illustrative Python API call (requires real trace.db / running session)
out = api.agent_analysis(
    input={
        "database_path": "/tmp/trace.db",
        "top_kernels": 10,
        "att_dir": None,                   # optional — auto-detected
    },
    airgap=True,
)
```

Output: `primary_bottleneck`, `confidence`, `time_breakdown`,
`hot_kernels`, `counter_data_available`.

### `agent_recommendation` — Layer-1 technique proposer

```python
# SKIP-SAMPLE — illustrative Python API call (requires real trace.db / running session)
findings = api.agent_analysis(
    input={"database_path": "/tmp/trace.db"},
    airgap=True,
)

out = api.agent_recommendation(
    input={
        "findings": findings,
        "kernel_filter": None,
        "edit_history": [],
        "seen_recommendation_hashes": [],
    },
    airgap=True,
)
```

Output: `recommendations` (flat, ranked, deduplicated),
`specialist_used` (`compute`/`memory`/`latency`/`none`),
`plateau_detected`.

### `agent_correctness` — Layer-1 gate-verdict narrator

```python
# SKIP-SAMPLE — illustrative Python API call (requires real trace.db / running session)
# gate_verdict is a dict mirroring runtime.gate_cascade.GateVerdict.
out = api.agent_correctness(
    input={
        "gate_verdict": {
            "status": "reject",
            "failing_gate": "sol",
            "detail": "L2 hit rate regressed from 72% to 38%",
            "delta_pct": -47.2,
            "metrics": {},
            "rejected_patch_sha": "abc123",
            "per_kernel_deltas": {"gemm_k0": -12.4},
        },
        "kernel_name": "gemm_k0",
        "last_technique": "lds_blocking",
        "edit_history": [],
    },
    airgap=True,
)
```

Output: `verdict` (`pass`/`reject`/`regressed`), `action`
(`accept`/`revert`/`reject_and_log`), `narrative`,
`alternative_technique`, `follow_up_task_id`.

### `agent_compute_specialist` — Layer-2 compute-bound expert

```python
# SKIP-SAMPLE — illustrative Python API call (requires real trace.db / running session)
out = api.agent_compute_specialist(
    input={
        "gfx_id": "gfx942",
        "hot_kernels": [{"name": "gemm_k0", "duration_ns": 12000000}],
        "counter_data": {"vgpr_used": 128, "waves_per_eu": 4},
        "source_hints": {},
    },
    airgap=True,
)
```

Output: `techniques` (list of `{name, rationale, expected_impact,
effort, risk}`), `confidence`, `citations`.

### `agent_memory_specialist` — Layer-2 memory-bound expert

```python
# SKIP-SAMPLE — illustrative Python API call (requires real trace.db / running session)
out = api.agent_memory_specialist(
    input={
        "gfx_id": "gfx942",
        "hot_kernels": [{"name": "stream_copy", "duration_ns": 8000000}],
        "memcpy_data": {"host_to_device_bytes": 1024 * 1024 * 512},
        "counter_data": {"l2_hit_rate": 0.41},
    },
    airgap=True,
)
```

Output: same shape as compute specialist — `techniques`,
`confidence`, `citations`.

### `agent_latency_specialist` — Layer-2 launch / dependency-chain expert

```python
# SKIP-SAMPLE — illustrative Python API call (requires real trace.db / running session)
out = api.agent_latency_specialist(
    input={
        "gfx_id": "gfx942",
        "hot_kernels": [{"name": "launch_heavy_loop", "count": 10000}],
        "api_overhead_pct": 34.2,
        "avg_kernel_duration_us": 6.4,
    },
    airgap=True,
)
```

Output: same shape — `techniques`, `confidence`, `citations`.

### `agent_diff_specialist` — Layer-2 run-to-run diff narrator

Compares a baseline rocprofiler-sdk database against a new run and
returns a structured verdict (improved / regressed / neutral) plus
ranked per-kernel deltas and a prose narrative. Wraps
`trace_diff.diff_runs` internally — the arithmetic is deterministic;
the LLM (when attached) only rewrites the narrative.

```python
# SKIP-SAMPLE — illustrative Python API call (requires two real .db files)
out = api.agent_diff_specialist(
    baseline_db="baseline.db",
    new_db="new.db",
    top_kernels=20,
    user_intent="summarize the diff",
    airgap=True,
)
# {'wall_delta_pct': +12.3,
#  'regressions': [{'name': 'matmul', 'delta_pct': +34.2, ...}, ...],
#  'improvements': [...],
#  'verdict': 'regressed',
#  'narrative': 'The new run finished in +12.3% wall-time vs baseline. '
#               '4 kernels regressed, 1 improved. Top regression: matmul (+34.2%).',
#  'confidence': 0.7}
print(out["verdict"], out["narrative"])
for k in out["regressions"][:3]:
    print(f"  {k['name']}: {k['delta_pct']:+.1f}%")
```

Output keys: `wall_delta_pct`, `regressions`, `improvements`,
`verdict` ∈ {`improved`, `regressed`, `neutral`}, `narrative`,
`confidence` (0..1). Use this when the backend TUI LLM decides
"compare two runs" rather than "re-analyze twice" —
`agent_diff_specialist` is one tool call; running `agent_root`
twice is two.

### `trace_diff_diff_runs` — compare two rocpd databases

Deterministic, credential-free — no LLM required. Same engine as
`perfxpert diff`, `perfxpert ci`, and the `--baseline` splice.

```python
# SKIP-SAMPLE — illustrative Python API call (requires two real .db files)
from perfxpert import api

diff = api.trace_diff_diff_runs(
    baseline_db="baseline.db",
    new_db="new.db",
    top_kernels=20,
)
# schema_version == "0.3.1"
print(f"wall delta: {diff['wall_delta_pct']:+.2f}%")
for k in diff["primary_regressions"]:
    print(f"  REGRESSED {k['name']}: {k['delta_pct']:+.1f}%")
```

Output keys: `schema_version`, `baseline_db`, `new_db`, `wall_delta_ns`,
`wall_delta_pct`, `per_kernel`, `primary_regressions`,
`primary_improvements`, `narrative`. See
[schemas.md](../contributing/schemas.md) for the `trace_diff` contract.

### `roofline_plot_points` — per-kernel live-roofline points

Builds the `(arithmetic_intensity, achieved_flops_per_s)` points used
by the webview's **Live Roofline** chart. READ_ONLY against a rocpd
database — no network, no writes, no LLM. The same payload is embedded
under `"roofline"` in the agentic JSON output.

```python
# SKIP-SAMPLE — illustrative Python API call (requires a Tier-2 .db with
# SQ_INSTS_VALU + FETCH_SIZE + WRITE_SIZE counters).
from perfxpert.tools import roofline

rf = roofline.plot_points("trace.db", top_k=10)
# {'schema_version': '0.3.x',
#  'arch': 'gfx942',
#  'arch_peaks': {'fp32': 1.634e14, 'fp16': 1.307e15, ...},
#  'dtype': 'bf16',
#  'dtype_confidence': 'from_kernel_name',
#  'kernels': [{'name': 'gemm_bf16_kernel',
#               'ai': 48.2,
#               'achieved_flops_per_s': 1.12e14,
#               'bottleneck_class': 'compute',
#               'fp_type': 'bf16',
#               'confidence': 'high',
#               'duration_pct': 94.3}],
#  'ridge_point': {'ai': 30.8, 'flops_per_s': 1.634e14}}
for k in rf["kernels"]:
    print(f"{k['name']}: AI={k['ai']:.1f}, "
          f"{k['achieved_flops_per_s']/1e12:.1f} TF/s, {k['bottleneck_class']}")
```

Formula — deterministic, no LLM involvement:

```text
flops = SQ_INSTS_VALU × 64 + sum(SQ_INSTS_VALU_MFMA_MOPS_* × 512)
      # fallback when only the legacy aggregate counter is present:
      + SQ_INSTS_VALU_MFMA × mfma_flops_per_inst[dtype]
bytes = (FETCH_SIZE + WRITE_SIZE) × 1024       # TCC KiB → bytes
ai    = flops / bytes
rate  = flops / (duration_ns / 1e9)
```

Dtype detection is a regex over the demangled kernel name
(`_bf16`, `_fp16`, `_fp8`, `_int8`; falls back to `fp32` +
`dtype_confidence: "default"`). Pass `dtype_hint="bf16"` to force a
single dtype for every kernel — useful when your kernels don't follow
the naming convention. See
[schemas.md](../contributing/schemas.md#roofline-payload) for the full
contract.

## Error handling

All callables raise the standard provider taxonomy on LLM failure —
`AuthError`, `RateLimitError`, `QuotaExceededError`,
`TransientError`, `FatalError`, `TimeoutError`, and
`ProviderChainExhausted` when a `PERFXPERT_LLM_FALLBACK_CHAIN`
cascade runs out of candidates. Import from
`perfxpert.providers._exceptions`; see
[agentic-mode.md](agentic-mode.md) §"Fallback chain" for the full
table of which errors cascade vs surface immediately.

```python
# SKIP-SAMPLE — illustrative Python API call (requires real trace.db / running session)
from perfxpert import api
from perfxpert.providers._exceptions import (
    RateLimitError,
    QuotaExceededError,
    ProviderChainExhausted,
)

try:
    out = api.agent_root(
        database_path="trace.db",
        user_query="why slow?",
        provider="anthropic",
    )
except QuotaExceededError:
    # Retry is futile — fall through to airgap.
    out = api.agent_root(
        database_path="trace.db",
        user_query="why slow?",
        airgap=True,
    )
```

## Advanced-specialist classifier / knowledge tools

These are raw callables (not agent entry points) that specialists invoke
internally; each is also exposed over the MCP wire.

| Tool | Python import | Role |
|------|---------------|------|
| `kernel_fusion.find_fusion_candidates` | `from perfxpert.tools.kernel_fusion import find_fusion_candidates` | Adjacent-short-kernel fusion candidates — ranked list with `(est_speedup_lo, est_speedup_hi)` brackets. |
| `gpu_runtime_monitor.parse_amd_smi_json` | `from perfxpert.tools.gpu_runtime_monitor import parse_amd_smi_json` | Parse a captured `amd-smi monitor --json` log. |
| `gpu_runtime_monitor.parse_rocm_smi_json` | `from perfxpert.tools.gpu_runtime_monitor import parse_rocm_smi_json` | Parse a captured `rocm-smi --json` log. |
| `gpu_runtime_monitor.analyze_thermal` | `from perfxpert.tools.gpu_runtime_monitor import analyze_thermal` | Thermal envelope + throttle-event summary over a captured log. |
| `unified_memory.analyze_paging` | `from perfxpert.tools.unified_memory import analyze_paging` | HtoD/DtoH paging + MI300X cross-die (XCD) penalty analysis. |
| `dependency_graph.reconstruct_dag` | `from perfxpert.tools.dependency_graph import reconstruct_dag` | DAG reconstruction + `critical_path` + `bubbles` + `total_bubble_ns` + `sync_event_count`. |
| `rccl_analysis.analyze_collectives` | `from perfxpert.tools.rccl_analysis import analyze_collectives` | RCCL collective-ops analysis — per-op `effective_bw_gbps`, `efficiency_pct`, `efficiency_label`. |
| `interconnect.lookup_peaks` | `from perfxpert.tools.interconnect import lookup_peaks` | XGMI / Infinity-Fabric / PCIe peak bandwidth lookup (used by `analyze_collectives` for efficiency ratios). |
| `roofline.plot_points` | `from perfxpert.tools.roofline import plot_points` | Live Roofline — per-kernel `(ai, achieved_flops_per_s, bottleneck_class, fp_type, confidence)` from pmc_events. |
| `pragma.lookup_pragmas` | `from perfxpert.tools.pragma import lookup_pragmas` | Enumerate the 3 allowlisted LLVM loop-hint pragmas (+ 7 rejected entries kept for fence visibility). |
| `pragma.explain_pragma` | `from perfxpert.tools.pragma import explain_pragma` | Full catalog entry for a given `pragma_id`. |
| `pragma.suggest_pragmas_for_kernel` | `from perfxpert.tools.pragma import suggest_pragmas_for_kernel` | Amdahl-gated pragma candidates for a hot kernel. |
| `predict_impact.predict_change_impact` | `from perfxpert.tools.predict_impact import predict_change_impact` | Change-Impact Prediction: return `{predicted_speedup_range, confidence, rationale, roofline_delta, assumptions, source_citation, prediction_id}` for a baseline DB + kernel + change_type. Amdahl + tier-2 gates enforced internally. **Stub today — durable prediction store lands in a follow-up; status field surfaces "unsupported" when hit.** |
| `predict_impact.list_supported_changes` | `from perfxpert.tools.predict_impact import list_supported_changes` | Enumerate the change_type ids in `knowledge/change_impact_models.yaml`. Returns `[{id, applies_to, required_metrics}]`. |
| `predict_impact.explain_prediction` | `from perfxpert.tools.predict_impact import explain_prediction` | Re-hydrate a prediction by its `prediction_id`. In-process only today; durable store lands in a follow-up. |

All are READ_ONLY (safe for external callers) and deterministic
(no LLM, no sudo, no live device access). The runtime-monitor parsers
ingest a user-supplied JSON log — set `PERFXPERT_GPU_MONITOR_LOG=<path>`
to let specialists pick it up automatically.

## See also

- [MCP server](../integration/mcp-server.md) — same 8 agents
  available over JSON-RPC; identical input/output schemas.
- [Agent hierarchy](../architecture/agent-hierarchy.md) — tier
  diagram and source-tree locations.
- `perfxpert/agents/schemas.py` — authoritative Pydantic schema
  definitions (frozen models, field caps).
