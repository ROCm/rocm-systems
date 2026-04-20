# Python API (`perfxpert.api`)

`perfxpert.api` is a **1:1 mirror** of the 7 agent-hierarchy MCP
tools. Every callable here IS the same function the MCP server wraps
— the Python API and the MCP surface share a single implementation.
Use this module to embed PerfXpert's analysis brain in your own
tooling without running the MCP server.

Cross-links:

- [MCP server](../integration/mcp-server.md) — the same 7 agent tools
  + 34 classifier / knowledge tools re-exposed over stdio JSON-RPC.
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

Every callable honors `PERFXPERT_AIRGAP=1` and the full provider /
fallback-chain ladder (`PERFXPERT_LLM_FALLBACK_CHAIN`) because the
agent tools defer to `perfxpert.agents.runtime.build_session`.

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
`agent_latency_specialist`) accepts an optional
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
        "counter_data": {"vgpr_used": 128, "waves_per_eu": 2},
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

## See also

- [MCP server](../integration/mcp-server.md) — same 7 agents
  available over JSON-RPC; identical input/output schemas.
- [Agent hierarchy](../architecture/agent-hierarchy.md) — tier
  diagram and source-tree locations.
- `perfxpert/agents/schemas.py` — authoritative Pydantic schema
  definitions (frozen models, field caps).
