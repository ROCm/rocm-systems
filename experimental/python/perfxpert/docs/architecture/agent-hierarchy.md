# Agent Hierarchy

PerfXpert's agentic runtime (v0.2.0+) is a three-tier hierarchy of
OpenAI Agents SDK agents plus deterministic middleware. This document
is the authoritative reference for how the agents relate, how the
fence-slice pattern keeps each agent's system prompt narrow, and where
to look in source.

**Each of the 8 agents is callable from two surfaces:**

- **MCP:** `perfxpert_agent_root`, `perfxpert_agent_analysis`,
  `perfxpert_agent_recommendation`, `perfxpert_agent_correctness`,
  `perfxpert_agent_compute_specialist`,
  `perfxpert_agent_memory_specialist`,
  `perfxpert_agent_latency_specialist`,
  `perfxpert_agent_diff_specialist` (READ_ONLY, auto-registered
  from `perfxpert.tools.agents.*`).
- **Python API:** `perfxpert.api.agent_root(...)`,
  `perfxpert.api.agent_analysis(...)`, etc. — **1:1 mirror** of the
  MCP tools (same function, same schema).

The two surfaces resolve to the same underlying Python function; the
MCP registration is a thin wrapper around the `perfxpert.api.*`
callable. Agents are reachable from any entry point — the Root is NOT
the only way into the hierarchy. Backend TUIs (Claude Code, Gemini
CLI, Codex CLI, opencode) pick whichever agent matches the user's
intent based on the `AGENTS.md` reference.

Cross-links:
- [Gate cascade](gate-cascade.md) — the 5 deterministic correctness
  gates that sit between agents
- [MCP server](../integration/mcp-server.md) — how READ_ONLY tools are
  re-exposed to external clients
- [Python API](../guides/python-api.md) — in-process embedding with
  the same schemas as the MCP tools
- [Agentic mode guide](../guides/agentic-mode.md) — end-user provider
  ladder and air-gap behavior

## Three tiers

```
                     ┌───────────────┐
                     │     Root      │   Tier 0 — intent classify
                     └───────┬───────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ┌──────────────┐ ┌───────────────┐ ┌──────────────────┐
    │   Analysis   │ │  Correctness  │ │  Recommendation  │   Tier 1
    └───────┬──────┘ └───────────────┘ └────────┬─────────┘
            │                                    │
            └───────────────┬────────────────────┘
                            ▼
              ┌─────────┬────┴────┬────────────┐
              ▼         ▼         ▼            ▼
       ┌──────────┐ ┌────────┐ ┌────────┐ ┌────────┐
       │ Compute  │ │ Memory │ │Latency │ │  Diff  │  Tier 2
       │ special. │ │ spec.  │ │ spec.  │ │ spec.  │
       └──────────┘ └────────┘ └────────┘ └────────┘
```

- **Tier 0 (Root)** — classifies the user's natural-language query
  into a routing decision (profile, analyze, recommend, diagnose).
  Owns no tools; its only job is handoff selection.
- **Tier 1 (Layer-1 agents)** — three siblings:
  - **Analysis** — reads trace/source artifacts via READ_ONLY tools,
    emits a structured finding (primary bottleneck, metrics,
    kernel_runtimes).
  - **Correctness** — receives a `GateVerdict` from the gate-cascade
    middleware and narrates it (never runs gates itself). The tier-0
    diagram above shows Correctness as a sibling of the other two
    tier-1 agents, but the data flow is: Recommendation proposes an
    edit → middleware runs the gate cascade → the resulting
    `GateVerdict` is handed to Correctness. See
    [gate-cascade.md](gate-cascade.md).
  - **Recommendation** — converts findings into proposed code changes,
    delegates to specialists when needed.
- **Tier 2 (Specialists)** — four narrow experts invoked by
  Recommendation (or directly from a TUI backend) when the user's
  intent falls in their domain:
  - `compute_specialist` — VALU occupancy, waves-per-EU, VGPR pressure.
    Also consumes `kernel_fusion.find_fusion_candidates` to surface
    adjacent-short-kernel fusion recipes (elementwise+elementwise,
    GEMM+bias+act, ...) drawn from `knowledge/fusion_patterns.yaml`.
    MFMA vs VALU ratio classifier (`mfma_vs_valu_ratio_classification`
    in `knowledge/sol_metrics.yaml`) derived from existing `metrics.*`
    helpers. Attention-scope patterns (Flash / naive SDPA / KV-cache
    amplification) from `knowledge/attention_patterns.yaml`. Pragma
    suggestions (allowlisted LLVM loop hints) gated behind `--advanced`.
  - `memory_specialist` — HBM bandwidth, cache miss rate, LDS conflicts.
    Binds `unified_memory.analyze_paging` for MI300X cross-die + paging
    detection. Multi-level bandwidth chain (HBM → L2 → L1 → LDS) with
    per-level classifier + recommendation in
    `knowledge/memory_patterns.yaml`. Attaches change-impact
    predictions to each surfaced technique via
    `predict_impact.predict_change_impact`.
  - `latency_specialist` — dependency chains, kernel launch overhead,
    async-stream gaps.
    Binds `dependency_graph.reconstruct_dag` for critical-path +
    GPU-bubble attribution. Binds `rccl_analysis.analyze_collectives`
    + `interconnect.lookup_peaks` for per-collective bus-bandwidth +
    NIC efficiency ratios. Reads `gpu_runtime_monitor.*` outputs from
    the user-supplied `PERFXPERT_GPU_MONITOR_LOG` to flag thermal /
    power throttle (MCP surface only — not in the cap-5 allowlist
    because thermal analysis is diagnostic / out-of-band).
  - `diff_specialist` — run-to-run comparison; wraps
    `trace_diff.diff_runs` + `regression.compare_runs` +
    `roofline.classify` and returns an
    `improved` / `regressed` / `neutral` verdict with per-kernel deltas.

## Fence-slice pattern

Each agent's system prompt lives in `perfxpert/agents/fence/<agent>.md`
and is loaded verbatim at runtime — no runtime string concatenation of
prose. Slices are enforced at ≤ 400 lines by CI.

Slice files (one per agent, plus `always.md` loaded into every agent):

```
perfxpert/agents/fence/
├── always.md              # safety preamble loaded into every agent
├── root.md                # tier 0
├── analysis.md            # tier 1
├── correctness.md         # tier 1
├── recommendation.md      # tier 1
├── compute_specialist.md  # tier 2
├── memory_specialist.md   # tier 2
├── latency_specialist.md  # tier 2
└── diff_specialist.md     # tier 2
```

Why fence-slicing:

1. **Scope containment** — each agent only sees rules relevant to
   its job. Cross-agent leakage is impossible because the other slices
   are never loaded into that agent's context.
2. **Length bound** — the 400-line cap forces focused guidance. Content
   that grows past the cap is a signal to split agents or move facts
   into `perfxpert/knowledge/*.yaml`.
3. **Reviewability** — every slice change shows as a small diff. The
   prior monolithic reference guide (removed in Phase 7.1) made
   cross-agent behavior hard to reason about in a single review pass.

## Invoking the hierarchy

From Python, you interact with the hierarchy through the session handle
returned by `build_session()`:

```python
from perfxpert.agents.runtime import build_session
from perfxpert.agents.schemas import RootInput

session = build_session(airgap=True)  # or provider='anthropic'
output = session.run_root(RootInput(
    user_query="Identify the hot kernel and suggest a first fix.",
    airgap=True,
))
print(output.primary_bottleneck)
print(output.recommendations)
```

The session also exposes `run_analysis`, `run_correctness`, and
`run_recommendation` for direct tier-1 invocation — useful for
integrations that already know the routing decision.

## Where to find each piece in source

| Concern | Path |
|---------|------|
| Session handle + agent runner | `perfxpert/agents/runtime.py` |
| Schemas (Root/Analysis/Correctness/Recommendation I/O) | `perfxpert/agents/schemas.py` |
| Per-agent definitions | `perfxpert/agents/{root,analysis,correctness,recommendation}.py` |
| Specialists | `perfxpert/agents/{compute,memory,latency,diff}_specialist.py` |
| Fence slices (system prompts) | `perfxpert/agents/fence/*.md` |
| Agent framework + framework-level helpers | `perfxpert/agents/framework.py` |
| Gate cascade (middleware between agents) | `perfxpert/runtime/gate_cascade.py` |
| Recursion guard (prevents opencode→opencode loops) | `perfxpert/runtime/recursion_guard.py` — `ensure_not_recursive` is also re-exported from `perfxpert.runtime` and imported from there by `agents/runtime.py` |
| Intent classifier (deterministic pre-route) | `perfxpert/runtime/intent_classifier.py` |

## Adding a new agent

The contributor walkthrough is in
[contributing/agents.md](../contributing/agents.md). In brief:

1. Create `perfxpert/agents/<name>.py` with an `AgentSpec` + handoff
   dataclass.
2. Add the fence slice `perfxpert/agents/fence/<name>.md` (≤ 400 lines).
3. Wire it into the parent agent's tools list (Recommendation →
   specialist allowlist; Root → tier-1 allowlist).
4. Add tests under `tests/test_agents/test_<name>.py`.
5. Update this page's diagram.

Every agent addition is a small diff: one Python module, one fence
slice, two tests. No architectural review required for agents that fit
within an existing tier.
