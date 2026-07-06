# Memory Bandwidth Analysis in Memory Chart

**Status:** Draft (architecture revision)  
**Target:** rocprofiler-compute, gfx950, analyze mode (CLI)  
**Related:** `3000_mem_bw.yaml`, `mem_chart_gfx9.py`, `sample/membw_analysis_test_suite/`

---

## Glossary

| Term | Definition |
|------|------------|
| **Memory Chart** | High-level ASCII diagram of the memory hierarchy rendered in analyze mode (`cli_style: mem_chart`). |
| **Block 30** | User-facing section number for Memory Bandwidth Analysis. Corresponds to panel ID `3000` in `analysis_configs/<arch>/3000_mem_bw.yaml` (panel IDs 3000–3099). |
| **GL1** | First-level graphics cache (L1 / TCP path in the memory hierarchy). |
| **GL2** | Second-level graphics cache (L2 / TCC path). |
| **EA** | Efficiency Arbiter — SoC demarcation between L2 and the memory fabric (HBM, GMI, PCIe). |
| **TCP** | Texture Cache Per CU (L1 cache front-end). |
| **UTCL1** | Unified Translation Cache Level 1 (address translation / TLB path). |
| **TCC** | L2 cache controller block. |
| **TA** | Texture Addresser (address generation front-end). |
| **Optiq** | Downstream performance visualization tool (internal consumer; persistence is phase 2). |
| **`--membw-analysis`** | CLI flag enabling Memory Bandwidth Analysis (block 30). Note: the flag is `--membw-analysis`, not `--mem-bw-analysis`. |
| **`normal_unit`** | Normalization unit passed to `plot_mem_chart()` (e.g. per-wave, per-cycle) used to scale displayed metric values. |

### Requirement prefixes

| Prefix | Meaning |
|--------|---------|
| **FR** | Functional requirement — externally observable behavior. |
| **NFR** | Non-functional requirement — performance, compatibility, operability. |

---

## System Context

In analyze mode, rocprof-compute renders the **Memory Chart**, a high-level graphical representation of the memory hierarchy populated with numeric metrics to aid user understanding of their workload's memory performance.

On gfx950, rocprof-compute has memory bandwidth analysis metrics describing bandwidth, latency, and throughput within the memory hierarchy. These metrics live in block 30 (`3000_mem_bw.yaml`).

Currently, gfx950 memory bandwidth analysis metrics are hidden behind `--experimental --membw-analysis`. While these metrics are relatable to what the memory chart presents, there is no relationship between the two features.

---

## Problem Statement

**[PS1] No bottleneck identification.**  
The memory chart presents metric results but does not identify which memory levels are bottlenecks. Users must manually interpret request counts, hit rates, latencies, and stall indicators across memory levels.

**[PS2] Bottleneck analysis exists but is inaccessible.**  
Bottleneck detection logic for GL1/GL2/EA exists in the codebase (as indicators and thresholds in `3000_mem_bw.yaml`) but is gated behind `--experimental --membw-analysis`, disconnected from the memory chart, and presented only as flat metric tables (block 30).

**[PS3] No guided analysis.**  
Users lack structured guidance when diagnosing memory performance issues. Memory bandwidth guided analysis has been a long-standing user request.

**[PS4] No structured data export.**  
Downstream tools (Optiq) hardcode the memory chart structure rather than consuming it from the analysis database. Bottleneck results have no database representation at all. Decoupling these tools requires a structured, data-driven export.

---

## Scope

### In scope (v1)

1. **gfx950 only** (initial target)
2. **Analyze mode only** (profiled data is assumed available)
3. **Memory levels:** GL1, GL2, EA (where bottleneck equations exist today)
4. **CLI renderer only** — TUI and GUI renderers are out of scope for v1, but the analysis result model must be renderer-agnostic

### Out of scope (v1)

| Item | Notes |
|------|-------|
| gfx115 topology updates (formerly PR1) | Deferred to a separate gfx115 milestone |
| Optiq database persistence | Phase 2 — design result model to be JSON-serializable in v1 |
| TUI / GUI rendering of annotations | Phase 2 — consume same `MemBwAnalysisResult` |

---

## Assumptions

- **[A1]** mem-bw-analysis metrics must be validated and promoted out of experimental (see FR2).
- **[A2]** Bottleneck thresholds in `3000_mem_bw.yaml` comments (e.g. `>= 10%`) are the authoritative starting defaults unless overridden in `membw_tree_spec.yaml`.
- **[A3]** v1 ships only for architectures with `analysis_configs/gfx950/3000_mem_bw.yaml` and a working memory chart renderer (`mem_chart_gfx9.py`).

---

## Requirements

### FR1 — Bottleneck detection on the memory chart (PS1)

Show bottleneck detection results on the memory chart at the relevant memory level, showing only detected (active) bottlenecks along with their supporting metrics.

- **[FR1.1]** Evaluate bottleneck equations from profiled data to produce boolean results per memory level and sub-condition.
- **[FR1.2]** Display only active (`state == active`) bottlenecks on the chart — no visual noise from non-triggered indicators.
- **[FR1.2a]** When no bottlenecks are detected across GL1/GL2/EA, print a single status line below the chart: `Memory Bandwidth Analysis: No bottlenecks detected (GL1 / GL2 / EA).`
- **[FR1.3]** Show supporting metric values alongside each active bottleneck (e.g. `stall rate = 15.2%`). Multiple metrics use the format: `metric1=42% | metric2=1.2M`.
- **[FR1.4]** CLI renderer only (initial target).

**Acceptance (FR1.1):** Given a profiled workload from `membw_analysis_test_suite` with a known bottleneck, the engine must mark the corresponding leaf node `active` and all mutually exclusive siblings `inactive`.

### FR2 — Remove experimental gate (PS2)

mem-bw-analysis becomes a standard part of the analyze flow.

- **[FR2.1]** Counters required by mem-bw-analysis must be collected in standard profile mode on gfx950 without `--experimental`.
- **[FR2.2]** `--experimental --membw-analysis` continues to work for **two minor releases** with a deprecation warning printed to stderr, then the experimental requirement for membw is removed entirely. The `--membw-analysis` flag itself is retained.

### FR3 — Guided analysis text (PS3)

When a bottleneck is detected, provide textual guidance to the user.

- **[FR3.1]** Each guidance block for an active leaf bottleneck must include:
  - **Condition** — human-readable label and threshold
  - **Measured** — actual metric value(s) from profiled data
  - **Impact** — what the bottleneck means for memory performance
  - **Action** — at least one recommended remediation step
- **[FR3.2]** Guidance must incorporate actual metric values via templates, not static text only.
- **[FR3.3]** **Resolved by Decision 3:** Guidance is printed below the plotille canvas, separated by a visible delimiter, immediately following chart output. Canvas blocks carry short annotation labels only.

### FR4 — Structured export for downstream tools (PS4, phase 2)

Persist results for downstream consumption (Optiq).

- **[FR4.1]** v1: Define `MemBwAnalysisResult` as a JSON-serializable dataclass (schema version 1). Full database persistence deferred to phase 2 after Optiq schema review.
- **[FR4.2]** Coordinate with Optiq on database schema before phase 2 ships. Owner: TBD. Target milestone: TBD.

---

## Non-Functional Requirements

- **[NFR1]** Bottleneck tree evaluation must add less than 50 ms to total analyze runtime for a single-kernel workload on gfx950.
- **[NFR2]** Minimum terminal width for full canvas rendering: 240 columns. Narrower terminals or piped output fall back to tabular guidance without canvas annotations (chart may truncate with a warning).
- **[NFR3]** Bottleneck indicators must be visible without color: every annotation includes a symbol prefix (e.g. `[!]`) in addition to optional color.
- **[NFR4]** Maximum active guidance blocks displayed: 5. Overflow: `...and N more (see block 30 for full detail)`.

---

## Design Decisions

### Decision 1: Where do bottleneck equations live?

Bottleneck equations form a dependency tree where child conditions reference parent results and use negation (e.g. `gl1_bottleneck_tcp_utcl1` requires `gl1_bottleneck_tcp == true`; `gl1_bottleneck_tcp_other` requires siblings to be false).

| Option | Description | Pros | Cons |
|--------|------------|------|------|
| Pure YAML | Define all equations as YAML metrics with a new annotation | Consistent with existing metric authoring | YAML engine cannot reference other metric results, express negation, or model parent/child dependencies |
| **Hybrid (chosen)** | **Raw stall rate metrics in YAML; thresholds + tree in `membw_tree_spec.yaml`; evaluation in Python** | Reuses existing YAML metrics; tree logic expressed naturally in Python; thresholds remain editable in config | Two (plus tree spec) places to understand the full picture |
| Pure Python | All definitions (thresholds, tree, boolean logic) in Python | Single location for all logic | Threshold values in code, inconsistent with metric authoring pattern, harder to tune |

**Artifact boundaries:**

| Artifact | Location | Owns |
|----------|----------|------|
| Raw metrics | `analysis_configs/gfx950/3000_mem_bw.yaml` | Counter equations, units, descriptions |
| Thresholds + tree | `membw_tree_spec.yaml` (new) | Node IDs, parent refs, threshold keys, guidance template IDs |
| Evaluator | `src/rocprof_compute_analysis/membw/engine.py` | Boolean logic, indeterminate propagation, template fill |

Startup validation: every `metric:` key in the tree must exist in `3000_mem_bw.yaml`; every `threshold:` key must exist in the `thresholds:` section of `membw_tree_spec.yaml`.

### Decision 2: How do bottleneck results reach the memory chart renderer?

`plot_mem_chart(normal_unit, metric_dict)` currently receives a flat dictionary. Bottleneck results are hierarchical (parent/child booleans with supporting values).

| Option | Description | Pros | Cons |
|--------|------------|------|------|
| Extend `metric_dict` | Flatten bottleneck booleans into the existing dict | Minimal change to data flow | Loses hierarchy; mixes raw metrics with derived analysis |
| **Separate data channel (chosen)** | **Pass `MemBwAnalysisResult` as a new optional argument to `plot_mem_chart()`** | Clean separation; structured representation preserves hierarchy | Requires updating function signature and call sites |

**API:**

```python
def plot_mem_chart(
    normal_unit: str,
    metric_dict: dict[str, Any],
    membw: MemBwAnalysisResult | None = None,
) -> str:
    ...

def render_membw_guidance(membw: MemBwAnalysisResult) -> str:
    ...  # called by tty.py after plot_mem_chart; concatenated for stdout
```

### Decision 3: How are bottleneck indicators displayed to the user?

The CLI renderer (`mem_chart_gfx9.py`) draws on a fixed 234×42 plotille canvas with absolute coordinates per block.

| Option | Description | Pros | Cons |
|--------|------------|------|------|
| Annotate blocks only | Colored text within or adjacent to block rectangles | Visual proximity to memory level | Limited space; no room for guidance text |
| Dedicated canvas region | New area on the canvas listing active bottlenecks | More space for detail | Canvas size must increase; layout complexity |
| Separate panel only | Rich table or text block after the plotille canvas | Simplest implementation | Loses visual connection to memory levels |
| **Annotate blocks + guidance below (chosen)** | **Short labels on canvas blocks; detailed guidance text printed below the chart** | Visual proximity on chart; guidance has room for detail and metric values | Two rendering concerns (canvas labels + post-chart text) |

**Canvas annotation rules:**

- Short label only: `[!] GL1: TCP` or `BN: L2`
- Derive coordinates from existing `MemChart` block constants — no independent hardcoded offsets
- Color when TTY supports it; symbol prefix always present (NFR3)

**Guidance panel rules:**

- Printed after `canvas.plot()`, separated by `─` × 80
- One block per active **leaf** bottleneck (max 5 per NFR4)
- Word-wrap to 80 columns
- Template structure: Condition / Measured / Impact / Action

**Example output:**

```
┌─ Memory Chart (plotille canvas) ──────────────────────────┐
│  [GL1 block annotated: [!] TCP]                          │
└───────────────────────────────────────────────────────────┘
────────────────────────────────────────────────────────────────
Memory Bandwidth Analysis
────────────────────────────────────────────────────────────────
[GL1] TCP stalled by UTCL1
  Condition : UTCL1 translation stall rate >= 10%
  Measured  : 15.2% (threshold: 10.0%)
  Impact    : Address generation blocked waiting on L1 TLB
  Action    : Reduce address working set; check TLB pressure
```

### Decision 4: Feature availability and gate consolidation (new)

`membw_analysis` is currently checked in four independent callsites (`rocprof_compute_base.py`, `soc_base.py` ×2, `tty.py`, `tui_utils.py`). v1 consolidates to a single policy function.

```python
@dataclass(frozen=True)
class MemBwAvailability:
    enabled: bool
    state: Literal["available", "arch_unsupported", "counters_missing", "disabled_by_flag"]
    message: str | None

def membw_availability(args, arch, profile_config, counter_set) -> MemBwAvailability:
    ...
```

All existing gate callsites delegate to `membw_availability()`.

### Decision 5: Metric aggregation contract (new)

All ratio metrics used by the bottleneck tree **must** use pairwise `SUM(numerator) / SUM(denominator)` aggregation across selected kernels/dispatches.

- **Never** `AVG(A) / AVG(B)` for bottleneck tree inputs
- If denominator `SUM == 0`, the metric is **indeterminate** (not zero, not inactive)
- Indeterminate nodes are omitted from display; they are never treated as non-bottlenecks
- Audit `3000_mem_bw.yaml` for `AVG(...)/AVG(...)` pairs referenced by tree nodes; align or exclude before merge
- Validate threshold changes against `sample/membw_analysis_test_suite/` workloads before merge

---

## Architecture

### Data flow

```
Profiled counter DB
        │
        ▼
┌───────────────────┐     ┌─────────────────────┐
│ 3000_mem_bw.yaml  │     │ membw_tree_spec.yaml │
│ (raw metrics)     │     │ (thresholds + tree)  │
└────────┬──────────┘     └──────────┬──────────┘
         │                            │
         ▼                            ▼
    metric_extract.py ──────────► engine.py
         │                            │
         │                     MemBwAnalysisResult
         │                            │
         ├──────────────┬─────────────┼──────────────┐
         ▼              ▼             ▼              ▼
  plot_mem_chart   render_membw   block 30      JSON export
  (annotations)    _guidance()    (unchanged)   (phase 2)
```

**Principle:** One evaluation pass produces one `MemBwAnalysisResult`. Renderers and exporters consume it; they never re-evaluate the tree.

**Pipeline insertion point:** After `calc_metrics()` normalization, before `plot_mem_chart()` is called in `tty.py`.

### MemBwAnalysisResult contract

```python
@dataclass(frozen=True)
class SupportingMetric:
    key: str            # YAML metric key, e.g. "L1 Cache - TCP miss rate"
    value: float | None
    unit: str           # "Percent", "Cycles per Request", etc.
    display: str        # pre-formatted, e.g. "42.3%"

@dataclass(frozen=True)
class BottleneckNode:
    id: str             # e.g. "gl1_bottleneck_tcp_utcl1"
    label: str          # user-facing short label for canvas
    level: Literal["GL1", "GL2", "EA"]
    state: Literal["active", "inactive", "indeterminate"]
    supporting: tuple[SupportingMetric, ...]
    children: tuple["BottleneckNode", ...]

@dataclass(frozen=True)
class MemBwAnalysisResult:
    schema_version: int = 1
    arch: str
    availability: Literal["full", "partial", "unavailable"]
    availability_reason: str | None
    nodes: tuple[BottleneckNode, ...]
    guidance_blocks: tuple[str, ...]
```

### membw_tree_spec.yaml sketch

```yaml
schema_version: 1
thresholds:
  stall_rate_high: 10.0   # percent; matches ">= 10%" in 3000_mem_bw.yaml comments

nodes:
  gl1_bottleneck_tcp:
    level: GL1
    metric: "L1 Cache - TA stalled by TCP (aggregated)"
    op: gte
    threshold: stall_rate_high
    children:
      gl1_bottleneck_tcp_utcl1:
        metric: "L1 Cache - TCP stalled by UTCL1"
        op: gte
        threshold: stall_rate_high
        guidance_id: gl1_tcp_utcl1
      gl1_bottleneck_tcp_other:
        requires_parent: true
        requires_siblings_false: [gl1_bottleneck_tcp_utcl1]
        guidance_id: gl1_tcp_other

guidance_templates:
  gl1_tcp_utcl1: |
    [GL1] UTCL1 translation pressure detected.
    TCP stalled by UTCL1: {metric:L1 Cache - TCP stalled by UTCL1} (threshold: {threshold:stall_rate_high}%).
    Consider improving TLB locality or reducing working-set span.
```

### JSON export schema (v1 stub, phase 2 persistence)

```json
{
  "schema_version": 1,
  "arch": "gfx950",
  "availability": "full",
  "memory_levels": {
    "GL1": {
      "bottlenecks": [
        {
          "id": "gl1_bottleneck_tcp_utcl1",
          "label": "TCP stalled by UTCL1",
          "supporting": [
            { "key": "L1 Cache - TCP stalled by UTCL1", "value": 15.2, "unit": "Percent" }
          ],
          "guidance": "..."
        }
      ]
    },
    "GL2": { "bottlenecks": [] },
    "EA": { "bottlenecks": [] }
  }
}
```

### Package layout (new)

```
src/rocprof_compute_analysis/membw/
├── __init__.py
├── models.py          # MemBwAnalysisResult, BottleneckNode, SupportingMetric
├── policy.py          # membw_availability()
├── metric_extract.py  # keyed metric values from calc_metrics output
├── engine.py          # tree evaluation
├── tree_spec.py       # load + validate membw_tree_spec.yaml
├── guidance.py        # template rendering
└── tree_spec/
    └── gfx950_membw_tree_spec.yaml
```

### Rendering separation

Canvas annotation (`mem_chart_gfx9.py`) and guidance text (`guidance.py`) are implemented as separate functions with independent error paths. A failure in guidance rendering must not suppress the memory chart.

---

## Failure Mode Matrix

| Failure | User-visible behavior | Severity |
|---------|----------------------|----------|
| Unsupported arch (not gfx950) | Memory chart renders normally; one-line stderr notice: `Memory bandwidth analysis not supported on <arch>` | Info |
| Legacy profile (membw counters not collected) | Memory chart renders; guidance area shows: `Memory bandwidth analysis unavailable (re-profile with current rocprof-compute)` | Info |
| `--membw-analysis` not set | Block 30 omitted from default analyze; memory chart has no annotations (current behavior) | N/A |
| Required counter missing / NaN | Affected tree nodes marked `indeterminate` and omitted; other nodes evaluated normally | Warning if all indeterminate |
| Malformed `membw_tree_spec.yaml` | Hard error at analyze startup with file path and validation message | Error |
| All nodes indeterminate | `Memory Bandwidth Analysis: Inconclusive (insufficient counter data).` | Warning |
| Terminal width < 240 columns | Canvas may truncate with warning; guidance still prints in tabular form | Warning |
| Guidance template render failure | Chart + annotations render; guidance shows fallback: `See block 30 for detail` | Warning |

---

## Counter Collection

FR2.1 requires membw counters in standard gfx950 profile mode.

**Validation steps before merge:**

1. List all counters referenced in `3000_mem_bw.yaml` via `tools/counter_grouping_inspector.py`
2. Confirm single-pass (or documented multi-pass) feasibility for TA, TCP/L1, and TCC/L2 counter groups on gfx950
3. Extend default counter set for gfx950 standard profile to include block 30 counters
4. Document any counters that require an extra profiling pass (fallback: partial availability with notice)

---

## Testing Strategy

### Unit tests

| Area | Test |
|------|------|
| Tree engine | Parameterized vectors per GL1/GL2/EA: input metric values → expected node states |
| Indeterminate propagation | Missing/NaN counter → node `indeterminate`, not `inactive` |
| Sibling negation | `gl1_bottleneck_tcp_other` active only when parent true and siblings false |
| Policy | `membw_availability()` returns correct state for each arch/counter/flag combination |
| Tree spec validation | Invalid metric key or threshold reference fails at load time |

### Integration tests

| Area | Test |
|------|------|
| Legacy profiles | Analyze pre-FR2.1 profiled data → chart renders, unavailable notice, no crash |
| Pipeline | `calc_metrics()` → engine → `plot_mem_chart()` + `render_membw_guidance()` on fixture workload |
| Gate consolidation | All four former gate callsites behave consistently via `membw_availability()` |

### End-to-end / golden tests

| Area | Test |
|------|------|
| CLI stdout | `rocprof-compute analyze` on `membw_analysis_test_suite` baseline/optimized pairs; snapshot annotation strings |
| Deprecation | `--experimental --membw-analysis` prints stable deprecation warning (FR2.2) |
| Zero bottlenecks | Workload with no active bottlenecks → FR1.2a status line |

### Test fixtures

Primary: `sample/membw_analysis_test_suite/` (GL1, GL2, EA workloads with documented baseline vs optimized expectations).

---

## Implementation Sequence

| Phase | Deliverable | Depends on |
|-------|-------------|------------|
| 1 — Foundation | `MemBwAnalysisResult`, `membw_tree_spec.yaml` schema, startup validation | — |
| 2 — Policy | `membw_availability()`, gate consolidation | Phase 1 |
| 3 — Core | `metric_extract.py`, `engine.py`, unit tests | Phase 1–2 |
| 3b — Aggregation audit | Align `AVG/AVG` pairs in `3000_mem_bw.yaml` used by tree | Phase 1 |
| 4 — UX | Canvas annotations in `mem_chart_gfx9.py`, `guidance.py` | Phase 3 |
| 5 — Ship | Remove experimental gate (FR2), golden e2e tests | Phase 4 |
| 6 — Phase 2 | Optiq JSON/DB persistence (FR4) | Phase 5 + Optiq schema |

---

## Open Items

| Item | Owner | Target |
|------|-------|--------|
| Optiq schema review (FR4) | TBD | Phase 2 milestone |
| gfx115 memory chart topology | TBD | Separate gfx115 milestone |
| Counter single-pass feasibility sign-off on MI350 hardware | TBD | Before FR2.1 merge |
| Guidance template content for all GL2/EA leaf nodes | TBD | Phase 4 |

---

## References

- `src/rocprof_compute_soc/analysis_configs/gfx950/3000_mem_bw.yaml` — raw metrics and threshold comments
- `src/utils/mem_chart_gfx9.py` — memory chart renderer (`plot_mem_chart`)
- `src/utils/tty.py` — analyze output orchestration (memory chart branch)
- `sample/membw_analysis_test_suite/` — validation workloads
- `tests/test_profiler_base.py` — experimental gating tests (update for FR2)
