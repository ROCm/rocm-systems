# Contributing: evolving the schema

## What you're changing

The data contracts the report renderers, Python API, and MCP tool
surface all share. PerfXpert has **three** distinct schema layers —
changes usually touch at least two of them, and a change to the JSON
contract requires a `schema_version` bump.

## The three layers

### Layer A — Agent I/O (Pydantic)

`perfxpert/agents/schemas.py` defines the immutable handoff schemas
between agents:

- `RootInput` / `RootOutput` — top-level orchestrator.
- `AnalysisInput` / `AnalysisOutput` — deterministic analysis agent.
- `RecommendationInput` / `RecommendationOutput` — recommendation agent.
- `CorrectnessInput` / `CorrectnessOutput` — 5-gate correctness cascade.
- `ComputeSpecialistInput` / `ComputeSpecialistOutput`,
  `MemorySpecialistInput` / `MemorySpecialistOutput`,
  `LatencySpecialistInput` / `LatencySpecialistOutput`,
  `DiffSpecialistInput` / `DiffSpecialistOutput` — Layer-2 experts.
  `DiffSpecialistOutput` packs its per-kernel deltas into a single
  `kernel_deltas: Dict[str, List[Dict]]` field with `"regressions"` /
  `"improvements"` keys so the 5-field output cap still holds; the MCP
  wrapper (`perfxpert.tools.agents.diff.agent_diff_specialist`)
  flattens them back to top-level keys for the public wire shape.

All models inherit from `_FrozenModel` (`frozen=True, extra="forbid"`)
and are subject to CI-enforced field caps: **≤10 input fields, ≤5
output fields** (see `tests/test_agents/test_schema_field_caps.py`).

### Layer B — Deterministic payload (dict)

`perfxpert/analysis/payload.py::build_analysis_payload` returns a plain
dict that every formatter consumes:

```python
# SKIP-SAMPLE — illustrative shape returned by build_analysis_payload
{
    "database_path": "...",
    "time_breakdown":         {...},   # kernel/memcpy/overhead %, ns totals
    "hotspots":               [...],   # list of kernel dicts
    "memory_analysis":        {...},   # per-direction bandwidth + counts
    "hardware_counters":      {...},   # {has_counters, metrics, counters}
    "kernel_resources":       {...},   # VGPR/SGPR/LDS/scratch + occupancy
    "api_overhead":           {...},   # per-API totals + launch overhead
    "warmup_issues":          {...},   # outlier first-call detection
    "thread_trace":           {...} | None,  # Tier-3 ATT
    "tier0_findings":         {...} | None,  # source scanner output
    "recommendations_deterministic": [...],  # rule-driven recs
    "metadata":               {...},
}
```

### Layer C — External JSON doc

`perfxpert/formatters/json_fmt.py::_format_as_json` serialises Layer B
into the public JSON document, stamped with a top-level
`schema_version` field. The full additive chain is
**`0.3.0` → `0.3.1` → `0.3.2` → `0.3.3` → `0.3.4` → `0.4.0`** — each
step advances as richer data becomes available (see `# CHANGES` below
and the "Schema versioning policy" table). Tier-3 ATT pins to `0.4.0`
regardless of which other sections are present.
`perfxpert/analyze.py::_format_agentic_output`
(~line 790) then overlays the agentic brain (`narrative`,
`primary_bottleneck`, `warnings`, `tier0_findings`) and re-bumps the
version if it was still at `0.1.0` / `0.2.0`.

<!-- # CHANGES — 0.3.0 → 0.3.1
     Additive: `hotspots[i].source_locations: list[{file, line, kind,
     severity, severity_label}]` where `kind ∈ {"definition", "launch"}`
     and `severity ∈ {"HIGH","MEDIUM","LOW","INFO"}` (label
     CRITICAL/HOT/WARM/COOL). Emitted when `--source-dir` was supplied
     and the Tier-0 scanner correlated at least one hotspot with a
     detected kernel. Absent field when no source scan was performed;
     empty list when the scanner ran but no basename matched. See
     Confluence row #5 (Source Code Line numbers) for the UI rollout
     details.

     Additive: top-level `trace_diff` key (Confluence row #7). Emitted
     by `perfxpert analyze -i new.db --baseline baseline.db --format json`
     and by the standalone `perfxpert diff --format json` command.
     Shape:

       "trace_diff": {
         "schema_version": "0.3.1",
         "baseline_db": str, "new_db": str,
         "wall_delta_ns": int, "wall_delta_pct": float,
         "per_kernel": [
           {"name": str,
            "baseline_ns": int, "new_ns": int,
            "delta_ns": int, "delta_pct": float,
            "regressed": bool, "was_hot": bool}, ...
         ],
         "primary_regressions": [<per_kernel entries with delta_pct > +3%>],
         "primary_improvements": [<per_kernel entries with delta_pct < -3%>],
         "narrative": str
       }

     The `trace_diff` section carries its OWN `schema_version` so
     downstream consumers can parse it independently when it arrives
     as the top-level payload of `perfxpert diff --format json` (no
     enclosing analyze report). -->


The three layers form a pipeline:

```
agents/schemas.py (Layer A)  ──►  payload.py (Layer B)  ──►  json_fmt.py (Layer C)
       Pydantic handoffs             deterministic dict        public JSON
       frozen, field-capped           no LLM involvement        schema_versioned
```

## How to evolve an agent schema (Layer A)

Example: add `gfx_id: str` to `AnalysisOutput` so downstream agents
don't re-derive it.

1. **Extend the model** in `perfxpert/agents/schemas.py`:

   ```python
   # SKIP-SAMPLE — illustrative Pydantic field addition
   class AnalysisOutput(_FrozenModel):
       primary_bottleneck: BottleneckType
       confidence: float = Field(..., ge=0.0, le=1.0)
       time_breakdown: Dict[str, float]
       hot_kernels: List[Dict[str, Any]]
       counter_data_available: bool
       gfx_id: str = ""  # NEW — check field cap (≤5 outputs).
   ```

   Watch the `≤5 output fields` cap — the CI test in
   `tests/test_agents/test_schema_field_caps.py` will fail the
   build if you exceed it.

2. **Populate it** in the agent's runner — `perfxpert/agents/analysis.py`
   (or wherever `run_analysis` lives):

   ```python
   # SKIP-SAMPLE — illustrative
   return AnalysisOutput(
       primary_bottleneck=bottleneck,
       confidence=conf,
       time_breakdown=tb,
       hot_kernels=hk,
       counter_data_available=has_counters,
       gfx_id=detect_gfx_id(connection),
   )
   ```

3. **Thread it downstream.** Either carry through `RootOutput.metadata`
   (untyped escape hatch, no field-cap cost) or extend a specialist
   input schema (typed, cap-bound). Both options require test
   coverage.

4. **Render it in all four formatters** — text, markdown, json_fmt,
   webview. Skipping a format here creates format parity drift
   (Layer B has the data, but Layer C loses it on render). The
   reviewer checklist at the bottom of this doc catches this.

5. **Document in the Python API.** `docs/guides/python-api.md`
   exposes `perfxpert.api.agent_analysis` — the field surfaces there
   automatically via Pydantic, but the docstring / field table must
   be updated by hand.

6. **MCP tool discovery is automatic.** The 8 agent tools
   (`perfxpert_agent_<name>`) read their JSON schema from the Pydantic
   model via reflection — no manual MCP descriptor edit is needed.
   Verify with `perfxpert-mcp --describe` that the new field appears
   in the tool-list output.

7. **Bump `schema_version`** only if the change reaches Layer C
   (JSON doc). A field added to `AnalysisOutput` that's consumed
   internally but never serialised into `_format_as_json`'s output
   does NOT require a bump. See the versioning policy below.

## How to add a new deterministic-payload section (Layer B)

Example: `thread_trace` was added this way.

1. **Add the key to `build_analysis_payload`** in
   `perfxpert/analysis/payload.py` — populate it inside the
   `payload: Dict[str, Any] = {...}` literal at ~line 353, and
   attach a best-effort computation branch with try/except so a
   missing data source never fails the whole pass:

   ```python
   # SKIP-SAMPLE — illustrative new payload section
   payload["my_new_section"] = {}
   if optional_precondition_met:
       try:
           payload["my_new_section"] = analyze_my_new_section(connection)
       except Exception:
           payload["my_new_section"] = {"has_data": False, "reason": "..."}
   ```

2. **Render in all four formatters.**
   - `text.py` — a banner + tabular block inside `format_analysis_output`.
   - `markdown.py::_format_as_markdown` — an `##` section.
   - `json_fmt.py::_format_as_json` — a top-level key on the JSON
     doc (rename if the external contract name differs from the
     payload key).
   - `webview.py::_format_as_webview` — a `<section class="scard">`
     block; wire the template in
     `perfxpert/formatters/templates/webview.html`.

3. **Assert presence + ordering** in
   `tests/test_formatters/test_report_structure.py` — the existing
   four tests show the pattern (`find()` indices + `assert i_a < i_b`).

4. **Document the new key** in
   `docs/guides/getting-started.md` "Report structure" subsection
   (§4 of that guide) so users know to look for it.

### Pure-rendering derivatives (no schema change)

Some webview visualisations render directly from an existing
deterministic-payload key and do **NOT** introduce a new payload
section. Examples:

- **ATT flame graph** — `perfxpert/formatters/_att_flamegraph.py`
  consumes the existing `payload["thread_trace"]` block (`kernels`,
  `top_stalling_instructions`, `stall_category`) and produces an
  inline SVG for the Thread Trace `.scard` in the webview. The
  flame graph is webview-only; the Markdown / text / JSON formats
  already surface the same data via the stall table and the
  `thread_trace` JSON key, so no Layer-A / Layer-B / Layer-C
  schema change is required. The renderer is a pure function of
  the payload — no new MCP tool, no new agent, no new dependency.

When adding a pure-rendering derivative, only wire the renderer
into `webview.py::_format_as_webview` (via a new
`%%…_SECTION%%` template slot inside the existing scard) and add a
unit test under `tests/test_formatters/` that asserts the rendered
fragment shape on a synthetic payload. Do NOT bump
`schema_version` — the versioning policy below only applies to
changes in the external JSON document (Layer C).

## Schema versioning policy (Layer C)

The JSON doc carries a top-level `schema_version` field. Consumers
MUST check it before parsing. Bumps follow semver-lite:

| Change | Bump |
|--------|------|
| Breaking: field renamed, field removed, semantics changed | **Major** (`0.x.y` → `1.0.0`) |
| Additive: new top-level key, new sub-field under existing key | **Minor** (`0.3.0` → `0.4.0`) |
| Bugfix only: value format corrected, no key-set change | **Patch** (`0.3.0` → `0.3.1`) |

The current tree:

- `0.1.0` — pre-TraceLens baseline (trace-only reports).
- `0.2.0` — tier-0 source-scanner addition (used by
  `_format_tier0_json`).
- `0.3.0` — agentic brain (`narrative`, `primary_bottleneck`,
  `warnings`) + tier-0 separation + summary section.
- `0.3.1` — additive `hotspots[i].source_locations` field
  cross-referencing each hotspot with its Tier-0 definition +
  launch site (Confluence row #5) — each entry now also carries
  `severity` + `severity_label` (HIGH/MEDIUM/LOW/INFO;
  CRITICAL/HOT/WARM/COOL). Also additive: top-level `trace_diff`
  section emitted by `perfxpert diff` / `perfxpert analyze --baseline`
  (Confluence row #7).
- `0.3.4` — current: additive top-level `roofline` section emitted by
  `roofline.plot_points` (Live Roofline chart). Shape:
  ```json
  "roofline": {
    "schema_version": "0.3.x",
    "arch": "gfx942",
    "arch_peaks": {"fp32": 1.634e14, "fp16": 1.307e15, "bf16": 1.307e15,
                    "fp64": 8.17e13, "fp8": 2.614e15, "int8": 2.614e15},
    "hbm_bandwidth_bytes_per_s": 5.3e12,
    "dtype": "bf16",
    "dtype_confidence": "from_kernel_name",
    "kernels": [
      {
        "name": "gemm_bf16_kernel",
        "ai": 48.2,
        "achieved_flops_per_s": 1.12e14,
        "flops": 9.6e13,
        "bytes": 2_000_000_000,
        "duration_ns": 860_000,
        "duration_pct": 94.3,
        "fp_type": "bf16",
        "bottleneck_class": "compute",
        "confidence": "high"
      }
    ],
    "ridge_point": {"ai": 30.8, "flops_per_s": 1.634e14}
  }
  ```
  Populated only when Tier-2 hardware counters are available in the
  trace DB. Drives the webview's **Live Roofline** `.scard` (log-log
  inline SVG). `_format_as_json` bumps `schema_version` from `0.3.3`
  -> `0.3.4` when the key is set. ATT (`0.4.0`) still trumps.
- `0.3.3` — additive change-impact-prediction fields on each
  recommendation (`predicted_impact_range`, `predicted_confidence`,
  `predicted_rationale`, `source_citation`, `roofline_delta`). Emitted
  by the change-impact-prediction pipeline
  (`perfxpert.tools.predict_impact`) whenever the specialist-attached
  or category-mapped technique matches an entry in
  `knowledge/change_impact_models.yaml`. Shape of the recommendation
  entry when a prediction fires:
  ```json
  {
    "id": "ROCPD-COMPUTE-001",
    "priority": "HIGH",
    "category": "Compute-Bound Kernel",
    "issue": "...",
    "suggestion": "...",
    "predicted_impact_range": [1.15, 1.45],
    "predicted_confidence": 0.70,
    "predicted_rationale": "Technique 'vgpr_reduction' applied to ...",
    "source_citation": "knowledge/proven_optimizations.yaml#vgpr_reduction_compute_bound",
    "roofline_delta": {"before": null, "after": null}
  }
  ```
  Rules enforced by `predict_impact.predict_change_impact`: Amdahl
  guard (kernel < 5% runtime ⇒ no emission), tier-2 gate (missing
  counters ⇒ no emission), conservative bracket
  (`hi = catalog_hi × 0.85`), and every rec cites the seed entry in
  `knowledge/proven_optimizations.yaml` via `source_citation`.
  `_format_as_json` bumps `schema_version` from `0.3.2` → `0.3.3` when
  any rec in the document carries
  `predicted_impact_range != null`. `0.3.4` (Live Roofline) and ATT
  (`0.4.0`) still trump when their own fields are populated.
- `0.3.2` — additive top-level `communication` section
  emitted by `rccl_analysis.analyze_collectives` (RCCL / NIC
  communication analysis). Shape:
  ```json
  "communication": {
    "collectives": [
      {
        "op_type": "AllReduce",
        "msg_bytes": 1048576,
        "duration_ns": 1000000,
        "effective_bw_gbps": 1.57,
        "peak_bw_gbps": 340.0,
        "efficiency_pct": 0.46,
        "efficiency_label": "poor",
        "overlap_ratio": 48.5,
        "algo_hint": "Ring",
        "topology_hint": "intra-node",
        "regime": "algo-dependent",
        "ranks": 4
      }
    ],
    "summary": {
      "op_count": 1,
      "ranks": 4,
      "dominant_op": "AllReduce",
      "peak_bw_gbps": 340.0,
      "avg_bw_gbps": 1.57,
      "avg_efficiency_pct": 0.46,
      "overlap_pct": 48.5,
      "capture_incomplete": false
    }
  }
  ```
  The key is absent (not `null`) when the trace contains no RCCL
  spans. When set, `_format_as_json` bumps `schema_version` from
  `0.3.1` -> `0.3.2`. ATT (`0.4.0`) still trumps: a trace with both
  ATT + RCCL data pins `0.4.0`.
- `0.4.0` — bumped automatically by `_format_as_json` when
  `att_analysis.has_att_data=True` (Tier-3 ATT).

The bumps in `_format_as_json` live at the end of that function; the
overlay in `_format_agentic_output` (`perfxpert/analyze.py` ~line 790)
only upgrades `0.1.0` / `0.2.0` → `0.3.0` by default and conditionally
bumps to `0.3.1` when any hotspot carries `source_locations`, so a
later ATT-driven bump to `0.4.0` is preserved. Preserve that ordering
when you add a new minor / patch bump.

## Cross-format parity guarantee

Every top-level key in the JSON output must have a visible
representation in `text` + `markdown` + `webview`. If a section is
structurally empty (`memory_analysis` on a compute-only trace,
`hardware_counters` on a Tier-1 run), formatters render a graceful
placeholder, **not** an error. The regression guard in
`test_json_has_all_required_keys_and_bumped_schema_version`
(`tests/test_formatters/test_report_structure.py`) plus the
webview/markdown/text ordering tests enforce this.

## Recommendation subtype — `pragma`

Recommendations now carry an optional `subtype` field in the Layer B
dict (and Layer C JSON). The only defined value today is `"pragma"`,
emitted exclusively by the LLVM loop-hint advanced-recommendations
pathway:

```json
{
  "priority": "MEDIUM",
  "subtype": "pragma",
  "category": "Loop Hint (clang_loop_unroll_count)",
  "issue": "...",
  "suggestion": "Apply `#pragma clang loop unroll_count(N)` at src/hot.hip:42",
  "actions": ["...", "Verify with: perfxpert diff -i <baseline>.db -i <new>.db"],
  "estimated_impact": "1.1x-1.5x",
  "source_file": "src/hot.hip",
  "source_line": 42,
  "pragma_id": "clang_loop_unroll_count",
  "factor_sweep": [2, 4, 8],
  "risk": "medium"
}
```

Rendering invariants:

* The text / markdown / webview formatters render an ``[advanced]``
  badge next to the priority label for every rec with
  ``subtype == "pragma"``.
* ``perfxpert analyze`` filters pragma recs out of the rendered
  output unless ``--advanced`` is passed or
  ``PERFXPERT_ADVANCED_RECS=1`` is set — the Layer-B payload still
  includes them (so the JSON-contract version is available to
  downstream consumers who want the raw list).
* Every pragma rec MUST carry a Tier-0 source anchor
  (``source_file`` + ``source_line``); this is enforced by the fence
  slice in ``perfxpert/agents/fence/compute_specialist.md``.

See ``perfxpert/analysis/recommendations.py::build_pragma_recommendation``
for the canonical constructor.

## Advanced-specialist tool shapes

The advanced-specialist MCP tools (kernel fusion, GPU runtime monitor,
unified memory, dependency graph) each return a plain dict; none carry
`schema_version` because they sit alongside — not inside — the
analyze-report JSON. Consumers that want schema-stable outputs should
pin the tool's release notes in `docs/integration/mcp-server.md`.

### `kernel_fusion.find_fusion_candidates` (Feature A)

Returns `List[Dict]` ranked by `est_speedup_hi` descending:

```json
{
  "pair": ["add_kernel", "add_kernel"],
  "signature": "9b7a2e4c1ff0",
  "gap_ns": 120,
  "est_speedup_lo": 1.08,
  "est_speedup_hi": 1.25,
  "confidence": 0.78
}
```

### `gpu_runtime_monitor.*` (Feature B)

`parse_amd_smi_json` / `parse_rocm_smi_json` return a normalised
envelope:

```json
{
  "source": "amd-smi",
  "samples": [
    {"gpu_id": 0, "temp_c": 72.5, "power_w": 410.0,
     "sclk_mhz": 1950, "mclk_mhz": 1600, "gfx_busy_pct": 96}
  ],
  "gpu_count": 1
}
```

`analyze_thermal` consumes the envelope and returns:

```json
{
  "max_temp_c": 72.5,
  "avg_temp_c": 71.2,
  "p95_temp_c": 72.5,
  "max_power_w": 410.0,
  "throttle_events": 0,
  "power_throttle_suspected": false,
  "thermal_headroom_c": 32.5,
  "verdict": "healthy"
}
```

### `unified_memory.analyze_paging` (Feature C)

```json
{
  "paging_events": 3,
  "cross_die_bytes": 2147483648,
  "page_faults": 0,
  "estimated_penalty_ns": 1006632960,
  "recommendations": ["Pin host-shared buffers ...", "..."]
}
```

### `dependency_graph.reconstruct_dag` (Feature D)

```json
{
  "nodes": [{"id":"k0","name":"matmul","start_ns":0,"end_ns":5000,"duration_ns":5000,"stream":0}],
  "edges": [{"from":"k0","to":"k1","kind":"stream_order"}],
  "critical_path": ["k0", "k1"],
  "bubbles": [{"start":5000,"end":15000,"cause":"idle_gap_stream_0","duration_ns":10000}],
  "total_bubble_ns": 10000,
  "sync_event_count": 2
}
```

## Roofline payload

<a name="roofline-payload"></a>

The `roofline` top-level key (added in `schema_version: 0.3.4`) carries
the per-kernel live-roofline points produced by
`perfxpert.tools.roofline.plot_points`. Full shape and
bumping-semantics are documented in the Layer-C version-table above
(see **0.3.4** under "Schema versioning policy"). Key points:

- Populated only when Tier-2 hardware counters
  (`SQ_INSTS_VALU` / `FETCH_SIZE` / `WRITE_SIZE`) are available.
  Absent otherwise (formatters skip the section gracefully).
- `arch_peaks` is in FLOPs/s (not TFLOPs) — the formatter divides.
- `ridge_point.ai` = dominant-dtype peak / HBM bandwidth
  (FLOPs/Byte); `ridge_point.flops_per_s` = dominant-dtype peak.
- `kernels[i].confidence` is `"low"` when only one side of the byte
  pair (FETCH_SIZE / WRITE_SIZE) was collected — the other is
  assumed 0, so AI is an upper bound.
- The webview formatter renders the Live Roofline as an inline SVG
  `<section class="scard">`; the Markdown / text formats render a
  table with columns (kernel, AI, achieved GFLOPs/s, regime, dtype,
  confidence); the JSON format is a passthrough of this shape.

## Tier-0 carve-out

`tier0_findings` is conditionally included — it's only present when
`--source-dir` was supplied. Even then, the instrumentation-advice
sub-fields are conditionally **stripped** when a DB is also present
(`has_profiling=True`). The exact keys stripped, from
`perfxpert/analyze.py` ~line 776:

```json
{
  "suggested_counters": "...",
  "profiling_plan": "...",
  "profiling_plan_actions": "...",
  "suggested_first_command": "..."
}
```

The rationale: in combined mode (`-i` + `--source-dir`) the user
already has profiling data. Suggesting they run `rocprofv3` again
with a hand-picked counter set is noise — the Analysis agent uses
the existing DB instead. In source-only mode those sub-fields are
the whole point of Tier 0 output, so they stay.

Every `_format_tier0_<fmt>` accepts `has_profiling: bool = False`
and gates those sub-fields internally. New tier-0 formatters MUST
honour the same gate.

## Reviewer checklist

- [ ] Agent-schema change respects field caps (≤10 input, ≤5 output
      per model) — CI test `tests/test_agents/test_schema_field_caps.py`
      passes.
- [ ] Agent-schema change populated in the agent's `run_*` function,
      not just declared on the model.
- [ ] Rendered in **all four** built-in formatters (text, json_fmt,
      markdown, webview) OR explicitly documented as internal-only.
- [ ] Python-API docstring updated if the field is agent-output
      (reflected through `perfxpert.api.agent_<name>`).
- [ ] `docs/guides/getting-started.md` "Report structure" subsection
      mentions the new key if it's user-facing.
- [ ] `schema_version` bumped iff Layer C (JSON doc) shape changed;
      use major/minor/patch per the policy above.
- [ ] Regression guard added to
      `tests/test_formatters/test_report_structure.py` (presence +
      ordering, or `schema_version` assertion).
- [ ] `has_profiling` gate respected for any new tier-0 sub-field.
- [ ] `scripts/lint.sh --strict`, `scripts/link-checker.py --strict`,
      `scripts/test-samples.py --strict` all return rc=0.

## See also

- [output_formats.md](output_formats.md) — the companion guide for
  format authors consuming the schema you just evolved.
- `perfxpert/agents/schemas.py` — Layer A models.
- `perfxpert/analysis/payload.py` — Layer B builder.
- `perfxpert/formatters/json_fmt.py` — Layer C serialiser.
- `tests/test_formatters/test_report_structure.py` — the regression
  guards for all three layers' final render.
- `tests/test_agents/test_schema_field_caps.py` — the field-cap
  gate for Layer A.
