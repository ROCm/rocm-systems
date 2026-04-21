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
  `LatencySpecialistInput` / `LatencySpecialistOutput` — Layer-2 experts.

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
`schema_version` field (currently **`0.3.1`** — see `# CHANGES`
below; Tier-3 ATT bumps to `0.4.0`). `perfxpert/analyze.py::_format_agentic_output`
(~line 790) then overlays the agentic brain (`narrative`,
`primary_bottleneck`, `warnings`, `tier0_findings`) and re-bumps the
version if it was still at `0.1.0` / `0.2.0`.

<!-- # CHANGES — 0.3.0 → 0.3.1
     Additive: `hotspots[i].source_locations: list[{file, line, kind}]`
     where `kind ∈ {"definition", "launch"}`. Emitted when
     `--source-dir` was supplied and the Tier-0 scanner correlated
     at least one hotspot with a detected kernel. Absent field when
     no source scan was performed; empty list when the scanner ran
     but no basename matched. See Confluence row #5 (Source Code
     Line numbers) for the UI rollout details. -->


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

6. **MCP tool discovery is automatic.** The 7 agent tools
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
- `0.3.1` — current: additive `hotspots[i].source_locations` field
  cross-referencing each hotspot with its Tier-0 definition +
  launch site (Confluence row #5).
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
