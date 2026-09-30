# Contributing: new output format

## What you're adding

A new renderer for `perfxpert analyze --format <name>`. The four built-ins
(`text`, `json`, `markdown`, `webview`) cover ~all user-facing needs — a
new format is a design-review-worthy addition and should solve a concrete
ingestion problem the existing four cannot.

## When to add a new format

Plausible candidates that have come up:

- `csv` — drop-in for spreadsheet ingestion (hotspots + time-breakdown rows).
- `html-embedded-json` — pipe the canonical doc into a third-party dashboard
  (Datadog, Grafana Trace Panel) while keeping an HTML shell for humans.
- `slackdown` — paste-into-chat, Slack-flavoured subset of Markdown (no
  tables, bold via `*…*`, hotspot list as bullets).
- `ansi` — coloured terminal stream for `perfxpert-code` live rendering.

If what you want is "the JSON doc but with one extra field," you do
**not** need a new format — evolve the schema instead (see
[schemas.md](schemas.md)).

## Architecture overview

The render pipeline is four stages:

1. **Deterministic pass** — `perfxpert/analysis/payload.py:build_analysis_payload()`
   returns the canonical dict (`time_breakdown`, `hotspots`,
   `memory_analysis`, `hardware_counters`, `kernel_resources`,
   `api_overhead`, `warmup_issues`, `thread_trace`, `tier0_findings`,
   `recommendations_deterministic`, `metadata`). Runs for every format.
2. **Agentic brain + merge** — `perfxpert/agents/root.py::run_root` produces
   a `RootOutput` (`narrative`, `primary_bottleneck`, `recommendations`,
   `warnings`, `metadata`). Deterministic + LLM recommendations are merged
   by `perfxpert/analysis/payload.py::merge_recommendations` (dedupes by
   `(type, target)` / `(category, issue)`).
3. **Dispatch** — `perfxpert/analyze.py::_format_agentic_output`
   (~line 587) branches on `output_format` and calls the format-specific
   `_format_as_<fmt>()`. Source-only mode (`--source-dir` without `-i`)
   dispatches through `perfxpert/formatters/__init__.py::format_analysis_output`
   with `source_only=True` straight to `_format_tier0_<fmt>()`.
4. **Render** — the format-specific module builds the final string.

The `has_profiling: bool = False` gate on every `_format_tier0_<fmt>()`
strips instrumentation-advice blocks (Profiling Plan, Suggested Hardware
Counters, Suggested First Command) when a DB is present — those only
make sense in the source-only path.

## File layout

- Dispatcher: `perfxpert/analyze.py::_format_agentic_output` (~line 587).
- Extension map: `perfxpert/analyze.py::_ext_map` (~line 1392).
- CLI flag: `perfxpert/analyze.py::add_args` (~line 180) — `--format` choices.
- Format modules: `perfxpert/formatters/{text,json_fmt,markdown,webview}.py`.
- Package re-exports: `perfxpert/formatters/__init__.py` (`__all__`,
  `format_analysis_output` dispatcher).
- Webview HTML template: `perfxpert/formatters/templates/webview.html`
  (`%%PLACEHOLDER%%` substitution).
- Structural tests: `tests/test_formatters/test_report_structure.py`.

## Step-by-step recipe

### 1. Register the format name

Add the choice to the `--format` argparse declaration in
`perfxpert/analyze.py::add_args`:

```python
# SKIP-SAMPLE — illustrative
analysis_options.add_argument(
    "--format",
    type=str,
    dest="output_format",
    choices=["text", "json", "markdown", "webview", "csv"],  # add "csv"
    default="text",
    help="Output format: ... or csv. File extension is set automatically.",
)
```

And extend `_ext_map` in the same file (~line 1392):

```python
# SKIP-SAMPLE — illustrative
_ext_map = {
    "json": ".json",
    "markdown": ".md",
    "webview": ".html",
    "text": ".txt",
    "csv": ".csv",
}
```

### 2. Create `perfxpert/formatters/<fmt>.py`

Two functions. Use `markdown.py` as the minimal template — it has no HTML
escaping, no JSON schema, just string assembly.

```python
# SKIP-SAMPLE — illustrative skeleton for a new `csv` format
"""CSV formatter for perfxpert analyze."""

from typing import Any, Dict, List, Optional


def _format_as_csv(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    hardware_counters: Optional[Dict[str, Any]] = None,
    database_path: str = "",
    kernel_resources: Optional[Dict[str, Any]] = None,
    api_overhead: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the deterministic payload as flat CSV rows."""
    lines = ["section,name,calls,total_ms,avg_us,pct"]
    for k in hotspots or []:
        lines.append(
            f"hotspot,{k.get('name', '')},"
            f"{k.get('calls', 0)},"
            f"{k.get('total_duration', 0) / 1e6:.3f},"
            f"{k.get('avg_duration', 0) / 1e3:.3f},"
            f"{k.get('percent_of_total', 0):.2f}"
        )
    # memory_analysis, time_breakdown, recommendations rows follow.
    return "\n".join(lines) + "\n"


def _format_tier0_csv(tier0_result: Any, has_profiling: bool = False) -> str:
    """Render tier-0 findings as flat CSV rows.

    When ``has_profiling=True`` the instrumentation-advice blocks
    (profiling_plan, suggested_counters, suggested_first_command) are
    suppressed — they only make sense in the source-only path.
    """
    lines = ["section,kernel,file,line,pattern,severity"]
    for k in tier0_result.detected_kernels or []:
        lines.append(
            f"kernel,{k.get('name', '')},{k.get('file', '')},"
            f"{k.get('line', '')},,"
        )
    for p in tier0_result.detected_patterns or []:
        lines.append(
            f"pattern,,,,{p.get('pattern_id', '')},"
            f"{p.get('severity', 'info')}"
        )
    if not has_profiling:
        for c in tier0_result.suggested_counters or []:
            lines.append(f"counter_suggestion,,,,{c},")
    return "\n".join(lines) + "\n"
```

### 3. Re-export from `perfxpert/formatters/__init__.py`

```python
# SKIP-SAMPLE — illustrative
from .csv import _format_as_csv, _format_tier0_csv

__all__ = [
    # ... existing entries ...
    "_format_as_csv",
    "_format_tier0_csv",
]
```

### 4. Wire into `format_analysis_output()` (source-only + combined paths)

`perfxpert/formatters/__init__.py::format_analysis_output` handles the
source-only branch (`source_only=True, tier0_result is not None`) and
the combined branch. Add your format to both:

```python
# SKIP-SAMPLE — illustrative
if source_only and tier0_result is not None:
    if output_format == "csv":
        return _format_tier0_csv(tier0_result)
    # ... existing branches ...

if output_format == "csv":
    output = _format_as_csv(
        time_breakdown=time_breakdown,
        hotspots=hotspots,
        memory_analysis=memory_analysis,
        recommendations=recommendations,
        hardware_counters=hardware_counters,
        database_path=database_path,
        kernel_resources=kernel_resources,
        api_overhead=api_overhead,
    )
    if tier0_result is not None:
        output += "\n" + _format_tier0_csv(
            tier0_result, has_profiling=bool(database_path)
        )
    return output
```

### 5. Wire into `_format_agentic_output()`

`perfxpert/analyze.py::_format_agentic_output` (~line 587) is the
dispatcher used by the real CLI path. It splices the agent brain
(`narrative`, `primary_bottleneck`, `warnings`) on top of the
deterministic skeleton. Add a branch alongside the existing `json` /
`markdown` / `webview` / text branches, both in the `tier0_only`
shortcut and in the main path:

```python
# SKIP-SAMPLE — illustrative branch inside _format_agentic_output
if output_format == "csv":
    from .formatters import _format_as_csv
    body = _format_as_csv(
        time_breakdown=time_breakdown,
        hotspots=hotspots,
        memory_analysis=memory_analysis,
        recommendations=merged_recs,
        hardware_counters=hardware_counters,
        database_path=database_path,
        kernel_resources=kernel_resources,
        api_overhead=api_overhead,
    )
    # Prepend a `summary,<primary_bottleneck>,<narrative escaped>` row
    # so the agent brain is carried into the CSV contract.
    return f"summary,{primary_bottleneck},\"{narrative}\"\n" + body
```

### 6. Structural invariants test

Add a test in
`tests/test_formatters/test_report_structure.py` modelled on the
existing four. Use the `airgap` + `tiny_hip_src` fixtures and assert
the top-level section ordering plus the `has_profiling` gate:

```python
# SKIP-SAMPLE — illustrative pytest for a new `csv` format
def test_csv_has_summary_and_hotspots_and_honours_has_profiling(
    airgap, tiny_hip_src, tmp_path
):
    out = _render(fmt="csv", tmp_path=tmp_path, source_dir=tiny_hip_src)

    # Summary row precedes hotspot rows.
    i_summary = out.find("summary,")
    i_hotspot = out.find("hotspot,")
    assert 0 <= i_summary < i_hotspot, (
        "summary row must precede hotspot rows"
    )

    # has_profiling gate — with -i set, counter_suggestion rows are
    # suppressed (only source-only mode emits them).
    assert "counter_suggestion" not in out, (
        "instrumentation advice leaked into combined mode"
    )
```

### 7. Getting-started screenshot (optional)

If the format warrants a visual, update the checked-in sanity fixture
under `docs/guides/assets/gifs/04-analyze-<fmt>.gif` to match the
guide GIF style. Skip this step for machine-only formats (csv,
ndjson).

## Contract checklist

Every format MUST surface these sections (or an explicit
graceful-placeholder when a section is structurally empty):

| # | Section | Source key |
|---|---------|------------|
| 1 | Primary bottleneck | `primary_bottleneck` (from `RootOutput`) |
| 2 | Narrative | `narrative` (from `RootOutput`) |
| 3 | Time breakdown | `time_breakdown` |
| 4 | Hotspot list | `hotspots` |
| 4a | Source-line correlation (conditional) | `hotspots[i].source_locations` (present when `--source-dir` supplied; list of `{file, line, kind, severity, severity_label}`) |
| 5 | Memory analysis | `memory_analysis` |
| 6 | Hardware counters | `hardware_counters` (placeholder when `has_counters=False`) |
| 7 | Kernel resources | `kernel_resources` |
| 8 | API overhead | `api_overhead` |
| 9 | Recommendations (merged LLM + deterministic) | `recommendations` |
| 10 | Warnings | `warnings` (from `RootOutput`) |
| 11 | Tier-0 findings (conditional) | `tier0_findings` (only when `--source-dir` supplied) |
| 12 | Changed-vs-baseline (conditional) | `trace_diff` top-level (only when `--baseline <db>` supplied OR when rendering a standalone `perfxpert diff` report) |
| 13 | Thread Trace + ATT Flamegraph (conditional) | `thread_trace` top-level + inline-SVG flamegraph rendered in webview from `thread_trace.kernels`/`top_stalling_instructions` (only when ATT data is present — schema pins `0.4.0`) |
| 14 | Live Roofline (conditional) | `roofline` top-level from `roofline.plot_points`: per-kernel `(ai, achieved_flops_per_s, bottleneck_class, fp_type, confidence)` + `ridge_point`; webview renders inline-SVG log-log plot, markdown/text render a table (schema pins `0.3.4`) |
| 15 | Communication (conditional) | `communication` top-level from `rccl_analysis.analyze_collectives`: per-collective `effective_bw_gbps`/`efficiency_pct`/`efficiency_label`/`overlap_ratio` + rollup `summary` (schema pins `0.3.2`) |

### Hotspot → source correlation (Confluence row #5)

When `--source-dir` is supplied, the Tier-0 scanner populates
`detected_kernels: [{name, file, line, launch_type}]`. The
formatter-agnostic helper
`perfxpert/formatters/_source_correlation.py::correlate_hotspots_with_source`
canonicalises both names (strip namespace, template args, function-
pointer decoration; match by basename, case-insensitive) and returns
each hotspot with a `source_locations: list[{file, line, kind}]`
field, sorted definitions-first, then launches.

Every built-in format surfaces this data:

- **webview** — expandable `▾` per-row panel (class `h-src-toggle`)
  inside the Top Kernel Hotspots `.scard`; shows Definition +
  Launch-site lines each with a Copy button and a launch-type badge.
- **markdown** — appends a `- Source: file.hip:42 (definition), …`
  line under each hotspot row in the `## Top Kernel Hotspots` table.
- **text** — appends a `      Source: file.hip:42 (definition), …`
  indented line under each hotspot row in the HOTSPOTS banner.
- **json** — emits `hotspots[i].source_locations` verbatim (schema
  `0.3.1`).

**CSV authors**: when `--source-dir` was supplied, every hotspot row
MUST include at minimum a `source_file` + `source_line` + `source_kind`
triple (or repeat the hotspot row once per source-location, with a
`rank.source_idx` column to preserve ordering). Omitting these in a
new format creates cross-format drift — the JSON contract and the
webview both surface the correlation, so a machine-oriented format
that drops it is a regression.

Empty sections render a placeholder, not an error — `memory_analysis`
on a compute-only trace, `hardware_counters` on a Tier-1 (no-PMC) run,
etc. Missing sections DO crash: if `build_analysis_payload` would have
set the key, the formatter must handle it.

### Severity coloring (Confluence row #5 refinement)

Each entry in `hotspots[i].source_locations` carries a `severity` and
`severity_label` pair derived from the owning hotspot's
`percent_of_total` (VTune / NSight-style buckets):

| Bucket | Threshold (`percent_of_total`) | `severity` | `severity_label` | Hex |
|---|---|---|---|---|
| Critical | ≥ 20% | HIGH | CRITICAL | `#e84040` |
| Hot | ≥ 5% | MEDIUM | HOT | `#f08432` |
| Warm | ≥ 1% | LOW | WARM | `#caa828` |
| Cool | < 1% | INFO | COOL | `#4d8ef2` |

Every format must propagate severity visually:

- **webview** — the `.h-src-row` panel has a `border-left:3px solid
  <hex>` + a pill-shaped `CRITICAL / HOT / WARM / COOL` badge. The hex
  palette matches the existing `.r-card[data-p=…]` recommendation-card
  palette; do not invent a new palette.
- **markdown / text** — citation line ends with `[CRITICAL]` /
  `[HOT]` / `[WARM]` / `[COOL]`.
- **json** — emits `severity` + `severity_label` verbatim inside each
  `source_locations` entry (in addition to `severity_color`).

### ATT Flamegraph (Thread Trace)

When `thread_trace.has_att_data == True`, the webview renders an
inline-SVG flamegraph directly under the Thread Trace `.scard`
(`perfxpert/formatters/_att_flamegraph.py`) — hot basic-blocks widen
proportionally to stall ns, coloured by `stall_category`. No new
payload key: the flamegraph is a pure-rendering derivative of the
existing `thread_trace.kernels` + `top_stalling_instructions`
subfields. Markdown / text / json keep the pre-existing stall table +
top-instructions output (they don't rerender the SVG). Schema-wise,
ATT presence bumps the top-level `schema_version` to `0.4.0`.

### Multi-DB runtime accounting

Multi-DB reports carry two runtime notions on purpose:

- `total_runtime` / `execution_breakdown.total_runtime_ns` stay wall-clock.
- `normalized_runtime` / `execution_breakdown.normalized_runtime_ns` represent
  the summed shard envelope used for percentage math when shards overlap.

Formatters MUST not mix these within a single breakdown table. If a formatter
shows normalized percentages for kernel / memcpy / API overhead, its total row
must also use normalized runtime and label that fact explicitly. Wall-clock
runtime may still be shown elsewhere as metadata, but not as the denominator of
the same normalized table.

### Live Roofline (`roofline.plot_points`)

When Tier-2 counters (`SQ_INSTS_VALU` / `FETCH_SIZE` / `WRITE_SIZE`)
are present, `perfxpert.tools.roofline.plot_points` produces a
top-level `roofline` key with per-kernel `(ai,
achieved_flops_per_s, bottleneck_class, fp_type, confidence)` plus
`arch_peaks` + `ridge_point`. The webview renders a log-log inline
SVG (no external JS/CSS — the SVG string is embedded directly into
the template). Markdown / text emit a per-kernel table with columns
`(kernel, AI, achieved GFLOPs/s, regime, dtype, confidence)`; json
is a passthrough. Populating the key bumps `schema_version` to
`0.3.4`; ATT still trumps.

### Communication (`rccl_analysis.analyze_collectives`)

When the trace contains RCCL collective spans,
`perfxpert.tools.rccl_analysis.analyze_collectives` produces a
top-level `communication` key with a per-collective list
(`op_type`, `msg_bytes`, `duration_ns`, `effective_bw_gbps`,
`peak_bw_gbps`, `efficiency_pct`, `efficiency_label`,
`overlap_ratio`, `algo_hint`, `topology_hint`, `regime`, `ranks`)
and a rollup `summary`. Validation at the boundary goes through
the Pydantic `CommunicationBlock` + `CollectiveEntry` models in
`perfxpert/agents/schemas.py`. Webview renders a table + stacked
efficiency bars; markdown emits a GFM table; text emits a
fixed-width banner + rows; json is a passthrough. Populating the
key bumps `schema_version` to `0.3.2` (ATT `0.4.0` + roofline
`0.3.4` still trump when their fields are also set).

### Diff reports (`perfxpert diff`, `perfxpert ci`, `analyze --baseline`)

Diff reports are a separate top-level surface. Every format that
participates in the diff surface MUST render, at a minimum:

1. An overview block (wall-delta %, regression count, improvement count).
2. A per-kernel regression-vs-improvement table (kernel name, baseline
   ns, new ns, delta %, hot-kernel flag).
3. A narrative paragraph.

The webview diff adds stacked delta bars (red for regressions, green
for improvements) as the VTune-style visual cue. Formats that cannot
render bars (text / markdown / json) MUST still surface the
regression-vs-improvement split — e.g. text puts primary regressions
and improvements in separate bullet lists within the narrative;
markdown uses a sortable GFM table + a fenced-code narrative block.

This is a HARD contract: a diff format that only shows "5 kernels
changed" without telling the reader which way is a regression gate
escape. The gate cascade consumes these same numbers via
`runtime.gate_cascade.trace_diff_regression_rule`, so a diff format
that hides direction also blinds the gate.

## Do not

- **Do not expose LLM routing prose.** Strings like `"delegating to
  Analysis agent"` or `"Let me proceed…"` are the Root agent's
  intermediate routing outputs — they must not reach the user. The
  `narrative` field is the final analysis synthesis; that's what you
  render.
- **Do not render instrumentation-advice when a DB is present.** Guard
  the Profiling Plan / Suggested Hardware Counters / Suggested First
  Command blocks with `has_profiling`. The JSON formatter already
  strips these keys when `database_path` is set (see
  `perfxpert/analyze.py` ~line 775); your format should follow suit.
- **Do not couple to `RootOutput` directly.** Read via
  `_format_agentic_output`'s local helper — that function accepts
  either a Pydantic `RootOutput` or a plain dict (the shape returned
  by `perfxpert.api.agent_root`) and normalises the access pattern.

## Testing

Three canned snippets cover the invariants every format must hold.

**Structural invariants** — section ordering + non-empty agent brain:

```python
# SKIP-SAMPLE — structural invariants for the new format
def test_<fmt>_structural_invariants(airgap, tiny_hip_src, tmp_path):
    out = _render(fmt="<fmt>", tmp_path=tmp_path, source_dir=tiny_hip_src)
    assert "<primary_bottleneck marker>" in out
    assert "<narrative marker>" in out
    # Ordering: summary/narrative precedes time-breakdown precedes hotspots.
    assert out.find("<summary marker>") < out.find("<time marker>")
    assert out.find("<time marker>") < out.find("<hotspot marker>")
```

**Cross-format schema parity** — every top-level key in the JSON doc
has a representation in the new format:

```python
# SKIP-SAMPLE — parity with JSON
def test_<fmt>_has_every_json_section(airgap, tmp_path):
    json_out = _render(fmt="json", tmp_path=tmp_path)
    fmt_out = _render(fmt="<fmt>", tmp_path=tmp_path)
    import json as _j
    doc = _j.loads(json_out)
    for section in ("time_breakdown", "hotspots", "memory_analysis",
                    "recommendations", "warnings"):
        if doc.get(section):
            assert "<marker-for-%s>" % section in fmt_out, (
                f"{section} present in JSON but not in <fmt> output"
            )
```

**Tier-0 gate regression** — instrumentation advice is stripped when a
DB is supplied:

```python
# SKIP-SAMPLE — has_profiling gate regression guard
def test_<fmt>_strips_instrumentation_advice_when_db_present(
    airgap, tiny_hip_src, tmp_path
):
    out = _render(fmt="<fmt>", tmp_path=tmp_path, source_dir=tiny_hip_src)
    # DB is the _FIXTURE_DB; has_profiling=True path.
    for banned in ("Suggested Hardware Counters",
                   "Profiling Plan",
                   "Suggested first command"):
        assert banned not in out, (
            f"instrumentation-advice leak: {banned!r} rendered with -i set"
        )
```

## Reviewer checklist

Before merging a new format:

- [ ] `--format <fmt>` accepted by argparse (`add_args` choices).
- [ ] `_ext_map` extended with the correct extension.
- [ ] Both `_format_as_<fmt>` and `_format_tier0_<fmt>` exported in
      `formatters/__init__.py::__all__`.
- [ ] `format_analysis_output()` dispatcher handles the new branch in
      **both** the source-only and combined paths.
- [ ] `_format_agentic_output()` dispatcher handles the new branch.
- [ ] `has_profiling` gate wired on `_format_tier0_<fmt>` — grep for
      the function definition and confirm `has_profiling=False` is the
      default and that instrumentation-advice blocks are skipped when
      `has_profiling=True`.
- [ ] `tests/test_formatters/test_report_structure.py` grows three
      new tests modelled on the existing four-format suite.
- [ ] No LLM routing prose leaks into output (`grep -n "delegating to"
      | "Let me proceed"` returns zero matches on the rendered
      fixture).
- [ ] `scripts/lint.sh --strict`, `scripts/link-checker.py --strict`,
      `scripts/test-samples.py --strict` all return rc=0.

## See also

- [schemas.md](schemas.md) — how to evolve the payload / schema the
  formatters consume.
- `perfxpert/analyze.py::_format_agentic_output` (line 587) — the live
  dispatcher.
- `perfxpert/formatters/__init__.py::format_analysis_output` — the
  legacy dispatcher used by source-only mode.
- `tests/test_formatters/test_report_structure.py` — the reference
  test suite your new format must mirror.
