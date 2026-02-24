# ROCpd AI Analysis Output - JSON Schema Changelog

This document tracks all changes to the JSON output schema for `rocpd analyze --format json`
and the `rocpd.ai_analysis` Python API.

## Versioning Policy

The schema follows **Semantic Versioning** (`MAJOR.MINOR.PATCH`):

| Change type | Version bump | Example |
|---|---|---|
| New required field, renamed field, type change, removed field | **MAJOR** | `0.x.x` → `1.0.0` |
| New optional field added | **MINOR** | `0.1.x` → `0.2.0` |
| Description/example correction, no structural change | **PATCH** | `0.1.0` → `0.1.1` |

> **Beta notice**: While `MAJOR` is `0` the schema is in beta. Minor versions may include
> breaking changes without a MAJOR bump. Consumers should pin to an exact version during beta.

**Compatibility rule**: A consumer written for schema version `0.x.x` MUST continue to work
on any `0.y.z` output where `y >= x` (except during MAJOR=0 beta where minor may break).
MAJOR version changes always require consumer updates.

## How to Check the Schema Version

Every JSON output document contains a top-level `schema_version` field:

```json
{
  "schema_version": "0.1.0",
  ...
}
```

**Recommended consumer pattern**:

```python
import json

with open("analysis.json") as f:
    data = json.load(f)

ver = data["schema_version"]
major, minor, _ = (int(x) for x in ver.split("."))
if major != 0 or minor < 1:
    raise RuntimeError(
        f"Unsupported schema version {ver!r}. "
        "Expected 0.1.x. See SCHEMA_CHANGELOG.md for migration guidance."
    )
```

## Schema File Naming

Each released version is a standalone file:

```
rocpd/ai_analysis/docs/
├── analysis-output.schema.json   ← current (v0.1.0)
├── SCHEMA_CHANGELOG.md           ← this file
├── AI_ANALYSIS_API.md            ← Python API documentation
└── LLM_REFERENCE_GUIDE.md       ← copy of share/llm-reference-guide.md (for reference)
```

The current schema can always be located programmatically:

```python
import importlib.resources as pkg_resources
schema_path = pkg_resources.files("rocpd.ai_analysis") / "docs" / "analysis-output.schema.json"
```

---

## Version History

---

### v0.1.5 — 2026-02-24

**No schema changes.** Webview bug fix only.

- Fixed hover tooltip text being invisible in light theme. The `#tt` floating tooltip
  had `color:var(--text)` which in light mode resolves to `#181828` (near-black) —
  the same as the always-dark `#0e0e1c` tooltip background. Fixed by replacing
  `color:var(--text)` with a pinned light color `#dde0f2` so the tooltip is readable
  in both dark and light themes.

---

### v0.1.4 — 2026-02-24

**No schema changes.** Webview bug fix only.

- Fixed key findings bullet icons rendering as literal text (e.g. `&#8594;`) instead
  of the intended arrow character. Root cause: CSS `content` property does not process
  HTML entities — `content:'&#8594;'` outputs the 7-character string literally.
  Fixed by using the actual Unicode character `→` (U+2192) directly in the CSS rule.

---

### v0.1.3 — 2026-02-23

**No schema changes.** Webview UI/UX redesign only.

- Redesigned webview layout inspired by AMD dashboard design language:
  - **Light/Dark theme toggle** — persisted in `localStorage` (`rocpd-theme` key);
    defaults to dark. Header always uses AMD dark gradient regardless of theme.
  - **Status summary badges** in header — Critical/Warning/Low/Info counts derived
    from recommendations so key issues are visible before scrolling.
  - **Metric pills row** — Runtime (ms), kernel dispatch count, analysis tier, generation
    timestamp, and database file path shown in a compact pill bar below the main header.
  - **Status-colored KPI cards** — Four cards in the overview section (Kernel Execution,
    Primary Bottleneck, Total Runtime, Analysis Tier) each have a colored top border
    (`--c-ok`/`--c-warn`/`--c-crit`/`--c-info`) reflecting health status.
  - **Section card pattern** (`.scard`) — Each report section uses a consistent
    card layout with an icon-titled header (`.shdr`) and section-level badge.
  - **Priority icons on recommendations** — 🔴 HIGH, 🟠 MEDIUM, 🟡 LOW, ℹ INFO icons
    precede each recommendation badge for quicker visual scanning.
  - **FAB scroll-to-top button** — Floating action button appears after scrolling 250 px.
  - **`@keyframes fadeInUp`** staggered entrance animations on section cards.
  - **Gradient execution bars** — Breakdown segment bars use color gradients.
  - **Improved typography** — System font stack (`-apple-system`, `Segoe UI`, etc.) and
    monospace stack (`JetBrains Mono`, `Cascadia Code`, `Fira Code`) for offline use.
  - **Improved table headers** — Uppercase, 2 px bottom border.
  - **Gauge cards** — Background fill and border on hover for hardware counter gauges.
- No changes to JSON output structure, schema version string, or analysis logic.

---

### v0.1.2 — 2026-02-19

**No schema changes.** Webview presentation improvements only.

- Added hover tooltips to all visual elements in the `--format webview` HTML report:
  gauges, execution breakdown bars, overview stat cards, hotspot table column headers,
  memory transfer direction cells and column headers, and hardware counter table rows.
- Counter rows use a `COUNTER_TIPS` JavaScript lookup covering 20+ known AMD GPU
  hardware counters (GRBM_*, SQ_*, TCP/TCC cache, FETCH_SIZE, WRITE_SIZE, etc.)
  with educational content about what each counter measures and why it matters.
- Unknown counters receive a generic fallback tooltip pointing to AMD ISA documentation.
- No changes to JSON output structure, schema version string, or analysis logic.

---

### v0.1.1 — 2026-02-19

**No schema changes.** Description and tooling improvements only.

- Added `webview` output format (`--format webview`) producing a self-contained
  interactive HTML report. The underlying JSON data structure is unchanged; the HTML
  report embeds the same payload as `--format json`.
- CLI `--format` now automatically appends the correct file extension to the output
  file name: `.json`, `.md`, `.html`, or `.txt` depending on the selected format.
  No schema-level change.

---

### v0.1.0 — 2026-02-18

**Initial beta release.**

#### Document structure

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | `string` `"0.1.0"` | ✅ | Always present; check before parsing |
| `metadata` | object | ✅ | Analysis run metadata |
| `profiling_info` | object | ✅ | Profiling session info |
| `summary` | object | ✅ | Bottleneck classification |
| `execution_breakdown` | object | ✅ | Time distribution in ns and % |
| `hotspots` | array | ✅ | Top kernels by total time |
| `memory_analysis` | object | ✅ | Per-direction memory copy stats |
| `hardware_counters` | object | ✅ | Tier 2 counter data (may be empty) |
| `recommendations` | array | ✅ | Prioritized optimization suggestions with structured commands |
| `warnings` | array | ✅ | Analysis quality warnings |
| `errors` | array | ✅ | Non-fatal errors (empty = success) |
| `llm_enhanced_explanation` | `string\|null` | — | Optional LLM text; null when not used |

#### `metadata` fields

| Field | Type | Notes |
|---|---|---|
| `rocpd_version` | `string` | e.g. `"6.3.0"` |
| `analysis_version` | `string` | SemVer, e.g. `"0.1.0"` |
| `database_file` | `string` | Path to analyzed `.db` file |
| `analysis_timestamp` | `string` | ISO 8601 |
| `analysis_duration_ms` | `integer` | Wall-clock analysis time |
| `custom_prompt` | `string\|null` | Value of `--prompt`, or null |

#### `profiling_info` fields

| Field | Type | Values |
|---|---|---|
| `total_duration_ns` | `integer` | Wall-clock duration of profiled app |
| `profiling_mode` | `string` | `sys_trace_only`, `sys_trace_with_counters`, `pc_sampling`, `thread_trace` |
| `analysis_tier` | `integer` | `1`–`4` |
| `gpus` | array of GPU objects | Each: `name`, `architecture`, `agent_id` |

#### `summary` fields

| Field | Type | Values |
|---|---|---|
| `overall_assessment` | `string` | Free text |
| `primary_bottleneck` | `string` | `compute`, `memory_transfer`, `memory_bandwidth`, `latency`, `mixed`, `unknown` |
| `confidence` | `number` | `0.0`–`1.0` |
| `key_findings` | `string[]` | Ordered, most significant first |

#### `execution_breakdown` fields

All time fields are **nanoseconds** (`_ns`). All percentage fields are `_pct` (`0.0`–`100.0`).

| Field | Description |
|---|---|
| `total_runtime_ns` | `MAX(end) - MIN(start)` across all operations |
| `kernel_time_ns` / `kernel_time_pct` | GPU kernel execution |
| `memcpy_time_ns` / `memcpy_time_pct` | All memory copies (all directions) |
| `api_overhead_ns` / `api_overhead_pct` | API and launch overhead |
| `idle_time_ns` / `idle_time_pct` | GPU idle gaps |

#### `hotspots` item fields

| Field | Type | Notes |
|---|---|---|
| `rank` | `integer` | 1-based, 1 = hottest |
| `name` | `string` | Demangled kernel name |
| `calls` | `integer` | Total dispatch count |
| `total_duration_ns` | `integer` | Sum of all dispatch durations |
| `avg_duration_ns` | `number` | Mean dispatch duration |
| `min_duration_ns` | `integer` | Minimum dispatch duration |
| `max_duration_ns` | `integer` | Maximum dispatch duration |
| `pct_of_total` | `number` | % of `total_runtime_ns` |

#### `memory_analysis` keys and value fields

Keys are transfer direction strings: `"Host-to-Device"`, `"Device-to-Host"`, `"Device-to-Device"`, `"Peer-to-Peer"`, `"Unknown"`.

| Field | Type | Notes |
|---|---|---|
| `count` | `integer` | Number of copy operations |
| `total_bytes` | `integer` | Total bytes transferred |
| `total_duration_ns` | `integer` | Total copy time |
| `avg_bytes` | `number` | Average transfer size |
| `avg_duration_ns` | `number` | Average copy duration |
| `bandwidth_gbps` | `number` | Achieved bandwidth in GB/s |

#### `hardware_counters` fields

| Field | Type | Notes |
|---|---|---|
| `has_counters` | `boolean` | Check this before using other fields |
| `metrics` | object or null | Derived metrics (GPU util%, waves) |
| `metrics.gpu_utilization_pct` | `number\|null` | From GRBM_GUI_ACTIVE/GRBM_COUNT |
| `metrics.avg_waves` | `number\|null` | From SQ_WAVES |
| `metrics.max_waves` | `number\|null` | |
| `metrics.min_waves` | `number\|null` | |
| `counters` | object or null | Raw counter stats keyed by counter name |
| `counters.<name>.sample_count` | `integer` | |
| `counters.<name>.avg_value` | `number` | |
| `counters.<name>.min_value` | `number` | |
| `counters.<name>.max_value` | `number` | |
| `counters.<name>.total_value` | `number` | |

#### `recommendations` item fields

| Field | Required | Notes |
|---|---|---|
| `id` | ✅ | Stable ID, e.g. `"ROCPD-MEMCPY-001"` |
| `priority` | ✅ | `HIGH`, `MEDIUM`, `LOW`, or `INFO` |
| `category` | ✅ | e.g. `"Memory Transfer"`, `"Compute Bottleneck"` |
| `issue` | ✅ | What was detected (with measurements) |
| `suggestion` | ✅ | What to do |
| `actions` | — | Ordered implementation steps |
| `estimated_impact` | — | Expected performance gain |
| `commands` | — | Structured per-tool profiling commands (see below) |

#### `recommendations[].commands` item fields

Each item represents one invocation of a ROCm profiling tool:

| Field | Type | Required | Notes |
|---|---|---|---|
| `tool` | `string` | ✅ | `"rocprofv3"`, `"rocprof-sys"`, or `"rocprof-compute"` |
| `description` | `string` | ✅ | Why this command is recommended for the specific issue |
| `flags` | `string[]` | ✅ | Boolean flags (no value), e.g. `["--sys-trace", "--hsa-trace"]` |
| `args` | `object[]` | ✅ | Named args; each has `name` (string) and `value` (`string\|null`) |
| `full_command` | `string` | ✅ | Complete ready-to-run command with `-- ./app` placeholder |

**Tool meanings**:
- `rocprofv3` — ROCm trace and counter collection (successor to rocprof)
- `rocprof-sys` — System-level profiling (Omnitrace) with timeline visualization
- `rocprof-compute` — Kernel-level hardware counter deep-dive analysis

#### `warnings` item fields

| Field | Required | Values |
|---|---|---|
| `severity` | ✅ | `"warning"` or `"info"` |
| `message` | ✅ | Human-readable text |
| `recommendation` | — | How to resolve |

#### Known limitations in v0.1.0

- `execution_breakdown.api_overhead_ns` is derived from `overhead_percent` of `total_runtime_ns` and is clamped to `0` internally. Similarly, `idle_time_ns` is clamped to `0`. Both fields are always non-negative.
- `profiling_info.gpus` may be an empty array when GPU info is not yet populated from the database.
- `hardware_counters.metrics.gpu_utilization_pct` requires both `GRBM_COUNT` and `GRBM_GUI_ACTIVE` counters to be collected. If only one is present, the field is `null`.

---

## Planned Future Versions

The following are **not committed** but represent the current design direction:

### v0.2.0 (planned)
- Add `hotspots[].counters` — per-kernel hardware counter breakdown (Tier 2)
- Add `profiling_info.gpus[].peak_fp64_tflops` and `peak_hbm_bandwidth_gbps`
- Add `summary.roofline` — arithmetic intensity and roof classification

### v0.3.0 (planned)
- Add `pc_sampling` section — instruction-level hotspots (Tier 3)

### v1.0.0 (planned — first stable release)
- Rename `recommendations[].issue` → `recommendations[].description` (aligns with Python API)
- Merge `recommendations` flat array into `recommendations.high_priority` / `medium_priority` / `low_priority` sub-arrays (aligns with `AnalysisResult` Python dataclass)
- Remove MAJOR=0 beta caveat from versioning policy

---

## Migration Guide

### From pre-schema outputs (before v0.1.0)

Pre-schema outputs from earlier development builds did not contain `schema_version`.
Detection heuristic:

```python
if "schema_version" not in data:
    # Legacy output — no structured parsing possible
    raise ValueError("Legacy output without schema_version is not supported.")
```

---

## Validation

To validate a JSON output document against this schema:

```bash
# Using jsonschema (pip install jsonschema)
python3 -m jsonschema \
  --instance analysis.json \
  rocpd/ai_analysis/docs/analysis-output.schema.json
```

```python
# Programmatic validation
import json
import jsonschema
import importlib.resources as pkg_resources

schema_text = (
    pkg_resources.files("rocpd.ai_analysis")
    .joinpath("docs/analysis-output.schema.json")
    .read_text()
)
schema = json.loads(schema_text)

with open("analysis.json") as f:
    instance = json.load(f)

jsonschema.validate(instance=instance, schema=schema)
print("Valid!")
```
