# ROCInsight AI Analysis Output - JSON Schema Changelog

This document tracks all changes to the JSON output schema for `rocinsight analyze --format json`
and the `rocinsight.ai_analysis` Python API.

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

A single schema file covers all emitted versions via its `schema_version` enum:

```
rocinsight/ai_analysis/docs/
├── analysis-output.schema.json   ← single schema; schema_version enum lists all valid values
│                                    Tier 1/2 output emits: "0.1.0"
│                                    Tier 0 (source-only) output emits: "0.2.0"
│                                    Tier 1/2 with TraceLens fields emits: "0.3.0"
│                                    All valid values: ["0.1.0", "0.2.0", "0.3.0"]
│                                    New versions are added to the enum without breaking consumers
├── SCHEMA_CHANGELOG.md           ← this file
├── AI_ANALYSIS_API.md            ← Python API documentation
└── LLM_REFERENCE_GUIDE.md       ← copy of share/llm-reference-guide.md (for reference)
```

The current schema can always be located programmatically:

```python
import importlib.resources as pkg_resources
schema_path = pkg_resources.files("rocinsight.ai_analysis") / "docs" / "analysis-output.schema.json"
```

---

## Version History

---

## v0.3.2 — 2026-03-25

**No schema changes.** Branding, output routing, and LLM hardening only.

**Branding:**
- Webview `<title>`, HTML logo span, and markdown output headers changed from
  "ROCpd AI Performance Analysis" to "ROCInsight AI Performance Analysis".
- `analysis-output.schema.json` `title` field updated to "ROCInsight AI Analysis Output".

**Output routing fix (`-d` without `-o`):**
- When `-d <dir>` is given without `-o <name>`, rocinsight now auto-generates the output
  filename from the database basename (e.g. `merged_opt.db` → `merged_opt.html`).
  Previously the analysis was printed to the terminal even when `-d` was specified.

**LLM truncation retry (`_call_openai`):**
- When `_call_openai` receives empty content with `finish_reason=="length"` (reasoning
  models like gpt-5/o1/o3 exhaust `max_completion_tokens=4096` on thinking tokens),
  `analyze_with_llm` now retries with the system prompt filtered to only
  `<!-- rocinsight-context: always -->` sections and `max_completion_tokens=16384`.
- If the retry also fails, a clean user-facing error is raised (no internal sentinel leaking).
- If content is non-empty but truncated (`finish_reason=="length"`), partial content is
  returned with a warning instead of raising.

**Smarter `_select_tags` (guide filtering by bottleneck):**
- Tier 1 with a clear bottleneck (e.g. `memory_transfer`): only `{"always"}` — saves ~40% tokens.
- Tier 1 with unclear bottleneck (`mixed` or `None`): `{"always", "tier1"}` — unchanged.
- Tier 2 (counters): `{"always", "tier2"}` — drops `tier1` (was `{"always", "tier1", "tier2"}`).
- `compiler` tag: only added when bottleneck is `compute` OR user prompt mentions
  compiler/flag/build/register topics. No longer added for `memory` bottleneck.
- Tier 0 (source-only): `{"always", "tier1", "source"}` — unchanged.

**Bottleneck-aware `_format_data_for_llm`:**
- `memory_transfer` bottleneck: top-3 kernels (no counter fields), full memory ops section.
- `compute` bottleneck: top-3 kernels with counter fields, no memory ops section.
- `latency`/`api` bottleneck: top-3 kernels (no counter fields), no memory ops section.
- `mixed` or `None`: top-5 kernels with all fields, all sections (previous behavior).

---

## v0.3.1 — 2026-03-12

**No schema changes.** Schema file validator corrections, Python 3.6 compatibility fixes,
and LLM hardening only.

**Schema file corrections (v0.2.0 spec was already correct; JSON file had bugs):**

The `analysis-output.schema.json` file was corrected to match the already-documented
v0.2.0 specification.  The emitted JSON format was never wrong; only the validator was:

| Schema file bug | Fix |
|---|---|
| `profiling_info.profiling_mode` enum missing `"source_only"` | Added `"source_only"` as first enum value |
| `profiling_info.analysis_tier` `minimum` was `1` | Lowered to `0` to allow Tier 0 documents |
| `execution_breakdown` type was `"object"` only | Changed to `["object", "null"]` so source-only documents validate |
| `tier0` property not declared in `properties` object | Added full `tier0` property definition with all 14 sub-fields |
| `$id` embedded a version string (`"rocpd-ai-analysis-output-v0.1.0"`) | Changed to `"rocpd-ai-analysis-output"` (stable; version is in `schema_version` field) |

Tier 0 JSON output (schema_version `"0.2.0"`) now passes `jsonschema.validate()` against
the schema file.  28 JSON schema conformance tests added (was 17): 11 new tests cover
Tier 0 source-only output and combined (Tier 0 + Tier 1/2) output validation.

**Python 3.6 compatibility (`re.Pattern` annotation):**

`tracelens_port.py` used `re.Pattern` in a module-level type annotation
(`_CATEGORY_PATTERNS: List[Tuple[str, re.Pattern]]`).  Python 3.6 evaluates these
annotations eagerly at import time; `re.Pattern` was added in Python 3.7.  This caused
an `AttributeError` on RHEL 8.8 (Python 3.6.8) that cascaded into all tests importing
`analyze.py` or `llm_analyzer.py`.  Fixed by changing the annotation to `Any` (already
imported from `typing`).

`test_analyze_schema.py` used `import importlib.resources` which also requires Python 3.7.
Fixed with a `try/except ImportError` shim that falls back to `pkgutil.get_data()`.

**`ROCINSIGHT_LLM_PRIVATE_HEADERS` dict validation:**

After `json.loads()`, the parsed result is now validated to be a `dict` before
`headers.update()` is called.  A non-dict JSON value (e.g. `"[1,2,3]"`) previously
raised an opaque `TypeError`; it now raises a `ValueError` with a clear message
showing the expected format.

**Stream chunk accumulation (`LLMConversation`):**

Both `_stream_anthropic` and `_stream_openai` now accumulate response chunks with
`chunks.append(text)` + `"".join(chunks)` instead of `result += chunk` string
concatenation, avoiding O(n²) memory allocation for long responses.

---

## v0.3.0 (2026-03-11)

### New Fields (additive — old consumers should ignore unknown top-level keys)

- `interval_timeline` (object): GPU wall-time breakdown using set-theoretic interval arithmetic
  (TraceLens methodology). More accurate than `execution_breakdown` which sums raw durations.
  Fields: `total_wall_ns`, `true_compute_ns/pct`, `exposed_memcpy_ns/pct`, `idle_ns/pct`.

- `kernel_categories` (array): Kernel execution time aggregated by TraceLens op category
  (GEMM, CONV, SDPA, NCCL, Elementwise, Normalization, Reduction, Other).
  Fields per entry: `category`, `count`, `total_ns`, `pct_of_kernel_time`, `avg_duration_ns`, `pct_of_total_time`.

- `short_kernels` (object): Short kernel analysis — kernels below 10μs threshold.
  Fields: `threshold_us`, `total_kernels`, `short_kernel_count`, `short_kernel_pct`,
  `wasted_ns`, `wasted_pct_of_kernel_time`, `histogram`, `top_offenders`.

### Versioning Policy
Tier 1/2 runs now emit `schema_version: "0.3.0"` when tracelens fields are present.
Tier 0 source-only runs remain at `schema_version: "0.2.0"`.
Prior `"0.1.0"` documents are unaffected.

---

### v0.2.1 — 2026-03-10

**No schema changes.** Security, correctness, and LLM-layer bug fixes only.

This release documents behavioral changes that affect output values and API
consumers without altering the JSON document structure or field names.

**Output value guarantees (metadata field):**
- `analysis_version` in `metadata` now always reflects the schema version string
  (e.g. `"0.1.0"` for Tier 1/2 documents, `"0.2.0"` for Tier 0 source-only
  documents). The value was already correct in practice but is now explicitly
  documented as schema-tied. Consumer code should continue to read
  `schema_version` (not `analysis_version`) for compatibility checks.

**`execution_breakdown.api_overhead_pct` is now guaranteed ≥ 0:**
- `compute_time_breakdown()` now applies `max(0.0, ...)` to the raw `overhead_percent`
  before returning. In some traces where kernel + memcpy time marginally exceeded the
  computed total runtime (timestamp rounding), this field could previously be a small
  negative value. It is now always non-negative in both CLI JSON output and the
  Python API `ExecutionBreakdown.api_overhead_pct` field.

**`memory_analysis[direction].bandwidth_bytes_per_sec` and `bandwidth_gbps` now use actual sizes:**
- `analyze_memory_copies()` now reads the `size` column from `memory_copies` rows.
  Previously `total_bytes` was always 0 and bandwidth was not computed. Consumers
  that previously saw `bandwidth_gbps: 0` for all directions may now see non-zero
  values, and the "Low memory bandwidth" recommendation (< 10 GB/s) can now fire
  based on real measurements.

**`recommendations[].commands[].full_command` kernel names are now shell-safe:**
- In the "Compute Bottleneck" recommendation, `--kernel-names` arguments in
  `full_command` strings are now wrapped with `shlex.quote()`. Kernel names
  containing shell metacharacters (single quotes, semicolons, spaces) are properly
  escaped. The `args[].value` field is unchanged (stores the raw kernel name for
  display purposes).

**LLM API calls now include `timeout=120`:**
- All Anthropic and OpenAI API calls include an explicit 120-second timeout.
  Previously calls could hang indefinitely. A timed-out call is caught and recorded
  as a non-fatal warning; local analysis results are still returned.

**Tier 0 webview XSS protection:**
- `</script>` sequences in the embedded JSON payload of `_format_tier0_webview()`
  are now escaped to `<\/script>`. This prevents a crafted kernel name or LLM
  explanation from breaking out of the `<script>` block. No change to JSON output.

---

### v0.2.0 — 2026-03-09

**Tier 0: Static Source Code Analysis support.**

New fields and values added to support `rocinsight analyze --source-dir` (no database required):

| Change | Details |
|---|---|
| `schema_version` bumped | `"0.1.x"` → `"0.2.0"` for Tier 0 source-only documents |
| `profiling_info.profiling_mode` | New value `"source_only"` |
| `profiling_info.analysis_tier` | Now allows `0` (was minimum 1) |
| `metadata.database_file` | Now nullable (`null`) when running source-only |
| `execution_breakdown` | Now nullable (`null`) when running source-only |
| `tier0` (new optional field) | Present in Tier 0 and combined (Tier 0 + Tier 1/2) documents |

**`tier0` object schema:**
```json
{
  "source_dir": "string",
  "analysis_timestamp": "ISO 8601",
  "programming_model": "HIP | HIP+ROCm_Libraries | PyTorch_HIP | JAX_HIP | OpenCL | Python_GPU | Unknown",
  "files_scanned": 0,
  "files_skipped": 0,
  "kernel_count": 0,
  "detected_kernels": [{"name": "", "file": "", "line": 0, "launch_type": ""}],
  "detected_patterns": [{"pattern_id": "", "severity": "high|medium|low|info", "category": "", "description": "", "count": 0, "locations": []}],
  "risk_areas": ["string"],
  "already_instrumented": false,
  "roctx_marker_count": 0,
  "recommendations": [...],
  "suggested_counters": ["string"],
  "suggested_first_command": "string",
  "llm_explanation": null
}
```

**Tier 1/2 documents unchanged** — `schema_version` remains `"0.1.0"` for existing DB-only analysis. The `tier0` field is omitted from Tier 1/2 documents unless `--source-dir` is also provided (combined mode).

**Migration**: Consumers checking `schema_version == "0.1.0"` continue to work unchanged. Consumers wanting Tier 0 data should additionally handle `"0.2.0"` and check for the presence of the `tier0` field.

---

### v0.1.8 — 2026-02-27

**No schema changes.** Recommendation engine fix, OpenAI model compatibility, and test
infrastructure improvements.

**Recommendation engine — PMC counter deduplication:**
- Recommendations no longer suggest re-collecting hardware counters that are already
  present in `pmc_events`.  Previously, if a trace was collected with
  `--pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES`, the engine still emitted commands like
  `rocprofv3 --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES ...`.
- `_detect_already_collected()` now queries `SELECT DISTINCT counter_name FROM pmc_events`
  and adds `"pmc:<NAME>"` entries to the covered frozenset for each counter found.
- `_filter_rec_commands()` strips already-collected counters from `--pmc` arg values:
  - **Partial overlap**: only uncovered counters remain in `--pmc` (and in `full_command`)
  - **Full overlap**: `--pmc` arg and flag are dropped entirely
  - `--kernel-names` is now treated as a scope filter (not data collection); a command
    reduced to only `--kernel-names` and output path args after stripping is dropped
  - A note listing stripped counters is appended to the `description` field so users
    can see why a command looks different from the documentation
- `rocprof-compute` commands are never dropped (always represent new deep analysis)
- 7 new unit tests cover full/partial/zero PMC stripping, `full_command` update,
  description note, kernel-names-only drop, and rocprof-compute always-kept behavior

**OpenAI model compatibility:**
- `_call_openai()` in `llm_analyzer.py` now tries `max_completion_tokens` first (required
  by gpt-5, o1, o3, and newer gpt-4o variants) and falls back to `max_tokens` only when
  the API explicitly rejects `max_completion_tokens` (for legacy models).  Transparent to
  callers — no API change.

**Output format hint:**
- When `execute()` writes a file with the default text format (i.e., `--format` was not
  specified), a tip is printed to stdout suggesting `--format webview`, `--format json`,
  and `--format markdown` so users can discover the other output options.

**CTest integration:**
- `test_ai_analysis_standalone.py` (23 AI analysis API unit tests) is now registered with
  CTest as `tests.integration.execute.rocprofv3-test-rocpd-ai-analysis-unit-tests`
  via `configure_file()` + `rocprofiler_add_integration_execute_test()` in
  `tests/rocprofv3/rocpd/CMakeLists.txt`.

---

### v0.1.7 — 2026-02-27

**No schema changes.** `rocinsight.ai_analysis` Python API bug fixes and hardening (audit AIA-001 through AIA-013).

**Critical fix:**
- AIA-001: `analyze_database()` was broken — it called `analyze_performance()` with wrong
  parameters (`top_n`, `format_output=False`), but that function accepts neither and always
  returns a formatted `str`. Fixed by calling individual analysis functions directly
  (`compute_time_breakdown`, `identify_hotspots`, `analyze_memory_copies`,
  `analyze_hardware_counters`, `generate_recommendations`).

**High severity fixes:**
- AIA-002: `_build_analysis_result()` used wrong key names from `generate_recommendations()`
  output (`title`→`issue`, `description`→`suggestion`, `impact`→`estimated_impact`,
  lowercase priority→uppercase priority comparison). Fixed key mapping.
- AIA-003: `OutputFormat` enum was missing `WEBVIEW`. Added `WEBVIEW = "webview"`.
- AIA-004: `to_json()` returned non-conformant dataclass dict missing `schema_version`.
  Fixed by delegating to `format_analysis_output(..., output_format="json")`.
  Added `to_webview()` method (was documented but missing). Both methods use raw
  analysis payloads stored on `result._raw` for schema-conformant output.
- AIA-012: Created `ai_analysis/tests/test_api_standalone.py` (23 tests) and source copy
  `tests/rocprofv3/rocpd/test_ai_analysis_standalone.py`. Updated docs.

**Medium severity fixes:**
- AIA-005: LLM auth/rate-limit errors (`LLMAuthenticationError`, `LLMRateLimitError`) now
  propagate when `enable_llm=True` instead of being silently swallowed as warnings.
- AIA-006: `_convert_result_to_llm_format()` replaced hardcoded `"AMD GPU"`, `"gfx90a"`,
  empty `kernels: []`, `memory_ops: {}` with real data from `result._raw`.
- AIA-007: Implemented file path redaction in `_sanitize_data()` using a regex pattern
  matching Unix (`/home/`, `/opt/`, `/root/`, `/tmp/`, `/var/`) and Windows paths.
- AIA-008: `ReferenceGuideNotFoundError` now accepts `List[str]` of all attempted paths
  and shows all in the error message (was only showing one path). Updated
  `get_reference_guide_path()` to collect all attempted paths before raising.
- AIA-009: Added `DEFAULT_ANTHROPIC_MODEL` / `DEFAULT_OPENAI_MODEL` constants. Model names
  are now configurable via `ROCINSIGHT_LLM_MODEL` environment variable at runtime or the new
  `--llm-model` CLI flag (`rocinsight analyze --llm anthropic --llm-model claude-opus-4-6`).
- AIA-013: `validate_database()` now queries `type IN ('table','view')` instead of
  `type='table'` so `kernels`, `memory_copies`, and `pmc_events` views are detected.

**Low severity fixes:**
- AIA-010: Fixed type hints in `exceptions.py` (`missing_tables: Optional[List[str]]`,
  `gpu_arch: Optional[str]`). Added `from typing import Optional` import.
- AIA-011: `ReferenceGuideNotFoundError` is now exported from `rocinsight.ai_analysis.__init__`.

---

### v0.1.6 — 2026-02-24

**No schema changes.** Recommendation engine improvement only.

- Recommendations no longer suggest re-running profiling flags that were
  already used in the original collection run.  The engine now inspects the
  database to infer what was already collected:
  - Presence of `kernels` rows → `--kernel-trace` covered
  - Presence of `regions` rows (HIP/HSA API spans) → API tracing covered
  - Presence of `memory_copies` rows → `--memory-copy-trace` covered
  - `kernels` + `regions` together → full `--sys-trace` implied, which
    subsumes `--hip-trace`, `--hip-api-trace`, `--hsa-trace`,
    `--kernel-trace`, `--memory-copy-trace`, `--marker-trace`
- Redundant trace flags are stripped from recommended `rocprofv3` commands;
  if a command has no remaining flags and no meaningful new args (beyond
  `-d`/`-o`), it is dropped entirely so only actionable next steps appear.
- Commands for `rocprof-sys` and `rocprof-compute` are never dropped — they
  always represent a new perspective even on already-collected data.
- New internal helpers: `_detect_already_collected()`,
  `_filter_rec_commands()`, `_SYS_TRACE_IMPLIED`.

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
  - **Light/Dark theme toggle** — persisted in `localStorage` (`rocinsight-theme` key);
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

### v0.4.0 (planned)
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
  rocinsight/ai_analysis/docs/analysis-output.schema.json
```

```python
# Programmatic validation
import json
import jsonschema
import importlib.resources as pkg_resources

schema_text = (
    pkg_resources.files("rocinsight.ai_analysis")
    .joinpath("docs/analysis-output.schema.json")
    .read_text()
)
schema = json.loads(schema_text)

with open("analysis.json") as f:
    instance = json.load(f)

jsonschema.validate(instance=instance, schema=schema)
print("Valid!")
```
