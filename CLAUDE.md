# ROCProfiler SDK Project

## Tech Stack
- Language: C++17
- Build System: CMake 3.21+
- Python: Python 3
- Target: ROCm profiling tools & runtimes

## Key Directories
- `projects/rocprofiler-sdk/` - Main source
- `projects/rocprofiler-sdk/source/lib/python/rocpd/` - Python CLI source
- `projects/rocprofiler-sdk/build/lib/python3.12/site-packages/rocpd/` - Built Python modules
- `/opt/rocm` - Install prefix (requires sudo)
- `/opt/rocm-7.0.0/lib/python3.12/site-packages/rocpd/` - System-installed Python modules

## Quick Commands
- Build: Use `/build-rocprofiler-sdk` command for full ROCProfiler-SDK build workflow
- Test: `ctest --test-dir projects/rocprofiler-sdk/build`
- Analyze GPU trace:
  ```bash
  ROCPD_SYS=/opt/rocm-7.0.0/lib/python3.12/site-packages
  PYTHONPATH="${ROCPD_SYS}" python3 -m rocpd analyze -i <trace.db>
  ```

## Important Notes
- Installation requires sudo access to /opt/rocm
- Use parallel builds (--parallel 32) for faster compilation
- **After editing `analyze.py` in source, copy to system package for immediate testing**:
  ```bash
  cp projects/rocprofiler-sdk/source/lib/python/rocpd/analyze.py \
     /opt/rocm-7.0.0/lib/python3.12/site-packages/rocpd/analyze.py
  ```
- **PYTHONPATH order is critical**: system path MUST come before source path to avoid
  circular import (`ImportError: cannot import name 'libpyrocpd' from partially initialized
  module 'rocpd'`). Always use `PYTHONPATH="${ROCPD_SYS}:${ROCPD_SRC}"`, not the reverse.

## Active Branch
- `aelwazir/rocpd-ai-analysis` — AI analysis module work in this repo

## Real Test Database (IMPORTANT)
A real multi-dispatch trace database with PMC counters is available at:
```
projects/rocprofiler-sdk/build/tests/rocprofv3/rocpd/rocpd-input-data/merged_db.db
```
- **2000 kernel dispatches** (`reproducible_dispatch_count` kernel)
- **64000 hardware counter samples** (3 counters: GRBM_COUNT, GRBM_GUI_ACTIVE, SQ_WAVES)
- Use this for testing both Tier 1 and Tier 2 analysis

**Do NOT use** `sample/trace_70b_1024_32.rpd` — has "view rocpd_kernel_dispatch is circularly
defined" schema error (legacy format conflict).

## AI Analysis Module

### Overview
AI-powered GPU trace analysis tool integrated into rocpd CLI as `rocpd analyze` subcommand.

### Key Files
- `source/lib/python/rocpd/analyze.py` - Main analysis module (~2300+ lines)
- `source/lib/python/rocpd/__main__.py` - CLI integration (modified to add analyze subcommand)
- `source/lib/python/rocpd/ai_analysis/` - AI analysis sub-module
  - `README.md` - Module overview
  - `docs/AI_ANALYSIS_API.md` - Python API reference
  - `docs/SCHEMA_CHANGELOG.md` - JSON schema version history (current: v0.1.8)
  - `docs/analysis-output.schema.json` - JSON output schema
  - `share/llm-reference-guide.md` - User-modifiable LLM "fence"
  - `interactive.py` - Interactive session module (`InteractiveSession`, `WorkflowSession`, `SessionContext`, `SessionData`, `SessionStore`)
- `tests/rocprofv3/rocpd/test_analyze.py` - Unit tests (76 tests: all rules, helpers, formatters, PMC filter)
- `tests/rocprofv3/rocpd/test_analyze_schema.py` - JSON schema conformance tests (17 tests)
- `tests/rocprofv3/rocpd/CMakeLists.txt` - Integration tests
- `source/lib/python/rocpd/ai_analysis/tests/test_interactive.py` - 22 interactive session unit tests
- `source/lib/python/rocpd/ai_analysis/tests/test_interactive_context.py` - 27 session context + persistence tests

### Usage
```bash
# Text output (produces <name>.txt)
rocpd analyze -i trace.db -d ./output -o analysis

# JSON output (produces analysis.json)
rocpd analyze -i trace.db --format json -d ./output -o analysis

# Markdown output (produces analysis.md)
rocpd analyze -i trace.db --format markdown -d ./output -o analysis

# Interactive HTML webview (produces analysis.html)
rocpd analyze -i trace.db --format webview -d ./output -o analysis

# More options
rocpd analyze -i trace.db --top-kernels 20 --format text

# Override LLM model (also configurable via ROCPD_LLM_MODEL env var)
rocpd analyze -i trace.db --llm anthropic --llm-model claude-opus-4-6
```

### Output Format → File Extension Mapping (IMPORTANT)
The `execute()` function in `analyze.py` automatically appends the correct extension
to the base output filename (`-o` arg). Do NOT manually specify extensions in `-o`.

| `--format` | Extension | Description |
|---|---|---|
| `text` (default) | `.txt` | Plain text report |
| `json` | `.json` | Machine-readable structured data |
| `markdown` | `.md` | Markdown with syntax highlighting |
| `webview` | `.html` | Self-contained interactive HTML |

### Key Bug Fixed: format→output_format mismatch
`add_args()` returns `{'format': 'json'}` in kwargs, but `analyze_performance()` expects
the parameter named `output_format`. Fixed in `execute()`:
```python
if "format" in kwargs:
    kwargs["output_format"] = kwargs.pop("format")
```
Without this, `--format json` was silently ignored and text was always produced.

### SQL Schema Notes (IMPORTANT)
Real rocprofv3 database schema differs from initial assumptions:

**kernels table**:
- Columns: id, guid, tid, name, start, end, duration
- `duration` is INTEGER type
- **Issue**: SQLite parameter binding causes "datatype mismatch" errors with WHERE/LIMIT clauses
- **Solution**: Use f-string formatting instead of parameter binding:
  ```python
  query = f"SELECT ... FROM kernels WHERE duration >= {int(value)} LIMIT {int(limit)}"
  results = execute_statement(connection, query, ()).fetchall()
  ```

**memory_copies table**:
- Uses `category` column (not `src_agent_name`/`dst_agent_name`)
- Category values: 'HostToDevice', 'DeviceToHost', 'DeviceToDevice'
- Has `size` column (INTEGER) for bytes transferred
- Query pattern: `WHERE category LIKE '%HostToDevice%' OR category LIKE '%H2D%'`

**rocpd importer schema** (for synthetic test databases):
- Requires `rocpd_metadata` table with UUID-based naming
- Tables named `rocpd_kernel_dispatch_<uuid>`, `rocpd_memory_copy_<uuid>`, etc.
- Importer creates unified views: `kernels`, `memory_copies`, `pmc_events`
- Plain `CREATE TABLE kernels (...)` won't work — must follow UUID schema or use a real trace DB

### CMake GPU-less Build Fix (IMPORTANT)
On machines without AMD GPUs, `rocminfo` returns an empty list, causing CMake configure
to crash with "list GET given empty list". Two files were fixed:

1. `cmake/Modules/rocprofiler-sdk-utilities.cmake` — both `rocprofiler_sdk_pc_sampling_disabled`
   and `rocprofiler_sdk_pc_sampling_stochastic_disabled` functions now guard `list(GET ...)`:
   ```cmake
   list(LENGTH rocprofiler-sdk-tests-gfx-info _gfx_list_len)
   if(_gfx_list_len EQUAL 0)
       set(${_VAR} TRUE PARENT_SCOPE)
       return()
   endif()
   list(GET rocprofiler-sdk-tests-gfx-info 0 pc-sampling-gpu-0-gfx-info)
   ```

2. `tests/CMakeLists.txt` — adds a `gfx000` placeholder when no GPUs detected:
   ```cmake
   list(LENGTH rocprofiler-sdk-tests-gfx-info _gfx_list_len)
   if(_gfx_list_len EQUAL 0)
       set(rocprofiler-sdk-tests-gfx-info "gfx000")
   endif()
   ```

### Analysis Features

**Tier 1 (Trace-Level Analysis)**:
1. **Time Breakdown**: Kernel execution %, memory copy %, API overhead %
2. **Hotspot Identification**: Top N kernels by total duration
3. **Memory Analysis**: Transfers by direction (H2D, D2H, D2D) with bandwidth

**Tier 2 (Hardware Counter Analysis)**:
1. **GPU Utilization**: Calculated from GRBM_GUI_ACTIVE / GRBM_COUNT
2. **Wave Occupancy**: Average and max waves from SQ_WAVES counter
3. **Per-Kernel Counters**: Aggregated counter stats by kernel name
4. **Derived Metrics**: Occupancy, utilization, and other derived values
5. **Auto-Detection**: Automatically activates when `pmc_events` table exists

**Recommendations** (8 rule-based suggestions):
   - **Tier 2**: Low occupancy (avg waves <16), Low GPU utilization (<70%)
   - **Tier 1**: High memory transfer (>20%), High API overhead (>15%)
   - **Tier 1**: Compute bottleneck (single kernel >50%), Many small kernels (avg <10μs)
   - **Tier 1**: Low memory bandwidth (<10 GB/s), Default (application well-optimized)

### Webview Format (_format_as_webview)
Added in `analyze.py`. Generates a self-contained single-file HTML report:
- AMD dark theme (bg `#0e0e14`, AMD red `#e01a22`)
- No external CDN dependencies — works offline
- Sections: Overview, Execution Breakdown (stacked bars), Recommendations (collapsible
  cards, HIGH priority auto-expanded, copy-to-clipboard commands), Hotspot Table
  (sortable, rows >20% highlighted), Memory Transfer Table, Hardware Counter Gauges
  (SVG donut, Tier 2 only)
- Embeds full JSON payload as `var ANALYSIS = {...}` for programmatic inspection
- **Hover tooltips** on every visual element: gauges, bars, stat cards, table headers,
  counter rows. Tooltip content includes counter formulas, target thresholds, and
  optimization guidance. Counter rows use a JS `COUNTER_TIPS` lookup (20+ AMD counters:
  GRBM_*, SQ_*, TCP/TCC, FETCH_SIZE, WRITE_SIZE, etc.). Unknown counters get a fallback.
  Tooltip system: `#tt` floating div, `[data-tip]` attribute (single-quoted to allow
  inner HTML with double quotes), `data-ctr` for counter rows (JS lookup instead of
  attribute to keep HTML clean).
- **CSS f-string gotcha**: all `{` and `}` in CSS must be doubled (`{{`, `}}`).
  JavaScript template literals (`${}`) must be avoided — use concatenation instead.
- **CSS `content` property gotcha**: the CSS `content` property does NOT process HTML
  entities. `content:'&#8594;'` renders the literal 7-char string `&#8594;`, not `→`.
  Always use the actual Unicode character (e.g. `content:'→'`) or a CSS unicode escape
  (e.g. `content:'\2192'`). Never use `&#xxxx;` HTML entities in CSS `content`.
- **Tooltip color gotcha**: the `#tt` floating tooltip has a hardcoded dark background
  (`#0e0e1c`). Never use `color:var(--text)` inside `#tt` — in light theme `--text`
  becomes near-black, making text invisible. Always pin an explicit light color such as
  `color:#dde0f2` on `#tt` so it is readable in both dark and light themes.
- **Tooltip tip content gotcha**: tip strings use single-quote delimited `data-tip`
  attributes. Any single quotes inside tip text must be `&#39;`. Double quotes are fine.

### Testing Notes
- **Tier 1 testing**: Use any `.db` file from `rocprofv3 --sys-trace`
- **Tier 2 testing**: Use `merged_db.db` (see Real Test Database section above)
- Integration tests: `ctest -R rocpd-analyze`
- Unit tests: `pytest --noconftest test_analyze_standalone.py`
- Schema tests: `pytest --noconftest test_analyze_schema_standalone.py`
- **AI Analysis API tests** (new): `pytest --noconftest test_ai_analysis_standalone.py`
  (Run from `build/tests/rocprofv3/rocpd/` with `PYTHONPATH=/opt/rocm-7.0.0/...`)
- **Interactive session tests**: Run from `/tmp` to avoid circular `libpyrocpd` import:
  ```bash
  ROCPD_SYS=/opt/rocm-7.2.0/lib/python3.12/site-packages
  TEST=source/lib/python/rocpd/ai_analysis/tests
  cd /tmp && PYTHONPATH="${ROCPD_SYS}" python3 -m pytest \
      /path/to/${TEST}/test_interactive.py \
      /path/to/${TEST}/test_interactive_context.py \
      --noconftest -v
  ```

### Python API (ai_analysis) Important Notes
- **`RocpdImportData` path gotcha**: Always pass the DB path as a LIST, not a bare
  string. `sanitize_input_list()` iterates over its input — a bare string produces
  character-by-character iteration. Use `RocpdImportData([str(db_path)])`.
- **`analyze_database()` via api.py**: Now correctly calls individual analysis
  functions (`compute_time_breakdown`, `identify_hotspots`, etc.) rather than
  `analyze_performance()` (which returns `str`, not `dict`).
- **`to_json()`**: Returns schema-conformant JSON (schema_version="0.1.0") by
  delegating to `format_analysis_output()`. Fallback to dataclass dict if `_raw`
  is not attached (e.g., result constructed manually).
- **`to_webview()`**: Returns full AMD-themed HTML; requires `result._raw` to be
  populated (always the case when using `analyze_database()`).
- **LLM auth errors propagate**: `LLMAuthenticationError` and `LLMRateLimitError`
  are re-raised when `enable_llm=True`. Other LLM errors → warning + local results.
- **PMC counter deduplication**: `_detect_already_collected()` inspects `pmc_events`
  for already-collected counter names (`pmc:GRBM_COUNT` etc.). `_filter_rec_commands()`
  strips those counters from `--pmc` args in recommendations; commands where all
  suggested counters are already present are dropped entirely. `--kernel-names` is
  treated as a scope filter (not data collection) for the purpose of this check.
- **OpenAI model compatibility**: `_call_openai()` uses `max_completion_tokens` first
  (required by gpt-5, o1, o3, newer gpt-4o), falls back to `max_tokens` for older
  models. Transparent — no API change required.

### Interactive Session (`interactive.py`)

**Two session classes:**
- `InteractiveSession` — menu-driven loop `[p]/[a]/[o]/[s]/[q]`; triggered after standard analysis when `--interactive` flag is set without a RUN_COMMAND
- `WorkflowSession` — 7-phase automated workflow; triggered by `--interactive "<app_command>"`

**WorkflowSession re-profiling loop behaviour (key fixes):**
- Phase 4 banner shows `(Run #N)` from iteration 2 onwards so the user can distinguish re-profile results from the initial run
- `_print_comparison` compares the NEW breakdown against history entry `[-1]` (the most-recent prior run). History is appended AFTER the comparison call, so the check is `len < 1`, not `< 2`.
- When `total_runtime_ns == 0` after a profiling run, a ⚠ warning is printed explaining that the app may use Python multiprocessing spawn (e.g. vLLM, PyTorch DDP) — GPU kernels run in worker processes and are not captured in the main-process DB. Suggests `rocprof-sys` or `--pid` profiling.
- After computing `ai_rec_cmd`, the PMC counters in the suggestion are compared against the last `trace_history` command. If all suggested counters are already present, `ai_rec_cmd` is cleared to prevent an infinite `[r] → re-profile → same INFO → [r]` loop.
- `_phase5_rec_menu` detects `already_reprofiled` (all INFO + iteration > 0 + no fresh `ai_rec_cmd`) and replaces the `[r]` option with a "result unchanged" note.
- Invalid rocprofv3 CLI flags are stripped from AI-recommended commands before use — the LLM fence documents valid flags but LLMs still hallucinate legacy names. Two categories stripped: (a) standalone boolean flags: `--hip-api-trace`, `--hsa-trace`, `--hip-trace`, `--hip-stats`, `--hsa-stats`; (b) value-taking flag: `--kernel-names <value>`. Pattern: `r"\s*--(hip-api-trace|hsa-trace|hip-trace|hip-stats|hsa-stats)\b"` and `r"--kernel-names\s+(?:'[^']*'|\"[^\"]*\"|\S+)"`.
- `_phase6_apply_direct` retries `_llm_rewrite_file` on failure (timeout, rate-limit, etc.) instead of silently falling through to Phase 7. Prompts `Retry LLM rewrite? [y/N]`.

**Key dataclasses** (all in `interactive.py`):
- `SessionContext` — compact per-session facts: `iteration`, `analyses` (capped 5), `suggestions_given` (capped 3, truncated 120 chars), `commands_run` (capped 5)
- `SessionData` — persistent session JSON: `session_id`, `source_dir`, `history`, `persistent_menu_items`, `context` (serialized `SessionContext`)
- `SessionStore` — saves/loads `SessionData` JSON to `~/.rocpd/sessions/` by default

**LLM context injection:**
- `_format_context_block()` → returns `""` on first call; otherwise a `### Session Context` block with prior analyses, suggestions, commands (≤~325 tokens)
- Injected at two sites: `_optimize_via_tier0()` and `_request_optimization_suggestions()` by prepending `ctx_block + "\n\n"` to the user prompt
- `_update_ctx_suggestion(llm_response)` called after each LLM response to record it

**`_run_tier1_analysis(db_path)`** returns `(recs, breakdown)` tuple. `breakdown` is a dict with `kernel_time_pct`, `memcpy_time_pct`, `api_overhead_pct`, `idle_time_pct`, `total_runtime_ns`; `None` on failure.

**AI command extraction:**
- `_extract_ai_commands(text, structured_cmds)` — `re.findall(r"rocprofv3\s+[^\n]+", text)` + structured list, deduped, max 5
- `_offer_run_ai_commands(commands)` — shows numbered menu; if user picks one, runs via `subprocess.run`, auto-detects output `.db`, re-analyzes, updates `_ctx`, auto-saves session

**Persistence pattern:**
```python
# Save (both _save_and_quit and [s] branch):
self._session.context = asdict(self._ctx)
self._store.save(self._session)

# Restore on resume (in _init_session):
raw_ctx = loaded.context or {}
self._ctx = SessionContext(**raw_ctx) if raw_ctx else SessionContext()
```

**Copy-to-system required** after editing `interactive.py` for tests to see changes:
```bash
cp source/lib/python/rocpd/ai_analysis/interactive.py \
   /opt/rocm-7.2.0/lib/python3.12/site-packages/rocpd/ai_analysis/interactive.py
```

### Hardware Counter Detection
The module automatically detects hardware counter data:
- Queries `pmc_events` table for counter data
- If counters exist: Tier 2 analysis activates automatically
- If no counters: Falls back to Tier 1 trace-level analysis only
- No user action required - detection is transparent

### Verification Status
✅ **Production Ready - Tier 1, Tier 2, and Webview Functional**
- Tier 1: Tested with real GPU trace (reproducible_dispatch_count, 1000 dispatches)
- Tier 2: Tested with hardware counters (48,000 counter samples, 3 counters)
- Webview: Generates valid HTML; verified with `merged_db.db` → 28 KB self-contained HTML