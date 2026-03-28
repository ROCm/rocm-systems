# rocinsight

**AI-powered GPU trace analysis for AMD ROCm — no C++ dependency required.**

rocinsight is a standalone Python package that reads rocprofiler-sdk SQLite trace databases
(`.db` files) and produces human-readable performance insights, bottleneck detection,
and actionable optimization recommendations.  It was extracted from the rocprofiler-sdk
`rocpd` tool so it can be installed and run independently — without compiling any C++
extensions (`libpyrocpd`).

---

**New to rocinsight?** Check out the [Getting Started Guide](docs/guides/getting-started.md) with animated demos for every feature.

## Table of Contents

1. [Why rocinsight?](#why-rocinsight)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [CLI Reference](#cli-reference)
5. [Python API](#python-api)
6. [Analysis Tiers](#analysis-tiers)
7. [Output Formats](#output-formats)
8. [LLM Enhancement](#llm-enhancement)
9. [Interactive Session](#interactive-session)
10. [Environment Variables](#environment-variables)
11. [Project Structure](#project-structure)
12. [Architecture: RocinsightConnection](#architecture-rocinsightconnection)
13. [JSON Schema](#json-schema)
14. [Building with CMake](#building-with-cmake)
15. [Testing](#testing)
16. [Differences from rocpd](#differences-from-rocpd)
17. [Contributing](#contributing)

---

## Why rocinsight?

The `rocpd` tool that ships with rocprofiler-sdk requires `libpyrocpd` — a compiled
C++ extension — to open `.db` trace files.  This means:

- The SDK must be **built from source** (or an exact binary match installed) before any
  analysis can run.
- The package cannot be installed on machines that only have the `.db` output files.
- CI/CD pipelines or post-processing jobs cannot use the Python API without a full
  SDK build.

**rocinsight solves this** by replacing `libpyrocpd` with a pure-Python `sqlite3` adapter
(`RocinsightConnection`) that reads the same `.db` files with zero compiled dependencies.
All analysis logic, LLM integration, interactive sessions, and output formatters are
identical to the rocprofiler-sdk version.

---

## Quick Start

```bash
# Install directly from git — no clone needed, no C++ compiler required
pip install 'rocinsight @ git+https://github.com/ROCm/rocm-systems.git#subdirectory=experimental/python/rocinsight'

# With LLM extras
pip install 'rocinsight[llm] @ git+https://github.com/ROCm/rocm-systems.git#subdirectory=experimental/python/rocinsight'

# Or from a local clone
pip install experimental/python/rocinsight

# Analyze a trace database
rocinsight analyze -i trace.db

# Generate an interactive HTML report
rocinsight analyze -i trace.db --format webview -d ./reports -o my_trace

# Analyze multiple databases (multi-process run)
rocinsight analyze -i run_0.db run_1.db run_2.db

# Tier 0: scan source code before you have a trace
rocinsight analyze --source-dir ./my_app

# Full interactive workflow (profile → analyze → optimize loop)
rocinsight analyze -i trace.db --interactive --llm anthropic
```

---

## Installation

### Option 1: pip from git (recommended — no clone needed)

```bash
# Core (stdlib only — no C++ required)
pip install 'rocinsight @ git+https://github.com/ROCm/rocm-systems.git#subdirectory=experimental/python/rocinsight'

# With LLM support (Anthropic + OpenAI)
pip install 'rocinsight[llm] @ git+https://github.com/ROCm/rocm-systems.git#subdirectory=experimental/python/rocinsight'

# All optional extras
pip install 'rocinsight[all] @ git+https://github.com/ROCm/rocm-systems.git#subdirectory=experimental/python/rocinsight'

# Pin to a specific commit
pip install 'rocinsight @ git+https://github.com/ROCm/rocm-systems.git@<commit>#subdirectory=experimental/python/rocinsight'
```

### Option 2: pip from local clone (development install)

```bash
cd rocm-systems-dev/experimental/python/rocinsight
pip install -e .                      # Core analysis (stdlib only)
pip install -e ".[llm]"               # + Anthropic + OpenAI support
pip install -e ".[all]"               # + rich terminal output
pip install -e ".[dev]"               # + pytest / flake8 / black
```

### Option 3: pip (non-editable, from super-repo root)

```bash
pip install ./experimental/python/rocinsight
```

### Option 4: CMake (as part of the super-repo build)

```bash
cmake -B build \
      -DCMAKE_INSTALL_PREFIX=/opt/rocm \
      experimental/python/rocinsight

cmake --build build
sudo cmake --install build
```

This installs the `rocinsight` script to `/opt/rocm/bin/` and the Python package to
`/opt/rocm/lib/python3.x/site-packages/rocinsight/`.

### Dependencies

| Dependency | Required for | How to install |
|---|---|---|
| Python ≥ 3.8 | Everything | — |
| `anthropic >= 0.18` | `--llm anthropic` | `pip install anthropic` |
| `openai >= 1.0` | `--llm openai` / `--llm private` | `pip install openai` |
| `rich >= 13.0` | Colored terminal output | `pip install rich` |
| `httpx` | `ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL=0` (SSL skip) | `pip install httpx` |
| `pytest` | Running tests | `pip install pytest` |

Core analysis (Tier 0–3) requires **no third-party packages** — only Python stdlib.

---

## CLI Reference

All commands use the `rocinsight analyze` subcommand.

### Synopsis

```
rocinsight analyze [-i DB [DB ...]] [--source-dir DIR] [OPTIONS]
```

At least one of `-i` (database) or `--source-dir` (source directory) is required.

### Input options

| Flag | Description |
|---|---|
| `-i DB [DB ...]` | One or more rocprofiler-sdk `.db` trace files.  When multiple files are given they are opened via SQLite ATTACH and unioned transparently. |
| `--source-dir DIR` | Path to GPU source code directory for Tier 0 static analysis.  Can be combined with `-i` for a combined Tier 0 + Tier 1/2 report. |
| `--att-dir DIR` | Directory containing `stats_*.csv` files from a `rocprofv3 --att` run.  Enables Tier 3 ATT stall analysis. |

### Output options

| Flag | Default | Description |
|---|---|---|
| `--format FORMAT` | `text` | Output format: `text`, `json`, `markdown`, `webview` |
| `-o NAME` | — | Base name for output file (extension added automatically) |
| `-d DIR` | `.` | Output directory |

When `-o` is not given, output is printed to stdout.

### Analysis options

| Flag | Default | Description |
|---|---|---|
| `--top-kernels N` | `10` | Number of hotspot kernels to show |
| `--prompt TEXT` | — | Natural language question to guide analysis |
| `--verbose` | off | Show debug information |

### LLM options

| Flag | Description |
|---|---|
| `--llm PROVIDER` | Enable LLM enhancement.  Providers: `anthropic`, `openai`, `private` |
| `--llm-model MODEL` | Override default model (e.g. `claude-opus-4-6`, `gpt-4o`) |
| `--llm-private-model MODEL` | Model name for `--llm private` |

### Interactive session options

| Flag | Description |
|---|---|
| `--interactive` | Launch menu-driven analysis loop (`InteractiveSession`) |
| `--interactive "CMD"` | Launch 7-phase automated workflow (`WorkflowSession`) with the given app command |
| `--resume-session ID` | Resume a previously saved `InteractiveSession` by session ID |
| `--llm-compact-every N` | Compact LLM conversation history every N turns (default: 10) |

### Usage examples

```bash
# ── Tier 1: basic trace analysis ────────────────────────────────
rocinsight analyze -i output.db

# ── Tier 1 + Tier 2: with hardware counters ──────────────────────
# (trace must have been collected with --pmc)
rocinsight analyze -i output.db

# ── Multiple databases (multi-process / multi-rank run) ──────────
rocinsight analyze -i results_0.db results_1.db results_2.db

# ── Tier 0: source code scan (no .db required) ───────────────────
rocinsight analyze --source-dir ./my_hip_app

# ── Combined Tier 0 + Tier 1 ─────────────────────────────────────
rocinsight analyze -i output.db --source-dir ./my_hip_app

# ── Tier 3: ATT instruction-level stall analysis ─────────────────
# First collect (run on a machine with AMD GPU):
#   rocprofv3 --att --att-library-path /opt/rocm/lib \
#             --att-target-cu 0 -d ./att_out -o trace -- ./app
rocinsight analyze -i ./att_out/trace*.db --att-dir ./att_out

# ── Output formats ────────────────────────────────────────────────
rocinsight analyze -i output.db --format json     -d ./out -o analysis  # → analysis.json
rocinsight analyze -i output.db --format markdown -d ./out -o analysis  # → analysis.md
rocinsight analyze -i output.db --format webview  -d ./out -o analysis  # → analysis.html

# ── LLM enhancement ───────────────────────────────────────────────
export ANTHROPIC_API_KEY="sk-ant-..."
rocinsight analyze -i output.db --llm anthropic

export OPENAI_API_KEY="sk-..."
rocinsight analyze -i output.db --llm openai

# Private/enterprise OpenAI-compatible server
export ROCINSIGHT_LLM_PRIVATE_URL="https://llm.corp.example.com/openai"
export ROCINSIGHT_LLM_PRIVATE_HEADERS='{"Ocp-Apim-Subscription-Key": "abc123"}'
rocinsight analyze -i output.db --llm private --llm-private-model gpt-4o

# ── Custom analysis prompt ─────────────────────────────────────────
rocinsight analyze -i output.db --prompt "Why is my matmul kernel slow?"
rocinsight analyze -i output.db --prompt "Focus on memory bandwidth bottlenecks"

# ── Interactive session ───────────────────────────────────────────
rocinsight analyze -i output.db --interactive
rocinsight analyze -i output.db --interactive --llm anthropic

# 7-phase workflow (profile → analyze → optimize loop)
rocinsight analyze --interactive "./my_app arg1 arg2" --llm anthropic

# Resume a saved session
rocinsight analyze -i output.db --interactive \
    --resume-session 2026-03-10_14-23-01_myapp
```

---

## Python API

Import from `rocinsight.ai_analysis` (not `rocpd.ai_analysis`):

```python
from rocinsight.ai_analysis import (
    analyze_database,
    analyze_database_to_json,
    analyze_source,
    get_recommendations,
    validate_database,
    AnalysisResult,
    SourceAnalysisResult,
    OutputFormat,
    LLMAnalyzer,
    LLMAuthenticationError,
    LLMRateLimitError,
)
from pathlib import Path
```

### `analyze_database()`

```python
def analyze_database(
    database_path: Path,
    *,
    custom_prompt: Optional[str] = None,
    enable_llm: bool = False,
    llm_provider: Optional[str] = None,   # "anthropic" | "openai" | "private"
    llm_api_key: Optional[str] = None,
    output_format: OutputFormat = OutputFormat.PYTHON_OBJECT,
    verbose: bool = False,
    top_kernels: int = 10,
    att_dir: Optional[Path] = None,        # Tier 3 ATT directory
) -> AnalysisResult
```

**Examples:**

```python
# Basic analysis — local mode, no internet
result = analyze_database(Path("output.db"))
print(result.summary.overall_assessment)
print(f"Primary bottleneck: {result.summary.primary_bottleneck}")

# High-priority recommendations
for rec in result.recommendations.high_priority:
    print(f"[{rec.priority}] {rec.title}")
    print(f"  {rec.description}")
    print(f"  Impact: {rec.estimated_impact}")

# Save webview report
Path("report.html").write_text(result.to_webview())

# With LLM
result = analyze_database(
    Path("output.db"),
    enable_llm=True,
    llm_provider="anthropic",
    custom_prompt="Why is kernel X slow?",
)
print(result.llm_enhanced_explanation)

# Tier 3: ATT analysis
result = analyze_database(
    Path("att_out/trace0.db"),
    att_dir=Path("att_out"),
)
```

### `analyze_source()`

Tier 0: static analysis of GPU source code — no `.db` required.

```python
def analyze_source(
    source_dir: Path,
    *,
    custom_prompt: Optional[str] = None,
    enable_llm: bool = False,
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    verbose: bool = False,
) -> SourceAnalysisResult
```

```python
from rocinsight.ai_analysis import analyze_source
from pathlib import Path

plan = analyze_source(Path("./my_hip_app"))
print(plan.suggested_first_command)
print(f"Detected {plan.kernel_count} GPU kernels")
for pat in plan.detected_patterns:
    print(f"[{pat['severity'].upper()}] {pat['description']}")
```

### `analyze_database_to_json()`

```python
json_str = analyze_database_to_json(
    database_path=Path("output.db"),
    output_json_path=Path("analysis.json"),   # optional — also saves to file
)
```

### `get_recommendations()`

```python
from rocinsight.ai_analysis import get_recommendations

recs = get_recommendations(
    Path("output.db"),
    priority_filter="high",       # "high" | "medium" | "low" | None (all)
    category_filter="memory",     # optional substring match on category
)
for rec in recs:
    print(rec.title, rec.estimated_impact)
```

### `validate_database()`

```python
from rocinsight.ai_analysis import validate_database

info = validate_database(Path("output.db"))
# info keys: is_valid, tier, has_kernels, has_memory_copies,
#            has_pmc_counters, tables, error
print(f"Valid: {info['is_valid']}, Tier: {info['tier']}")
```

### `AnalysisResult` dataclass

```python
@dataclass
class AnalysisResult:
    metadata: AnalysisMetadata            # version, db path, timestamp, custom_prompt
    profiling_info: ProfilingInfo         # duration_ns, mode, tier, gpus
    summary: AnalysisSummary             # bottleneck, confidence, key_findings
    execution_breakdown: ExecutionBreakdown  # kernel%, memcpy%, api%, idle%
    recommendations: RecommendationSet   # .high_priority / .medium_priority / .low_priority
    warnings: List[AnalysisWarning]
    errors: List[str]
    llm_enhanced_explanation: Optional[str]
    tier0: Optional[SourceAnalysisResult]  # when --source-dir also used
    _raw: dict                            # internal; used by to_json() / to_webview()

    # Serialization methods
    def to_dict() -> dict
    def to_json(indent: int = 2) -> str   # schema-conformant JSON (v0.1.0 / v0.3.0)
    def to_text() -> str
    def to_markdown() -> str
    def to_webview() -> str               # self-contained HTML report
```

### `SourceAnalysisResult` dataclass

```python
@dataclass
class SourceAnalysisResult:
    source_dir: str
    analysis_timestamp: str              # ISO 8601
    programming_model: str               # "HIP", "HIP+ROCm_Libraries", "PyTorch_HIP", …
    files_scanned: int
    files_skipped: int
    detected_kernels: List[Dict]         # {name, file, line, launch_type}
    kernel_count: int
    detected_patterns: List[Dict]        # {pattern_id, severity, category, description, count, locations}
    risk_areas: List[str]
    already_instrumented: bool           # ROCTx markers found
    roctx_marker_count: int
    recommendations: List[Dict]          # same shape as generate_recommendations() output
    suggested_counters: List[str]
    suggested_first_command: str         # ready-to-run rocprofv3 command
    llm_explanation: Optional[str]
```

### Exception hierarchy

```
AnalysisError
├── DatabaseNotFoundError
├── DatabaseCorruptedError
├── MissingDataError
├── UnsupportedGPUError
├── LLMAuthenticationError       ← propagates; not swallowed
├── LLMRateLimitError            ← propagates; not swallowed
├── ReferenceGuideNotFoundError
├── SourceDirectoryNotFoundError
└── SourceAnalysisError
```

### `LLMConversation` — persistent multi-turn session

```python
from rocinsight.ai_analysis.llm_conversation import LLMConversation

conv = LLMConversation(provider="anthropic", api_key="sk-ant-...")
conv.initialize("You are a GPU performance expert.")

response = conv.send("What is wave occupancy?")
print(response)

response2 = conv.send("How does it relate to LDS usage?")
# LLM has full context from previous message

# Serialize / restore across sessions
state = conv.to_dict()          # excludes api_key
restored = LLMConversation.from_dict(state, api_key="sk-ant-...")
```

---

## Analysis Tiers

rocinsight implements a progressive five-tier analysis model:

| Tier | Data Required | Collection Command | Key Insights |
|---|---|---|---|
| **0** | Source code (`--source-dir`) | *(none — static analysis)* | Detected kernels, profiling plan, suggested first command |
| **1** | Trace DB (`-i db.db`) | `rocprofv3 --sys-trace -- ./app` | Kernel hotspots, time breakdown, memory copy overhead, idle time |
| **2** | Trace DB + PMC counters | `rocprofv3 --pmc GRBM_COUNT SQ_WAVES FETCH_SIZE -- ./app` | GPU utilization %, wave occupancy, HBM bandwidth (Roofline/SOL) |
| **3** | Trace DB + ATT output | `rocprofv3 --att --att-library-path /opt/rocm/lib --att-target-cu 0 -d ./att_out -o trace -- ./app` | Per-instruction stall cycles, VMEM latency, LDS bank conflicts, branch divergence |
| **4** | External (rocprof-compute) | `rocprof-compute profile -- ./app` | Full micro-architecture breakdown (external tool) |

Tiers 0–3 are fully implemented in rocinsight.  Tier 4 is handled by `rocprof-compute`
(Omniperf) which rocinsight recommends when appropriate.

### Tier 3: ATT Instruction-Level Stall Analysis

ATT (Advanced Thread Trace) captures per-instruction hardware events at collection time
via the decode library bundled with rocprofiler-sdk.  The decoder runs **inside**
`rocprofv3` — there is no separate decode step.

**Collection:**
```bash
rocprofv3 --att \
          --att-library-path /opt/rocm/lib \
          --att-target-cu 0 \
          -d ./att_output \
          -o trace \
          -- ./my_app
# Produces: att_output/trace.db  AND  att_output/stats_*.csv
```

**Analysis:**
```bash
rocinsight analyze -i ./att_output/trace*.db --att-dir ./att_output
```

ATT is **auto-detected** in interactive sessions: rocinsight checks the `.db` parent
directory for `stats_*.csv` files and passes `att_dir` automatically — no manual flag needed.

**ATT outputs:**
- Per-instruction stall ratio and hit count
- Weighted stall metric combining frequency and severity
- Bottleneck classification: `VMEM Latency`, `LDS Bank Conflict`, `Dependency Chain`,
  `Branch Divergence`
- HIGH priority recommendations when `stall_ratio >= 0.40` or `hitcount >= 6400`
- INFO recommendation when ATT ran successfully but no significant stalls were found

---

## Output Formats

| `--format` | File extension | Description |
|---|---|---|
| `text` | `.txt` (or stdout) | ASCII bar charts, fixed-width tables.  Human-readable terminal output. |
| `json` | `.json` | Schema-conformant JSON document (see [JSON Schema](#json-schema)).  Machine-parseable for CI pipelines and integrations. |
| `markdown` | `.md` | GitHub-Flavored Markdown with tables and priority emoji (🔴 HIGH, 🟠 MEDIUM, 🟡 LOW, ℹ INFO). |
| `webview` | `.html` | Fully self-contained, offline-capable interactive HTML report (see below). |

### Webview Report Features

The `--format webview` output is a single `.html` file with no external dependencies.
Open it in any browser — no web server required.

- **Light/Dark theme toggle** — persisted in `localStorage`; AMD dark theme by default
- **Status summary badges** — Critical/Warning/Low/Info counts at a glance in the header
- **Metric pills** — Runtime (ms), kernel dispatch count, analysis tier, timestamp, DB path
- **Status-colored KPI cards** — color-coded top border (green/amber/red) reflecting health
- **SVG donut gauges** — hardware counter utilization at a glance
- **Priority icons on recommendations** — 🔴 HIGH, 🟠 MEDIUM, 🟡 LOW, ℹ INFO
- **Hover tooltips on every element** — gauges, bars, table headers, and hardware counter
  rows each explain the metric, target thresholds, and next steps
- **Collapsible recommendation cards**
- **FAB scroll-to-top button**
- **Staggered fade-in animations**
- Full embedded JSON payload (`var ANALYSIS = {...}`) for programmatic extraction

---

## LLM Enhancement

All core analysis runs locally with no internet connection.  LLM enhancement is an
**opt-in layer** that generates natural language explanations, interprets results
in application context, and answers targeted questions.

Sensitive data is **automatically redacted** before any LLM API call:

| Original value | Sent to LLM |
|---|---|
| `conv2d_forward_kernel` | `[KERNEL_1]` |
| `[256, 256, 1]` (grid size) | `[GRID_SIZE]` |
| `/home/user/my_app.cpp` | `[REDACTED]` |

Aggregated metrics (time percentages, bottleneck class) are preserved as-is.

### Supported Providers

#### Anthropic Claude (recommended)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
rocinsight analyze -i output.db --llm anthropic
```

Default model: `claude-sonnet-4-20250514`.  Override with `ROCINSIGHT_LLM_MODEL` or
`--llm-model claude-opus-4-6`.

#### OpenAI GPT

```bash
export OPENAI_API_KEY="sk-..."
rocinsight analyze -i output.db --llm openai
```

Default model: `gpt-4-turbo-preview`.

#### Private / enterprise OpenAI-compatible server

For HPC centers, air-gapped environments, or enterprise deployments that run a
locally-hosted OpenAI-compatible API:

```bash
export ROCINSIGHT_LLM_PRIVATE_URL="https://llm-api.example.com/OpenAI"
export ROCINSIGHT_LLM_PRIVATE_HEADERS='{"Ocp-Apim-Subscription-Key": "abc123", "api-version": "preview"}'
rocinsight analyze -i output.db --llm private --llm-private-model gpt-4o
```

The `user` HTTP header is automatically set to `os.getlogin()` (the logged-in OS user)
unless already present in `ROCINSIGHT_LLM_PRIVATE_HEADERS`.

Disable SSL verification for internal CAs:
```bash
export ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL=0   # requires httpx
```

#### Local Ollama

```bash
ollama serve &
ollama pull codellama:13b

rocinsight analyze -i output.db --llm-local ollama --llm-local-model codellama:13b
```

Default Ollama URL: `http://localhost:11434/v1`.  Override with `ROCINSIGHT_LLM_LOCAL_URL`.

### The LLM Reference Guide ("Fence")

The LLM reference guide is a **user-modifiable markdown file** that is loaded as the
LLM system prompt.  It defines:

- Correct ROCm tool names and commands (`rocprofv3`, `rocprof-compute`, `rocprof-sys`)
- AMD GPU hardware specifications (MI100, MI200 series, MI300 series, MI350 series, RDNA2/3)
- Hardware counter per-block limits (hard constraints that prevent error code 38 crashes)
- Performance analysis models (Roofline, Speed-of-Light, Top-Down)
- Recommendation quality standards
- Prohibited behaviors (no fabricated metrics, no deprecated tools)

**File location (lookup order):**
1. `ROCINSIGHT_LLM_REFERENCE_GUIDE` environment variable (if set)
2. Package-relative: `rocinsight/ai_analysis/share/llm-reference-guide.md`
3. System install: `/opt/rocm/share/rocprofiler-sdk/llm-reference-guide.md`

**To change LLM behavior, just edit the guide — no code changes needed:**

```bash
# Find the guide
python3 -c "from rocinsight.ai_analysis.llm_analyzer import get_reference_guide_path; print(get_reference_guide_path())"

# Edit it
nano <path-from-above>
# Changes take effect immediately on the next analysis run
```

---

## Interactive Session

The interactive session (`--interactive`) is an iterative profiling-and-optimization loop
that maintains a persistent LLM conversation across all actions within a session.

### Menu-Driven Mode (`InteractiveSession`)

Launched when `--interactive` is given **without** an app command:

```bash
rocinsight analyze -i output.db --interactive --llm anthropic
```

```
  [p]  Profile   — run a new rocprofv3 command and analyze the output
  [a]  Analyze   — re-analyze current .db and refresh recommendations
  [o]  Optimize  — ask the LLM for specific optimization advice
  [s]  Save      — save session state to disk
  [q]  Quit
```

The `LLMConversation` object is shared across all `[a]`, `[o]`, and code-edit LLM calls
within a session, so the LLM accumulates full context and doesn't repeat itself.

**Source file deduplication**: Files already sent to the LLM in a prior `[o]` call are
not re-transmitted.  The set of sent files is serialized into the session JSON and
restored on `--resume-session`.

**Automatic compaction**: Every `--llm-compact-every N` turns (default 10), older messages
are replaced by an LLM-generated summary.  The most recent 6 turns are always kept verbatim.

**Session resume:**
```bash
# List saved sessions
ls ~/.rocinsight/sessions/*.json

# Resume by session ID (format: YYYY-MM-DD_HH-MM-SS_<source_dir>)
rocinsight analyze -i output.db --interactive \
    --resume-session 2026-03-10_14-23-01_myapp

# Auto-resume prompt when source-dir matches a previous session
rocinsight analyze -i output.db --source-dir ./my_app --interactive
```

### 7-Phase Workflow Mode (`WorkflowSession`)

Launched when `--interactive` is given **with** an app command string:

```bash
rocinsight analyze --interactive "./my_app arg1 arg2" --llm anthropic
```

The workflow automates the full profiling loop:

| Phase | Name | What happens |
|---|---|---|
| 1 | Validate source paths | Warns if `--source` paths are missing |
| 1b | Quick workload analysis | Classifies app type; picks best starter flags |
| 2 | Generate + confirm profiling command | Builds tailored `rocprofv3` command; user approves |
| 3 | Run profiler | Streams stdout; records `TraceRun` on success |
| 4 | AI trace analysis | Runs `analyze_database()` → prints report |
| 5 | Recommendations menu | Adapts options based on priority |
| 6 | Apply changes | AI rewrites source files; `.bak` backup created |
| 7 | Re-profiling prompt | Choose same / edit / AI-recommended command |

**Phase 1b — Quick Workload Analysis:**

Before presenting the profiling command, the tool inspects the app command:

| Detected type | Binary pattern | Extra flags added |
|---|---|---|
| `python_ml` | `python` + PyTorch/JAX/TF | `--hip-trace` |
| `llm_inference` | `vllm`, `ollama`, LLM keywords | `--hip-trace` |
| `python_generic` | `python` without ML keywords | *(default)* |
| `hip_compute` | compiled binary | *(default)* |
| `mpi_multi` | `mpirun`, `srun`, `mpiexec` | `--process-sync` |

When `--source-dir` is also set, Phase 1b runs `SourceAnalyzer` on the source and
uses the recommended flags from its highest-priority profiling recommendation.

**Multi-process support**: When `mpirun`, `torchrun`, DeepSpeed, or DDP keywords are
detected (`uses_fork=True`), the workflow adds `--process-sync` and uses
`-o results_%nid%` so each process writes its own `.db`.  After the run,
`merge_sqlite_dbs()` merges all per-process databases into `merged_processes.db`.

**Phase 5 — INFO vs. Actionable split:**

- All recommendations are INFO (profiling guidance): shows `[r] Re-profile` only — not
  `[a] AI optimization` (which would incorrectly try to edit source for a data gap)
- Any HIGH/MEDIUM recommendation: shows `[a] Address all with AI optimization`
- Once all Tier 1/2 data is collected: shows deeper-tier escalation options:
  - `[d]` → PC sampling (Tier 3 stochastic): `ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1 rocprofv3 --pc-sampling ...`
  - `[t]` → ATT (Tier 3 deterministic): `rocprofv3 --att --att-library-path /opt/rocm/lib ...`
  - `[t]` is hidden once ATT has already been run in the current session

**Phase 6 — AI-edit revert:**

When the AI modifies source files, a `.bak` backup is created automatically.
Typing `revert` / `undo` / `v` at the recompile prompt triggers:

1. Prompt for error output (if none was pasted yet)
2. Immediate restore from `.bak`
3. LLM analysis: root cause of failed edit + suggested alternative
4. Offer to apply the alternative
5. What-next menu: `[f] Try a different fix` / `[p] Continue to re-profiling` / `[q] Quit`

**Cycle prevention**: The session maintains a fingerprint of all flags and counters
collected across **all** previous runs.  A suggested command that adds nothing new is
cleared before it reaches Phase 7, preventing sys-trace ↔ pmc cycling.

### Session Persistence

| Session type | Save triggers | File location |
|---|---|---|
| `InteractiveSession` | `[s]`, `[q]`, Ctrl+C | `~/.rocinsight/sessions/<ts>_<source>.json` |
| `WorkflowSession` | Phase 3 success, Phase 6 edit, exit | `~/.rocinsight/sessions/workflow_<ts>_<app>.json` |

`WorkflowSession` saves are for audit/debugging; resume (`--resume-session`) applies
only to `InteractiveSession`.

---

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | — |
| `OPENAI_API_KEY` | OpenAI GPT API key | — |
| `ROCINSIGHT_LLM_MODEL` | Override default model for `--llm anthropic` or `--llm openai` | Provider default |
| `ROCINSIGHT_LLM_REFERENCE_GUIDE` | Path to custom LLM reference guide | Package `share/llm-reference-guide.md` |
| `ROCINSIGHT_LLM_PRIVATE_URL` | Base URL for private OpenAI-compatible server (**required** for `--llm private`) | — |
| `ROCINSIGHT_LLM_PRIVATE_MODEL` | Model name for private server | — |
| `ROCINSIGHT_LLM_PRIVATE_API_KEY` | API key for private server | `"dummy"` |
| `ROCINSIGHT_LLM_PRIVATE_HEADERS` | Extra HTTP headers as JSON or Python-dict (e.g. `{"Ocp-Apim-Subscription-Key": "..."}`) | — |
| `ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL` | Set to `0` or `false` to skip SSL certificate verification (requires `httpx`) | enabled |
| `ROCINSIGHT_LLM_LOCAL_URL` | Ollama base URL | `http://localhost:11434/v1` |
| `ROCINSIGHT_LLM_LOCAL_MODEL` | Ollama model name | `codellama:13b` |

### `ROCINSIGHT_LLM_PRIVATE_HEADERS` format

Both JSON and Python-dict formats are accepted and single-quoted keys are normalized:

```bash
# JSON (double-quoted keys)
export ROCINSIGHT_LLM_PRIVATE_HEADERS='{"Ocp-Apim-Subscription-Key": "abc123", "api-version": "preview"}'

# Python dict (single-quoted keys — normalized automatically)
export ROCINSIGHT_LLM_PRIVATE_HEADERS="{'Ocp-Apim-Subscription-Key': 'abc123'}"
```

---

## Project Structure

```
experimental/python/rocinsight/
├── README.md                          ← this file
├── pyproject.toml                     ← PEP 517 package metadata; entry point: rocinsight
├── CMakeLists.txt                     ← pip-based install target + ctest wiring
├── tests/                             ← integration tests (use real .db files)
│
└── python/
    └── rocinsight/                    ← Python package
│   ├── __init__.py                    ← version="0.1.0"; exports RocinsightConnection
│   ├── __main__.py                    ← entry point for python -m rocinsight
│   ├── connection.py                  ← RocinsightConnection (pure-sqlite3, no C++)
│   ├── output_config.py               ← minimal replacement for libpyrocpd.output_config
│   ├── analyze.py                     ← ~6000-line analysis orchestrator + formatters
│   ├── tracelens_port.py              ← TraceLens interval arithmetic + kernel categorization
│   │
│   └── ai_analysis/                   ← AI analysis subpackage
│       ├── __init__.py                ← public API exports
│       ├── api.py                     ← analyze_database(), analyze_source(), AnalysisResult
│       ├── exceptions.py              ← AnalysisError hierarchy
│       ├── llm_analyzer.py            ← LLMAnalyzer: single-shot LLM calls with "fence"
│       ├── llm_conversation.py        ← LLMConversation: persistent multi-turn streaming
│       ├── source_analyzer.py         ← Tier 0: SourceAnalyzer, ProfilingPlan
│       ├── interactive.py             ← InteractiveSession + WorkflowSession
│       ├── README.md                  ← ai_analysis module documentation
│       ├── share/
│       │   └── llm-reference-guide.md ← user-modifiable LLM "fence"
│       ├── docs/
│       │   ├── analysis-output.schema.json  ← JSON output schema
│       │   ├── AI_ANALYSIS_API.md           ← Python API reference
│       │   ├── SCHEMA_CHANGELOG.md          ← schema version history
│       │   └── LLM_REFERENCE_GUIDE.md       ← fence documentation
│       └── tests/
│           ├── __init__.py
│           ├── test_api_standalone.py       ← 23 API unit tests
│           ├── test_interactive_context.py  ← interactive session tests
│           ├── test_llm_conversation.py     ← 51 LLMConversation tests
│           ├── test_local_llm.py            ← local Ollama integration tests
│           └── test_tracelens_port.py       ← TraceLens tests
│
└── tests/                             ← integration tests (use real .db files)
    ├── test_analyze.py
    ├── test_analyze_schema.py
    ├── test_ai_analysis_standalone.py
    └── test_guide_filter_standalone.py
```

---

## Architecture: RocinsightConnection

`RocinsightConnection` (`rocinsight/connection.py`) is the pure-Python replacement for
`libpyrocpd.RocpdImportData`.  It gives the analysis layer a `sqlite3.Connection`-like
interface without any compiled extension.

### Single database

```python
from rocinsight import RocinsightConnection

with RocinsightConnection("trace.db") as conn:
    rows = conn.execute("SELECT * FROM rocpd_kernel_dispatch LIMIT 5").fetchall()
```

Opens the file directly — zero overhead, no ATTACH.

### Multiple databases (multi-process runs)

```python
conn = RocinsightConnection(["results_0.db", "results_1.db", "results_2.db"])
```

Creates an in-memory SQLite database and ATTACHes each file as `db0`, `db1`, `db2`.
Then creates UNION ALL views for each standard rocpd table so all analysis queries work
unchanged:

```sql
-- Automatically created for each of:
-- rocpd_kernel_dispatch, rocpd_memory_copy, rocpd_api,
-- rocpd_agent, rocpd_metadata, pmc_events
CREATE TEMP VIEW rocpd_kernel_dispatch AS
    SELECT col1, col2, ... FROM db0.rocpd_kernel_dispatch
    UNION ALL
    SELECT col1, col2, ... FROM db1.rocpd_kernel_dispatch
    UNION ALL
    SELECT col1, col2, ... FROM db2.rocpd_kernel_dispatch
```

Tables that don't exist in a particular shard are skipped gracefully.

### `merge_sqlite_dbs()`

For permanent merges (as opposed to the in-memory UNION ALL view):

```python
from rocinsight import merge_sqlite_dbs

merged_path = merge_sqlite_dbs(
    ["results_0.db", "results_1.db", "results_2.db"],
    output_path="merged_processes.db",   # optional; defaults to sibling file
)
```

Copies the first database, then ATTACHes each subsequent source and
`INSERT OR IGNORE INTO main.<table> SELECT * FROM src<N>.<table>` for every table.

### `execute_statement()`

Compatibility shim used by `analyze.py`:

```python
from rocinsight import execute_statement

# Works with both RocinsightConnection and bare sqlite3.Connection
cursor = execute_statement(conn, "SELECT COUNT(*) FROM rocpd_kernel_dispatch")
```

---

## JSON Schema

rocinsight emits versioned JSON documents.  The schema file lives at:

```
rocinsight/ai_analysis/docs/analysis-output.schema.json
```

| `schema_version` | Produced by | Key additions |
|---|---|---|
| `"0.1.0"` | Tier 1/2 trace analysis (no counters or ATT) | Base fields |
| `"0.2.0"` | Tier 0 source-only (`--source-dir`) | `tier0` field; nullable `execution_breakdown` |
| `"0.3.0"` | Tier 1/2 with TraceLens fields | `interval_timeline`, `kernel_categories`, `short_kernels` |
| `"0.4.0"` | Tier 3 ATT analysis | ATT stall data in recommendations |

### Validate a JSON output document

```bash
pip install jsonschema
python3 -m jsonschema \
  --instance analysis.json \
  rocinsight/ai_analysis/docs/analysis-output.schema.json
```

```python
import json, jsonschema, importlib.resources as pkg

schema = json.loads(
    (pkg.files("rocinsight.ai_analysis") / "docs/analysis-output.schema.json").read_text()
)
with open("analysis.json") as f:
    doc = json.load(f)
jsonschema.validate(instance=doc, schema=schema)
```

Full schema history: [`rocinsight/ai_analysis/docs/SCHEMA_CHANGELOG.md`](rocinsight/ai_analysis/docs/SCHEMA_CHANGELOG.md)

---

## Building with CMake

rocinsight is a pure-Python package.  The CMakeLists.txt provides:
- A `cmake --install` target that runs `pip install --prefix $CMAKE_INSTALL_PREFIX`
- A CTest registration for unit and integration tests

```bash
# Standalone build
cmake -B build -DCMAKE_INSTALL_PREFIX=/opt/rocm experimental/python/rocinsight
cmake --build build
sudo cmake --install build

# As part of the rocm-systems super-repo build
cmake -B super-build \
      -DROCINSIGHT_INSTALL_PYTHON=ON \
      -DROCINSIGHT_BUILD_TESTS=ON \
      rocm-systems-dev

cmake --build super-build --target rocinsight_unit_tests
ctest --test-dir super-build -R rocinsight
```

CMake options:

| Option | Default | Description |
|---|---|---|
| `ROCINSIGHT_INSTALL_PYTHON` | `ON` | Install Python package via pip |
| `ROCINSIGHT_BUILD_TESTS` | `ON` | Register pytest as CTest test |

---

## Testing

rocinsight has two test suites: **unit tests** (no GPU or real .db required) and
**integration tests** (use a real rocprofv3 .db file).

### Quick start

```bash
# 1. Install rocinsight in editable mode (once)
cd experimental/python/rocinsight
pip install -e ".[dev]"

# 2. Run all unit tests
pytest experimental/python/rocinsight/ai_analysis/tests -v

# 3. Run all integration tests (need a real .db)
pytest tests -v --db-path /path/to/trace.db
```

### Unit tests (no GPU required)

Unit tests live in `experimental/python/rocinsight/rocinsight/ai_analysis/tests/`.
They mock all external I/O and cover the full API surface without needing a GPU or
a real profiling database.

```bash
# From the repo root
cd experimental/python/rocinsight
pytest experimental/python/rocinsight/ai_analysis/tests -v

# Run a specific file
pytest experimental/python/rocinsight/ai_analysis/tests/test_api_standalone.py -v

# Run with coverage report
pytest experimental/python/rocinsight/ai_analysis/tests -v --cov=rocinsight --cov-report=term-missing

# If not installed in editable mode, set PYTHONPATH manually
PYTHONPATH=python python3 -m pytest experimental/python/rocinsight/ai_analysis/tests -v
```

Test files and what they cover:

| File | Coverage |
|---|---|
| `test_api_standalone.py` | `analyze_database()`, `AnalysisResult` dataclass, all output formats (text/json/markdown/webview), schema validation |
| `test_connection.py` | `RocinsightConnection` single/multi-DB, UNION ALL views, `merge_sqlite_dbs()`, error handling |
| `test_analyze_core.py` | `compute_time_breakdown()`, `identify_hotspots()`, `analyze_memory_copies()`, `analyze_hardware_counters()`, `_split_pmc_into_passes()` (TCC derived counter isolation), `generate_recommendations()` |
| `test_interactive_context.py` | `InteractiveSession` menu logic, context persistence across menu selections |
| `test_interactive.py` | Full interactive session 7-phase workflow, session save/resume |
| `test_llm_conversation.py` | `LLMConversation` send/compact/serialize, multi-provider mocking (anthropic/openai/private/local) |
| `test_local_llm.py` | Ollama local model integration |
| `test_workflow.py` | `WorkflowSession` 7-phase loop, auto-save, revert-on-failure |
| `test_tracelens_port.py` | TraceLens interval arithmetic and kernel category classification |

### Integration tests (require a real .db file)

Integration tests in `tests/` run against a real rocprofv3 SQLite database.

```bash
# Run all integration tests
cd experimental/python/rocinsight
pytest tests -v --db-path /path/to/trace.db

# A real test database is available after building rocprofiler-sdk:
DB=rocm-systems-dev/projects/rocprofiler-sdk/build/tests/rocprofv3/rocpd/rocpd-input-data/merged_db.db
pytest tests -v --db-path $DB

# Run specific integration test file
pytest tests/test_analyze.py -v --db-path $DB
pytest tests/test_analyze_schema.py -v --db-path $DB
```

Integration test files:

| File | What it covers |
|---|---|
| `test_analyze.py` | End-to-end analysis pipeline against a real .db file |
| `test_analyze_schema.py` | JSON output schema validation against real trace data |
| `test_ai_analysis_standalone.py` | `analyze_database()` API with real database |
| `test_guide_filter_standalone.py` | LLM reference guide loading and filtering |

### CTest (within CMake super-build)

```bash
# Register and run tests via CTest
cmake -B build -DROCINSIGHT_BUILD_TESTS=ON experimental/python/rocinsight
cmake --build build
ctest --test-dir build -R rocinsight -V

# Or inside the full super-repo build
cmake --build super-build --target rocinsight_unit_tests
ctest --test-dir super-build -R rocinsight
```

---

## Differences from rocpd

| Feature | rocpd | rocinsight |
|---|---|---|
| **C++ dependency** | `libpyrocpd` required | None — pure Python stdlib |
| **Database opener** | `RocpdImportData` (C++) | `RocinsightConnection` (sqlite3) |
| **Entry point** | `python -m rocpd analyze` | `rocinsight analyze` or `python -m rocinsight analyze` |
| **Import path** | `from rocpd.ai_analysis import ...` | `from rocinsight.ai_analysis import ...` |
| **Output config** | `libpyrocpd.output_config` | `rocinsight.output_config` (pure Python) |
| **Multi-DB merge** | `rocpd.merge.merge_sqlite_dbs` | `rocinsight.merge_sqlite_dbs` (same logic, no C++) |
| **Analysis logic** | Identical | Identical |
| **LLM providers** | Identical | Identical |
| **Interactive session** | Identical | Identical |
| **Installation** | Requires full SDK build | `pip install ./experimental/python/rocinsight` |

All analysis functions, recommendation rules, output formats, LLM integration, and the
interactive session are **identical** between rocpd and rocinsight — only the database
connection layer differs.

---

## Contributing

### Code style

- Python: PEP 8, max line length 120 (`black --line-length 120 --target-version py38,py310`)
- Type hints on all public functions
- Google-style docstrings on public classes and functions

```bash
# Format
python3 -m black --line-length 120 --target-version py310 rocinsight/

# Lint
python3 -m flake8 --max-line-length 120 --extend-ignore E203,W503 rocinsight/
```

### Adding new analysis rules

1. Add the rule in `analyze.py:generate_recommendations()` (follows existing pattern)
2. Update `ai_analysis/api.py:_build_analysis_result()` if a new field is exposed
3. Update the LLM reference guide if the LLM should reference the new metric
4. Add a unit test in `ai_analysis/tests/test_api_standalone.py`
5. Update `SCHEMA_CHANGELOG.md` if the JSON output structure changes

### Modifying LLM behavior

Edit `rocinsight/ai_analysis/share/llm-reference-guide.md` — no code changes required.
See [`AI_ANALYSIS_API.md`](rocinsight/ai_analysis/docs/AI_ANALYSIS_API.md) for details.

---

## License

MIT License — Copyright (c) 2025 Advanced Micro Devices, Inc.

---

## See Also

- [`rocinsight/ai_analysis/README.md`](rocinsight/ai_analysis/README.md) — ai_analysis subpackage documentation
- [`rocinsight/ai_analysis/docs/AI_ANALYSIS_API.md`](rocinsight/ai_analysis/docs/AI_ANALYSIS_API.md) — Python API reference
- [`rocinsight/ai_analysis/docs/SCHEMA_CHANGELOG.md`](rocinsight/ai_analysis/docs/SCHEMA_CHANGELOG.md) — JSON schema version history
- [`rocinsight/ai_analysis/share/llm-reference-guide.md`](rocinsight/ai_analysis/share/llm-reference-guide.md) — LLM fence document
- [ROCprofiler-SDK documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/)
- [ROCm systems GitHub](https://github.com/ROCm/rocm-systems)
