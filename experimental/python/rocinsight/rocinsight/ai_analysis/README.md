# rocinsight AI Analysis Module

AI-powered GPU performance analysis for AMD ROCm profiling data.

## Overview

This module provides CLI and Python API access to AI-powered analysis of GPU profiling
traces produced by rocprofiler-sdk.  It analyzes rocprofiler-sdk database files (`.db`)
and generates human-readable insights, bottleneck identification, and actionable
optimization recommendations.

### Key Features

- **Local-first analysis** — Works offline, no API calls required by default
- **Tier 0 source analysis** — Scan GPU source code without a trace database (`analyze_source()`)
- **Tier 3 ATT analysis** — Per-instruction stall breakdown from `rocprofv3 --att` output
- **Optional LLM enhancement** — Natural language explanations via Anthropic Claude,
  OpenAI GPT, any OpenAI-compatible private server, or local Ollama
- **User-modifiable "fence"** — Customize LLM behavior by editing the reference guide
- **Privacy-focused** — Kernel names, grid sizes, and file paths are redacted before
  any LLM API call
- **Multiple output formats** — Python objects, JSON, text, Markdown, webview (interactive HTML)
- **Interactive session** — Menu-driven analysis loop with persistent multi-turn LLM
  conversation and session persistence across processes
- **Type-safe API** — Dataclass-based with type hints

---

## Quick Start

### CLI Usage

```bash
# Basic analysis (local mode)
rocinsight analyze -i output.db

# With LLM enhancement
export ANTHROPIC_API_KEY="sk-ant-..."
rocinsight analyze -i output.db --llm anthropic

# Private/enterprise OpenAI-compatible server
export ROCINSIGHT_LLM_PRIVATE_URL="https://llm-api.example.com/OpenAI"
export ROCINSIGHT_LLM_PRIVATE_HEADERS='{"Ocp-Apim-Subscription-Key": "abc123"}'
rocinsight analyze -i output.db --llm private --llm-private-model gpt-4o

# Local Ollama model
rocinsight analyze -i output.db --llm-local ollama --llm-local-model llama3

# With custom analysis prompt
rocinsight analyze -i output.db --llm anthropic --prompt "Why is my matmul kernel slow?"

# JSON output (produces analysis.json)
rocinsight analyze -i output.db --format json -d ./output -o analysis

# Markdown output (produces analysis.md)
rocinsight analyze -i output.db --format markdown -d ./output -o analysis

# Interactive HTML webview (produces analysis.html)
rocinsight analyze -i output.db --format webview -d ./output -o analysis

# Tier 0: static source code analysis (no .db required)
rocinsight analyze --source-dir ./my_app
rocinsight analyze --source-dir ./my_app --format json -d ./output -o plan

# Combined: Tier 0 + Tier 1/2
rocinsight analyze -i output.db --source-dir ./my_app

# Tier 3: ATT instruction-level stall analysis
# Collect (decoder library runs inside rocprofv3 — no separate decode step):
#   rocprofv3 --att --att-library-path /opt/rocm/lib \
#             --att-target-cu 0 -d ./att_out -o trace -- ./app
rocinsight analyze -i ./att_out/trace*.db --att-dir ./att_out

# Interactive menu session (persistent LLM conversation)
rocinsight analyze -i output.db --interactive
rocinsight analyze -i output.db --interactive --llm anthropic

# 7-phase automated workflow (profile → analyze → optimize loop)
rocinsight analyze --interactive "./my_app arg1" --llm anthropic

# Resume a previous interactive session
rocinsight analyze -i output.db --interactive \
    --resume-session 2026-03-10_14-23-01_myapp
```

### Python API Usage

```python
from rocinsight.ai_analysis import analyze_database
from pathlib import Path

# Analyze a trace database (Tier 1/2)
result = analyze_database(Path("output.db"))

# Access results
print(result.summary.overall_assessment)
print(f"Primary bottleneck: {result.summary.primary_bottleneck}")
print(f"Kernel time: {result.execution_breakdown.kernel_time_pct:.1f}%")

# Get recommendations
for rec in result.recommendations.high_priority:
    print(f"[HIGH] {rec.title}")
    print(f"  {rec.description}")
    print(f"  Impact: {rec.estimated_impact}")

# Render output formats
html = result.to_webview()           # self-contained HTML report
md   = result.to_markdown()          # GitHub-Flavored Markdown
j    = result.to_json()              # schema-conformant JSON string
txt  = result.to_text()              # plain text ASCII report

# Tier 0: source analysis
from rocinsight.ai_analysis import analyze_source
plan = analyze_source(Path("./my_app"))
print(plan.suggested_first_command)
print(f"Detected {plan.kernel_count} GPU kernels")

# Tier 3: ATT stall analysis
result_att = analyze_database(Path("att_out/trace.db"), att_dir=Path("att_out"))
```

---

## Module Structure

```
ai_analysis/
├── __init__.py              # Public API exports (all functions, dataclasses, exceptions)
├── api.py                   # Main API: analyze_database(), analyze_source(), AnalysisResult, etc.
├── llm_analyzer.py          # LLMAnalyzer: single-shot LLM calls + "fence" implementation
├── llm_conversation.py      # LLMConversation: persistent multi-turn streaming sessions
├── exceptions.py            # Exception hierarchy (AnalysisError and subclasses)
├── source_analyzer.py       # Tier 0: SourceAnalyzer, ProfilingPlan, DetectedKernel, DetectedPattern
├── interactive.py           # InteractiveSession (menu loop) + WorkflowSession (7-phase)
├── share/
│   └── llm-reference-guide.md  # LLM "fence" — user-modifiable; loaded as system prompt
├── docs/
│   ├── analysis-output.schema.json  # Versioned JSON output schema
│   ├── AI_ANALYSIS_API.md           # Complete Python API reference
│   ├── SCHEMA_CHANGELOG.md          # Schema version history (through v0.3.x; v0.4.0 planned)
│   └── LLM_REFERENCE_GUIDE.md       # Fence documentation
├── tests/
│   ├── __init__.py
│   ├── test_api_standalone.py       # 23 API unit tests (no .db required)
│   ├── test_interactive_context.py  # Interactive session unit tests
│   ├── test_llm_conversation.py     # 51 LLMConversation + integration tests
│   ├── test_local_llm.py            # Local Ollama integration tests
│   └── test_tracelens_port.py       # TraceLens interval arithmetic tests
└── README.md                # This file
```

---

## Architecture: The "Fence"

The LLM reference guide ("fence") is a **user-modifiable Markdown file** that controls
LLM behavior.  It is loaded as the system prompt for every LLM API request.

**Location (lookup order):**
1. `ROCINSIGHT_LLM_REFERENCE_GUIDE` environment variable (if set)
2. Package-relative: `rocinsight/ai_analysis/share/llm-reference-guide.md`
3. System install: `/opt/rocm/share/rocprofiler-sdk/llm-reference-guide.md`

**What the guide defines:**
- **ROCm tool names**: `rocprofv3`, `rocprof-compute`, `rocprof-sys` (never `rocprof` / `rocprof-v2`)
- **AMD GPU hardware specs**: MI100, MI210/250/250X, MI300A/300X/325X, MI350X/355X, RDNA2/3
  (peak FLOPS, HBM bandwidth, CU count, LDS size, ridge points)
- **Hardware counter per-block limits**: enforces the rule that FETCH_SIZE and WRITE_SIZE
  must each be in their own dedicated `rocprofv3` pass (exceeding the limit causes
  rocprofv3 error code 38)
- **Performance models**: Roofline, Speed-of-Light, Top-Down analysis methodology
- **Recommendation quality standards**: must be actionable, specific, and measurable
- **Prohibited behaviors**: no fabricated metrics, no generic advice, no external links

**To change LLM behavior, edit the guide — no code changes needed:**

```bash
# Find the active guide
python3 -c "
from rocinsight.ai_analysis.llm_analyzer import get_reference_guide_path
print(get_reference_guide_path())
"

# Edit it (changes take effect immediately on next analysis)
nano <path-from-above>
```

Example modifications:

```bash
# Add a new GPU: add MI400 specs to the Hardware Specifications section

# Change priority threshold:
# - High Priority: Impacts >5% of total execution time  (was >10%)

# Add company-specific requirements:
# ### Site Policy
# - Always report power consumption if GRBM_GUI_ACTIVE is available
```

---

## Data Flow

```
rocprofv3 --sys-trace --pmc GRBM_COUNT SQ_WAVES -- ./app
    │
    ▼ produces
output.db  (SQLite — rocprofiler-sdk format)
    │
    ▼
rocinsight analyze -i output.db --llm anthropic
    │
    ├─ 1. Local Analysis (always runs — no internet)
    │       Parse rocpd_kernel_dispatch, rocpd_memory_copy, pmc_events, …
    │       compute_time_breakdown()        → kernel%, memcpy%, api%, idle%
    │       identify_hotspots()             → top-N kernels by total duration
    │       analyze_memory_copies()         → per-direction bandwidth
    │       analyze_hardware_counters()     → GPU util%, wave occupancy (Tier 2)
    │       generate_recommendations()     → rule-based findings
    │
    ├─ 2. LLM Enhancement (opt-in)
    │       Load llm-reference-guide.md    → system prompt ("fence")
    │       Sanitize data                  → [KERNEL_N], [GRID_SIZE], [REDACTED]
    │       Call Anthropic / OpenAI API   → natural language explanation
    │
    ▼
Analysis results  →  text / JSON / Markdown / webview (HTML)
```

---

## Analysis Tiers

| Tier | Input | Collection Command | Key Capabilities |
|---|---|---|---|
| **0** | Source code (`--source-dir`) | *(none — static scan)* | Kernel detection, pattern analysis (NO_STREAMS, LOOP_DEVICE_SYNC, missing ROCTx), profiling plan, suggested first command |
| **1** | Trace DB (`-i db.db`) | `rocprofv3 --sys-trace -- ./app` | Hotspot ranking, time breakdown, memcpy overhead, idle time |
| **2** | Trace DB + PMC | `rocprofv3 --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES FETCH_SIZE -- ./app` | GPU utilization %, wave occupancy, HBM bandwidth, Roofline / SOL analysis |
| **3 ATT** | Trace DB + `stats_*.csv` | `rocprofv3 --att --att-library-path /opt/rocm/lib --att-target-cu 0 -d ./att_out -o trace -- ./app` | Per-instruction stall ratio, VMEM latency, LDS bank conflicts, dependency chains, branch divergence |
| **3 PC** | Trace DB + PC sampling | `ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1 rocprofv3 --pc-sampling -d ./pc_out -o trace -- ./app` | Statistical instruction-level hotspots (stochastic) |

### Tier 3 ATT Notes

- The ATT decode library (`librocprof-trace-decoder.so`) runs **inside** `rocprofv3` at
  collection time — there is no separate decode step or standalone binary.
- Do **NOT** use `--att-simd-select 0x0` — that bitmask disables all 4 SIMDs.
  The default (`0xF`) selects all 4 SIMDs and is correct for most workloads.
- ATT output includes `stats_*.csv` files in the `-d` output directory alongside the `.db`.
- In interactive sessions, rocinsight **auto-detects** ATT by checking the `.db` parent
  directory for `stats_*.csv` — no manual `--att-dir` flag needed.
- When no instruction has `stall_ratio >= 0.40` or `hitcount >= 6400`, an INFO
  recommendation is emitted confirming ATT ran successfully with no significant stalls.

---

## API Reference

### Main Functions

```python
# Analyze a trace database (Tier 1/2/3)
from rocinsight.ai_analysis import analyze_database

result = analyze_database(
    database_path: Path,
    *,
    custom_prompt: Optional[str] = None,
    enable_llm: bool = False,
    llm_provider: Optional[str] = None,   # "anthropic" | "openai" | "private"
    llm_api_key: Optional[str] = None,
    output_format: OutputFormat = OutputFormat.PYTHON_OBJECT,
    verbose: bool = False,
    top_kernels: int = 10,
    att_dir: Optional[Path] = None,        # Tier 3 ATT
) -> AnalysisResult

# Analyze source code (Tier 0 — no .db required)
from rocinsight.ai_analysis import analyze_source

plan = analyze_source(
    source_dir: Path,
    *,
    custom_prompt: Optional[str] = None,
    enable_llm: bool = False,
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    verbose: bool = False,
) -> SourceAnalysisResult

# Analyze and return JSON string (optionally write to file)
from rocinsight.ai_analysis import analyze_database_to_json

json_str = analyze_database_to_json(
    database_path: Path,
    output_json_path: Optional[Path] = None,
    **kwargs    # forwarded to analyze_database()
) -> str

# Get filtered recommendations only
from rocinsight.ai_analysis import get_recommendations

recs = get_recommendations(
    database_path: Path,
    priority_filter: Optional[str] = None,   # "high" | "medium" | "low"
    category_filter: Optional[str] = None,   # substring match on category
    **kwargs
) -> List[Recommendation]

# Validate a database without full analysis
from rocinsight.ai_analysis import validate_database

info = validate_database(database_path: Path) -> Dict[str, Any]
# keys: is_valid, tier, has_kernels, has_memory_copies,
#       has_pmc_counters, tables, error
```

### Key Data Classes

```python
@dataclass
class AnalysisResult:
    metadata: AnalysisMetadata          # version, db_file, timestamp, custom_prompt
    profiling_info: ProfilingInfo       # total_duration_ns, profiling_mode, analysis_tier, gpus
    summary: AnalysisSummary            # overall_assessment, primary_bottleneck, confidence, key_findings
    execution_breakdown: ExecutionBreakdown  # kernel/memcpy/api/idle _ns and _pct
    recommendations: RecommendationSet  # .high_priority / .medium_priority / .low_priority lists
    warnings: List[AnalysisWarning]     # severity, message, recommendation
    errors: List[str]
    llm_enhanced_explanation: Optional[str]   # populated when enable_llm=True
    tier0: Optional[SourceAnalysisResult]     # populated when --source-dir used

    # Serialization
    def to_dict() -> dict
    def to_json(indent: int = 2) -> str       # schema-conformant JSON
    def to_text() -> str                       # ASCII report
    def to_markdown() -> str                   # GFM
    def to_webview() -> str                    # self-contained HTML

@dataclass
class SourceAnalysisResult:
    source_dir: str
    analysis_timestamp: str               # ISO 8601
    programming_model: str                # "HIP" | "HIP+ROCm_Libraries" | "PyTorch_HIP" | …
    files_scanned: int
    files_skipped: int
    detected_kernels: List[Dict]          # {name, file, line, launch_type}
    kernel_count: int
    detected_patterns: List[Dict]         # {pattern_id, severity, category, description, count, locations}
    risk_areas: List[str]
    already_instrumented: bool            # True if ROCTx markers found
    roctx_marker_count: int
    recommendations: List[Dict]
    suggested_counters: List[str]
    suggested_first_command: str          # ready-to-run rocprofv3 command
    llm_explanation: Optional[str]
```

### Exception Classes

```python
AnalysisError                     # base class
├── DatabaseNotFoundError
├── DatabaseCorruptedError
├── MissingDataError
├── UnsupportedGPUError
├── LLMAuthenticationError        # propagates; not swallowed as warning
├── LLMRateLimitError             # propagates; not swallowed as warning
├── ReferenceGuideNotFoundError   # shows all attempted paths
├── SourceDirectoryNotFoundError  # analyze_source(): directory not found
└── SourceAnalysisError           # analyze_source(): scan error
```

---

## LLM Enhancement

### Enabling LLM Mode

**CLI:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
rocinsight analyze -i output.db --llm anthropic
```

**Python API:**
```python
result = analyze_database(
    Path("output.db"),
    enable_llm=True,
    llm_provider="anthropic",
    llm_api_key="sk-ant-...",    # or set ANTHROPIC_API_KEY
)
```

### Supported Providers

| Provider flag | Env var | Default model | Notes |
|---|---|---|---|
| `--llm anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` | Recommended |
| `--llm claude-code` | `ANTHROPIC_API_KEY` (Tier 1) or stored CLI credentials (Tier 2) | same as `--llm anthropic` | No separate API key needed when Claude Code CLI is authenticated |
| `--llm openai` | `OPENAI_API_KEY` | `gpt-4-turbo-preview` | — |
| `--llm private` | `ROCINSIGHT_LLM_PRIVATE_URL` | `ROCINSIGHT_LLM_PRIVATE_MODEL` | Any OpenAI-compatible endpoint |
| `--llm local` | `ROCINSIGHT_LLM_LOCAL_URL` | `ROCINSIGHT_LLM_LOCAL_MODEL` | Local Ollama |

Override model: `ROCINSIGHT_LLM_MODEL` env var or `--llm-model <model-name>`.

### Data Sanitization

| Data type | Original | Sent to LLM |
|---|---|---|
| Kernel name | `conv2d_forward_kernel` | `[KERNEL_1]` |
| Grid / block size | `[256, 256, 1]` | `[GRID_SIZE]` |
| Source file path | `/home/user/app.cpp` | `[REDACTED]` |
| Time percentages | `67.3%` | `67.3%` (preserved) |
| Bottleneck classification | `memory_transfer` | `memory_transfer` (preserved) |

---

## Recommendation Rules

### Tier 1 rules (always)

| Condition | Priority | Category |
|---|---|---|
| `memcpy_percent > 20%` | HIGH | Memory Transfer |
| Top kernel `pct_of_total > 50%` | HIGH | Compute Bottleneck |
| `overhead_percent > 15%` | MEDIUM | API Overhead |
| `total_calls > 1000 AND avg_duration < 10µs` | MEDIUM | Launch Overhead |
| Any direction `bandwidth < 10 GB/s` | MEDIUM | Memory Bandwidth |
| No issues found | INFO | *(default)* |

### Tier 2 rules (when `pmc_events` table exists)

| Condition | Priority | Category |
|---|---|---|
| `avg_waves < 16` | HIGH | GPU Occupancy |
| `gpu_utilization < 70%` | MEDIUM | GPU Utilization |

### Tier 3 ATT rules

| Condition | Priority | Category |
|---|---|---|
| `stall_ratio >= 0.40` OR `hitcount >= 6400` | HIGH/MEDIUM | `ATT VMEM Latency` / `ATT LDS Conflict` / `ATT Dependency Chain` / `ATT Branch Divergence` |
| ATT ran, no significant stalls | INFO | ATT analysis complete |

---

## Output Formats

| `--format` | Extension | Description |
|---|---|---|
| `text` | *(stdout)* or `.txt` | ASCII bar charts, fixed-width tables |
| `json` | `.json` | Schema-conformant (v0.1.0–v0.3.x), machine-parseable |
| `markdown` | `.md` | GFM tables, priority emoji (🔴 HIGH, 🟠 MEDIUM, 🟡 LOW, ℹ INFO) |
| `webview` | `.html` | Self-contained interactive HTML — AMD dark theme, hover tooltips, gauges, light/dark toggle, FAB scroll button, staggered animations |

---

## Interactive Session

### `InteractiveSession` (menu-driven)

Invoked with `rocinsight analyze -i output.db --interactive` (no app string):

```
  [p]  Profile app  — run rocprofv3, collect .db
  [a]  Analyze .db  — load existing trace and find bottlenecks
  [o]  Optimize     — AI code optimization suggestions
  [s]  Save session
  [q]  Quit
```

One `LLMConversation` is shared across all `[a]`, `[o]`, and code-edit calls within
the session.  Source files already sent to the LLM are not re-transmitted.  Conversation
is compacted every `--llm-compact-every N` turns (default 10) and restored on
`--resume-session`.

### `WorkflowSession` (7-phase loop)

Invoked with `rocinsight analyze --interactive "./my_app arg1"`:

```
Phase 1   Validate source paths
Phase 1b  Quick workload analysis (classify app type, pick starter flags)
Phase 2   Build + confirm profiling command (user-editable)
Phase 3   Run rocprofv3 (streams output, handles ENV=VALUE prefixes)
Phase 4   Analyze trace DB → print report with ATT auto-detection
Phase 5   Recommendations menu (INFO vs. HIGH/MEDIUM split; Tier 3 escalation)
Phase 6   AI file edit → .bak backup → revert flow on failure
Phase 7   Re-profiling choice: same / edit / AI-recommended / save-and-continue
```

Sessions save to `~/.rocinsight/sessions/`.  `--resume-session` applies to `InteractiveSession` only.

**`[s]` Save is available at every menu** — Phase 5 (all branches), Phase 7, and the
post-revert "What would you like to do next?" menu.  Selecting `[s]` calls
`WorkflowSession._save_session()` and re-shows the current menu without exiting.

### Terminal Rendering (`interactive.py`)

All rendering uses the `rich` library (falls back to raw ANSI if `rich` is not installed).

**AMD brand palette**: accent color `#E0001A` (AMD red) used for menu borders, `[key]`
labels, and `@@` diff headers.

**Rich markup bracket escaping**: In Rich markup, unrecognized `[TAG]` sequences are
silently stripped. Any text that must render as a literal `[WORD]` must be written as
`\[WORD]` in the markup string (Python: `"\\[WORD]"`).

| Helper | Pattern | Output |
|---|---|---|
| `_menu_opt(key, desc)` | `f"[bold #E0001A]\\[{key}][/bold #E0001A]  {desc}"` | `[p]  Profile…` in AMD red |
| `_priority_badge(pri)` | `f"[{style}]\\[{pri}][/{style}]"` | `[HIGH]` in bold red |
| Rec item `[N]` | `f"[bold #E0001A]\\[{i}][/bold #E0001A]"` | `[1]` in AMD red |

**`WorkflowSession` must NOT have `_conv`** — that attribute belongs to `InteractiveSession`.
Any save code in `WorkflowSession` must call `self._save_session()`; never reference `self._conv`,
`self._session`, or `self._store` (those are `InteractiveSession` attributes).

---

## Testing

```bash
cd projects/rocinsight

# Unit tests (no .db or GPU required)
PYTHONPATH=. python3 -m pytest rocinsight/ai_analysis/tests --noconftest -v

# Integration tests (real .db file)
PYTHONPATH=. python3 -m pytest tests -v \
    --db-path /path/to/merged_db.db
```

---

## Configuration

| Environment Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |
| `OPENAI_API_KEY` | OpenAI GPT API key |
| `ROCINSIGHT_LLM_MODEL` | Override default LLM model |
| `ROCINSIGHT_LLM_REFERENCE_GUIDE` | Path to custom reference guide |
| `ROCINSIGHT_LLM_PRIVATE_URL` | Private server base URL (required for `--llm private`) |
| `ROCINSIGHT_LLM_PRIVATE_MODEL` | Private server model name |
| `ROCINSIGHT_LLM_PRIVATE_API_KEY` | Private server API key (default: `"dummy"`) |
| `ROCINSIGHT_LLM_PRIVATE_HEADERS` | Extra HTTP headers as JSON / Python-dict |
| `ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL` | `0` = disable SSL verification (requires `httpx`) |
| `ROCINSIGHT_LLM_LOCAL_URL` | Ollama base URL (default: `http://localhost:11434/v1`) |
| `ROCINSIGHT_LLM_LOCAL_MODEL` | Ollama model (default: `codellama:13b`) |

---

## License

MIT License — Copyright (c) 2025 Advanced Micro Devices, Inc.
