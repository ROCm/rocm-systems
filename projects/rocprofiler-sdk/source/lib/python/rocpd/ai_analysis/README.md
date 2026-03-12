# rocpd AI Analysis Module

AI-powered GPU performance analysis for AMD ROCm profiling data.

## Overview

This module provides both CLI and Python API access to AI-powered analysis of GPU profiling traces. It analyzes rocpd database files and generates human-readable insights, bottleneck identification, and actionable optimization recommendations.

### Key Features

- **Local-first analysis** - Works offline, no API calls required by default
- **Tier 0 source analysis** - Scan GPU source code without a trace database (`analyze_source()`)
- **Optional LLM enhancement** - Natural language explanations via Anthropic Claude, OpenAI GPT, any OpenAI-compatible private server, or local Ollama
- **User-modifiable "fence"** - Customize LLM behavior by editing reference guide
- **Privacy-focused** - Data sanitization for LLM mode (kernel names, grid sizes redacted)
- **Multiple output formats** - Python objects, JSON, text, markdown, webview (interactive HTML)
- **Interactive session** - Menu-driven analysis loop with persistent multi-turn LLM conversation and session persistence
- **Type-safe API** - Dataclass-based with type hints

## Quick Start

### CLI Usage

```bash
# Basic analysis (local mode)
rocpd analyze -i output.db

# With LLM enhancement — Anthropic or OpenAI
export ANTHROPIC_API_KEY="sk-ant-..."
rocpd analyze -i output.db --llm anthropic

# Private/enterprise OpenAI-compatible server
export ROCPD_LLM_PRIVATE_URL="https://llm-api.example.com/OpenAI"
export ROCPD_LLM_PRIVATE_HEADERS='{"Ocp-Apim-Subscription-Key": "abc123", "api-version": "preview"}'
rocpd analyze -i output.db --llm private --llm-private-model gpt-4o

# Local Ollama model
rocpd analyze -i output.db --llm-local ollama --llm-local-model llama3

# With custom prompt
rocpd analyze -i output.db --llm anthropic --prompt "Why is my matmul kernel slow?"

# JSON output (produces analysis.json)
rocpd analyze -i output.db --format json -d ./output -o analysis

# Markdown output (produces analysis.md)
rocpd analyze -i output.db --format markdown -d ./output -o analysis

# Interactive HTML webview (produces analysis.html)
rocpd analyze -i output.db --format webview -d ./output -o analysis

# Tier 0: source code analysis (no .db required)
rocpd analyze --source-dir ./my_app
rocpd analyze --source-dir ./my_app --format json -d ./output -o plan

# Combined: Tier 0 + Tier 1/2
rocpd analyze -i output.db --source-dir ./my_app

# Interactive menu session (persistent LLM conversation, session-persistent)
rocpd analyze -i output.db --interactive
rocpd analyze -i output.db --interactive --llm anthropic
rocpd analyze --source-dir ./my_app --interactive "./my_app arg1" --llm private

# Resume a previous interactive session
rocpd analyze -i output.db --interactive --resume-session 2026-03-10_14-23-01_myapp
```

### Python API Usage

```python
from rocpd.ai_analysis import analyze_database
from pathlib import Path

# Analyze a database
result = analyze_database(Path("output.db"))

# Access results
print(result.summary.overall_assessment)
print(f"Primary bottleneck: {result.summary.primary_bottleneck}")

# Get recommendations
for rec in result.recommendations.high_priority:
    print(f"🔴 {rec.title}")
    print(f"   {rec.description}")
```

## Module Structure

```
ai_analysis/
├── __init__.py              # Public API exports (incl. LLMConversation, load_reference_guide)
├── api.py                   # Main API functions, AnalysisResult, SourceAnalysisResult
├── llm_analyzer.py          # Single-shot LLM integration with "fence" implementation
├── llm_conversation.py      # Persistent multi-turn LLM session (LLMConversation)
├── exceptions.py            # Exception classes (incl. SourceDirectoryNotFoundError)
├── source_analyzer.py       # Tier 0: static source code scanner
├── interactive.py           # Interactive session: InteractiveSession + WorkflowSession
│                            #   SessionData, SessionStore dataclasses
├── tests/
│   ├── __init__.py
│   ├── test_api_standalone.py         # 23 AI analysis API unit tests
│   ├── test_interactive.py            # 22 interactive session unit tests
│   └── test_llm_conversation.py       # 51 LLMConversation + integration tests
├── share/
│   └── llm-reference-guide.md  # LLM "fence" - user-modifiable reference guide
├── docs/
│   ├── AI_ANALYSIS_API.md      # API documentation
│   ├── SCHEMA_CHANGELOG.md     # JSON schema version history (current: v0.2.0)
│   └── LLM_REFERENCE_GUIDE.md  # Fence documentation
└── README.md                # This file
```

## Architecture: The "Fence"

The LLM reference guide ("fence") is a **user-modifiable markdown file** that controls LLM behavior:

**Reference guide search/override order:**
1. `ROCPD_LLM_REFERENCE_GUIDE` environment variable (explicit override — highest priority)
2. Module-relative `ai_analysis/share/llm-reference-guide.md` (i.e. inside the installed package at `/opt/rocm/lib/python3.12/site-packages/rocpd/ai_analysis/share/`)
3. System-wide fallback: `/opt/rocm/share/rocprofiler-sdk/llm-reference-guide.md`

In the common case, edit the file at path 2 (inside the site-packages install). Use the env var override to point to a custom guide without modifying the package.

**What's in the guide:**
- **ROCm Profiling Tools** - Correct tool names and commands (rocprofv3, rocprof-compute, rocprof-sys)
- **Tool Documentation Links** - Official ROCm documentation references
- **AMD GPU Hardware Specs** - MI100, MI210/MI250/MI250X, MI300A/MI300X/MI325X, MI350X/MI355X, RDNA2/RDNA3 specifications with ridge points
- **Performance Analysis Models** - Roofline, Speed-of-Light, Top-Down methodologies
- **Bottleneck Classification** - Rules for identifying compute/memory/latency bottlenecks
- **Optimization Techniques** - AMD-specific optimization strategies
- **Recommendation Standards** - Quality requirements for actionable recommendations
- **Output Format Rules** - Consistent plain text format across all LLM providers

**Enforced Tool Usage:**
- ✅ `rocprofv3` - Kernel-level profiling, counters, API tracing
- ✅ `rocprof-compute` - Roofline analysis, memory hierarchy metrics
- ✅ `rocprof-sys` (also known as `rocsys`) - System-wide, MPI, call-stack sampling
- ❌ NEVER `rocprof` or `rocprof-v2` (deprecated tools)

**How it works:**
1. LLMAnalyzer loads the reference guide at initialization
2. Guide is included in every LLM API request as system prompt
3. LLM generates analysis following the guide's rules strictly
4. **To change LLM behavior, just edit the guide - no code changes**
5. All profiling commands are validated against official ROCm documentation

Example modification:

```bash
# Edit the reference guide
sudo nano /opt/rocm/lib/python3.12/site-packages/rocpd/ai_analysis/share/llm-reference-guide.md

# Add new GPU specs, update tool commands, or change priority thresholds
# Save and exit - changes take effect immediately on next analysis
```

See [LLM Reference Guide Documentation](docs/LLM_REFERENCE_GUIDE.md) for details.

## Data Flow

```
rocprofv3 --sys-trace --pmc GRBM_COUNT -- ./app
    ↓
output.db created (SQLite database)
    ↓
rocpd analyze -i output.db --llm anthropic
    ↓
┌─────────────────────────────────────────┐
│ 1. Local Analysis (always runs)        │
│    - Parse database                     │
│    - Calculate metrics                  │
│    - Apply performance models           │
│    - Generate recommendations           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. LLM Enhancement (optional)           │
│    - Load reference guide ("fence")     │
│    - Sanitize data (privacy)            │
│    - Call Anthropic/OpenAI API          │
│    - Generate natural language output   │
└─────────────────────────────────────────┘
    ↓
Analysis results (text/JSON/markdown/webview)
```

## Analysis Tiers

| Tier | Data Required | Analysis Capabilities |
|------|---------------|----------------------|
| **Tier 0** | Source code directory (`--source-dir`) | Kernel detection, pattern scanning, profiling plan, suggested first command |
| **Tier 1** | Trace data (`-i db.db`) | Kernel hotspots, time breakdown, memory copy overhead |
| **Tier 2** | Trace + hardware counters (`--pmc`) | Roofline model, Speed-of-Light metrics, bottleneck classification |
| **Tier 3** | Trace + PC sampling (`--pc-sampling`) | Instruction-level hotspots within kernels |
| **Tier 4** | Trace + thread trace | Full instruction timeline, stall analysis |

Tiers 0–2 are implemented and production-ready. The interactive session automatically
suggests `ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1 rocprofv3 --pc-sampling` (Tier 3)
once all Tier 1/2 data has been collected.

## API Reference

### Main Functions

```python
# Analyze database and return result object (Tier 1/2)
def analyze_database(
    database_path: Path,
    *,
    custom_prompt: Optional[str] = None,
    enable_llm: bool = False,
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    output_format: OutputFormat = OutputFormat.PYTHON_OBJECT,
    verbose: bool = False,
    top_kernels: int = 10,
) -> AnalysisResult

# Analyze source code directory and return profiling plan (Tier 0)
def analyze_source(
    source_dir: Path,
    *,
    custom_prompt: Optional[str] = None,
    enable_llm: bool = False,
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    verbose: bool = False,
) -> SourceAnalysisResult

# Analyze and return JSON
def analyze_database_to_json(
    database_path: Path,
    output_json_path: Optional[Path] = None,
    **kwargs
) -> str

# Get filtered recommendations
def get_recommendations(
    database_path: Path,
    priority_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    **kwargs
) -> List[Recommendation]

# Validate database
def validate_database(database_path: Path) -> Dict[str, Any]
```

### Data Classes

```python
@dataclass
class AnalysisResult:
    metadata: AnalysisMetadata
    profiling_info: ProfilingInfo
    summary: AnalysisSummary
    execution_breakdown: ExecutionBreakdown
    recommendations: RecommendationSet
    warnings: List[AnalysisWarning]
    errors: List[str]
    llm_enhanced_explanation: Optional[str]  # If LLM enabled
    tier0: Optional[SourceAnalysisResult]    # If --source-dir also provided

    # Methods
    def to_dict() -> Dict[str, Any]
    def to_json(indent: int = 2) -> str
    def to_text() -> str
    def to_markdown() -> str
    def to_webview() -> str  # Self-contained interactive HTML report

@dataclass
class SourceAnalysisResult:
    source_dir: str
    analysis_timestamp: str
    programming_model: str        # "HIP", "HIP+ROCm_Libraries", "PyTorch_HIP", etc.
    files_scanned: int
    files_skipped: int
    detected_kernels: List[Dict]  # {name, file, line, launch_type}
    kernel_count: int
    detected_patterns: List[Dict] # {pattern_id, severity, category, description, count, locations}
    risk_areas: List[str]
    already_instrumented: bool
    roctx_marker_count: int
    recommendations: List[Dict]   # Same shape as generate_recommendations() output
    suggested_counters: List[str]
    suggested_first_command: str
    llm_explanation: Optional[str]
```

### Exceptions

```python
AnalysisError (base)
├── DatabaseNotFoundError
├── DatabaseCorruptedError
├── MissingDataError
├── UnsupportedGPUError
├── LLMAuthenticationError
├── LLMRateLimitError
├── ReferenceGuideNotFoundError
├── SourceDirectoryNotFoundError   # analyze_source(): directory not found
└── SourceAnalysisError            # analyze_source(): scanning error
```

## LLM Enhancement

### Enabling LLM Mode

**Option 1: Environment variable**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
rocpd analyze -i output.db --llm anthropic
```

**Option 2: Python API**

```python
result = analyze_database(
    Path("output.db"),
    enable_llm=True,
    llm_provider="anthropic",
    llm_api_key="sk-ant-..."
)
```

### Supported Providers

- **Anthropic Claude** (recommended)
  - Provider: `"anthropic"`
  - Env var: `ANTHROPIC_API_KEY`
  - Model: `claude-sonnet-4-20250514`

- **OpenAI GPT**
  - Provider: `"openai"`
  - Env var: `OPENAI_API_KEY`
  - Model: `gpt-4-turbo-preview`

- **Private/enterprise server** (any OpenAI-compatible endpoint)
  - Provider: `"private"` (`--llm private`)
  - Required: `ROCPD_LLM_PRIVATE_URL` — base URL of the server
  - Required: `ROCPD_LLM_PRIVATE_MODEL` or `--llm-private-model`
  - Optional: `ROCPD_LLM_PRIVATE_API_KEY` (default: `"dummy"` for header-authenticated servers)
  - Optional: `ROCPD_LLM_PRIVATE_HEADERS` — JSON or Python-dict of extra request headers
    (e.g. `{"Ocp-Apim-Subscription-Key": "abc", "api-version": "preview"}`)
    The `user` header is auto-set to `os.getlogin()` unless already present in `ROCPD_LLM_PRIVATE_HEADERS`
  - Optional: `ROCPD_LLM_PRIVATE_VERIFY_SSL=0` — disable SSL verification (requires `httpx`)

- **Local Ollama**
  - Provider: `--llm-local ollama`
  - Env var: `ROCPD_LLM_LOCAL_URL` (default: `http://localhost:11434/v1`)
  - Env var: `ROCPD_LLM_LOCAL_MODEL` (default: `codellama:13b`)

### Data Sanitization

When LLM mode is enabled, sensitive data is automatically redacted:

| Original | Sanitized |
|----------|-----------|
| `conv2d_forward_kernel` | `[KERNEL_1]` |
| `[256, 256, 1]` | `[GRID_SIZE]` |
| `/home/user/app.cpp` | `[REDACTED]` |

Aggregated metrics (time percentages, bottleneck classifications) are preserved.

## Examples

### Example 1: Basic Analysis

```python
from rocpd.ai_analysis import analyze_database
from pathlib import Path

result = analyze_database(Path("output.db"))

print(f"Summary: {result.summary.overall_assessment}")
print(f"Bottleneck: {result.summary.primary_bottleneck}")
print(f"Kernel time: {result.execution_breakdown.kernel_time_pct:.1f}%")
print(f"Memory copy: {result.execution_breakdown.memcpy_time_pct:.1f}%")

print("\nHigh Priority Recommendations:")
for rec in result.recommendations.high_priority:
    print(f"  - {rec.title}")
```

### Example 2: With LLM Enhancement

```python
import os
from rocpd.ai_analysis import analyze_database
from pathlib import Path

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

result = analyze_database(
    database_path=Path("output.db"),
    enable_llm=True,
    llm_provider="anthropic",
    custom_prompt="Focus on memory bottlenecks"
)

# LLM-generated natural language explanation
print(result.llm_enhanced_explanation)
```

### Example 3: JSON Output

```python
from rocpd.ai_analysis import analyze_database_to_json
from pathlib import Path

json_output = analyze_database_to_json(
    database_path=Path("output.db"),
    output_json_path=Path("analysis.json")
)

# JSON is also returned as string
import json
data = json.loads(json_output)
print(f"Analysis tier: {data['profiling_info']['analysis_tier']}")
```

### Example 4: Interactive HTML Webview

```bash
# Generate a self-contained HTML report for browser viewing
# Output file extension is applied automatically (.html for webview)
rocpd analyze -i output.db --format webview -d ./reports -o my_trace
# Produces: ./reports/my_trace.html
```

```python
from rocpd.ai_analysis import analyze_database
from pathlib import Path

result = analyze_database(Path("output.db"))
html_report = result.to_webview()
Path("analysis.html").write_text(html_report)
```

The HTML report is a fully self-contained, offline-capable file with:
- **Light/Dark theme toggle** — persisted in `localStorage`; defaults to AMD dark theme
- **Status summary badges** — Critical/Warning counts visible in the header at a glance
- **Metric pills row** — Runtime, kernel count, tier, timestamp, and DB path in the header
- **Status-colored KPI cards** — Kernel %, bottleneck type, runtime, and tier cards each
  have a green/amber/red top border reflecting health status
- **Priority icons on recommendations** — 🔴 HIGH, 🟠 MEDIUM, 🟡 LOW, ℹ INFO
- **FAB scroll-to-top button** — Floating button appears after scrolling
- **Staggered fade-in animations** on section cards
- **Hover tooltips on every visual element** — gauges, bars, table headers, counter rows,
  and overview stats explain what each metric measures, target thresholds, and how to
  act on issues. Hardware counter rows (GRBM_*, SQ_*, TCP/TCC, FETCH_SIZE, etc.)
  include educational content about the underlying hardware event being counted.

### Example 5: roc-optiq Integration

```python
from rocpd.ai_analysis import analyze_database
from pathlib import Path

def load_trace_for_optiq(trace_path: str):
    """Load trace and extract insights for Optiq UI"""
    result = analyze_database(Path(trace_path))

    return {
        "summary": result.summary.overall_assessment,
        "bottleneck": result.summary.primary_bottleneck,
        "recommendations": [
            {
                "title": rec.title,
                "description": rec.description,
                "priority": rec.priority
            }
            for rec in result.recommendations.high_priority[:3]
        ],
        "breakdown": {
            "kernel_pct": result.execution_breakdown.kernel_time_pct,
            "memcpy_pct": result.execution_breakdown.memcpy_time_pct
        }
    }
```

## Interactive Session

The interactive session (`--interactive`) launches a menu-driven loop for iterative profiling analysis. It maintains a **persistent multi-turn `LLMConversation`** across all calls within the same session — the LLM accumulates full message history and doesn't repeat itself.

### Session menu

```
[p] Profile   — run a new rocprofv3 command and analyze the output .db
[a] Analyze   — re-analyze the current .db and update recommendations
[o] Optimize  — ask the LLM for optimization suggestions
[s] Save      — save session to disk
[q] Quit
```

### LLM conversation persistence

`InteractiveSession` holds one `LLMConversation` for the entire session:
- All `[a]`, `[o]`, and code-edit LLM calls share the same conversation object
- The LLM sees the full message history from earlier in the session
- History is automatically compacted to stay within context limits (`--llm-compact-every N`, default 10 turns)
- Source files are tracked: a file sent once is not re-transmitted on repeat calls (only new files are sent); the file set is serialized into the session JSON and restored on `--resume-session`
- On `[s]` save, the conversation state is serialized into the session file
- On `--resume-session`, the conversation is restored so the LLM picks up exactly where it left off

### Phase 1b: Quick workload analysis (WorkflowSession)

Before presenting the initial profiling command in Phase 2, `WorkflowSession` runs a
lightweight workload analysis to pick the best starter flags:

1. **App-command heuristics** — always runs; inspects binary name and arguments:
   - `python` + ML keywords (torch, jax, paddle…) → `python_ml`; adds `--hip-trace`
   - `python` + LLM keywords (vllm, llama, gpt…) → `llm_inference`; adds `--hip-trace`
   - `python` without ML → `python_generic`; adds `--hip-trace`
   - MPI/Slurm launchers (`mpirun`, `srun`…) → warns about multi-rank capture limits
   - Compiled HIP/ROCm binary → `hip_compute`; uses default flag set
   - Multi-process patterns (torchrun, DDP, DeepSpeed) → warns about worker capture

2. **Tier 0 source analysis** — if `--source-dir` was provided, runs `SourceAnalyzer`
   on the source directory and extracts the recommended flags from its highest-priority
   profiling recommendation; overrides the pure-heuristic flag set.

3. **Fallback** — if neither source analysis nor heuristics yield specific flags, the
   safe default is used: `--sys-trace --kernel-trace --memory-copy-trace --stats`.

The analysis output is printed before the command box so the user can see what was
detected and why specific flags were chosen. The user always confirms or edits the
command in Phase 2.

**Example output:**
```
── Quick Workload Analysis ──────────────────────────────────────
  Detected: Python + ML framework (PyTorch / JAX / TF)
  Source scan: 14 files, 3 kernels, model=hip_python
  Source analysis suggests: rocprofv3 --sys-trace --hip-trace --kernel-trace --stats ...
  Starter command basis: source analysis

╭──────────────────────────────────────────────────────────────────╮
│  Profiling Command                                               │
│  rocprofv3 --sys-trace --hip-trace --kernel-trace --stats ...   │
╰──────────────────────────────────────────────────────────────────╯
  Would you like the interactive tool to run this command? [Y/n]
```

### Cycle prevention and going deeper (WorkflowSession)

The 7-phase `WorkflowSession` (`--interactive "<app>"`) automatically detects and
breaks counter-collection/API-tracing cycles:

- **Fingerprint all collection flags** — when deciding whether to re-suggest a command,
  the session checks `--sys-trace`, `--hip-trace`, `--kernel-trace`, `--memory-copy-trace`,
  `--hsa-trace`, `--stats`, and individual `--pmc` counter names.
- **Compares against all prior runs** — the dedup check looks at the union of everything
  collected across all previous trace runs, not just the last one.
- **Tier 3 escalation** — once all Tier 1/2 data has been collected, Phase 5 shows a
  "go deeper" menu:
  - TraceLens interval + kernel-category analysis is already shown in the report.
  - `[d]` builds a PC sampling command and wires it into Phase 7 as option `[3]`:
    ```
    ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1 rocprofv3 --pc-sampling \
      -d /tmp/rocpd_trace/run_<ts> -o results -- <app>
    ```
  - `ENV=VALUE` prefixes in commands are automatically extracted and injected into the
    subprocess environment (no `shell=True` needed).

### AI-edit revert

When the AI modifies source files (Phase 6), the session backs up each file to `<file>.bak`:

- **During recompile wait** — type `revert` (or `undo`/`v`) to instantly restore the
  original file. Paste compilation error text first and it will be included as context.
- **When profiling fails** (Phase 3) — `[v]` appears in the retry menu if there are AI
  edits; selecting it reverts the last edit and prompts recompile before retrying.
- **Session learning** — on every revert, the failure reason is injected into the active
  `LLMConversation` as a `user`/`assistant` exchange so the LLM knows what went wrong
  and avoids repeating the same pattern later in the session.

### AI-suggested commands

After the LLM responds to `[o]`, the session scans the response text for `rocprofv3 ...` commands and combines them with structured commands from the current recommendation list. If any are found, the user is offered a numbered menu to run one immediately. If run, the resulting `.db` is auto-analyzed and the LLM is notified.

### Session persistence

Sessions are saved as JSON under `~/.rocpd/sessions/` by default.

> **Note:** `--resume-session` applies only to **`InteractiveSession`** (the menu-driven
> `[p]/[a]/[o]/[s]/[q]` mode, triggered by `rocpd analyze -i db.db --interactive` **without**
> a `"<app_command>"` argument). `WorkflowSession` (7-phase workflow) starts a fresh state
> each invocation and does not support resume.

```bash
# Start a new InteractiveSession
rocpd analyze -i output.db --interactive --llm anthropic

# With private enterprise server
rocpd analyze -i output.db --interactive --llm private

# Control compaction interval (default 10 turns)
rocpd analyze -i output.db --interactive --llm anthropic --llm-compact-every 5

# List available session IDs (files in ~/.rocpd/sessions/)
ls ~/.rocpd/sessions/*.json | xargs -I{} python3 -c \
    "import json,sys; d=json.load(open('{}'));print(d['session_id'],'|',d['source_dir'])"

# Resume an existing session — restores LLM conversation, sent files, and history
# Session ID format: YYYY-MM-DD_HH-MM-SS_<source_dir_basename>
rocpd analyze -i output.db --interactive --resume-session 2026-03-10_14-23-01_myapp

# If the source dir matches a previous session, the tool auto-prompts to resume
# (no --resume-session needed)
rocpd analyze -i output.db --source-dir ./my_app --interactive
```

---

## Testing

### Unit Tests

```bash
# Run from /tmp to avoid circular import of libpyrocpd
ROCPD_SYS=/opt/rocm-7.0.0/lib/python3.12/site-packages
TEST_DIR=/path/to/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/tests

# All tests
cd /tmp && PYTHONPATH="${ROCPD_SYS}" python3 -m pytest ${TEST_DIR} --noconftest -v

# Interactive session tests only
cd /tmp && PYTHONPATH="${ROCPD_SYS}" python3 -m pytest ${TEST_DIR}/test_interactive.py --noconftest -v

# LLMConversation + integration tests
cd /tmp && PYTHONPATH="${ROCPD_SYS}" python3 -m pytest ${TEST_DIR}/test_llm_conversation.py --noconftest -v
```

### Integration Tests

```bash
cd rocm-systems-dev/projects/rocprofiler-sdk/build
ctest -R rocpd-ai-analysis
```

### Manual Testing

```bash
# Generate test trace
rocprofv3 --sys-trace --pmc GRBM_COUNT SQ_WAVES -- ./sample_app

# Analyze
rocpd analyze -i output.db

# With LLM (requires API key)
export ANTHROPIC_API_KEY="sk-ant-..."
rocpd analyze -i output.db --llm anthropic
```

## Configuration

### Environment Variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |
| `OPENAI_API_KEY` | OpenAI GPT API key |
| `ROCPD_LLM_MODEL` | Override default model for anthropic or openai provider |
| `ROCPD_LLM_REFERENCE_GUIDE` | Path to custom reference guide (overrides package default) |
| `ROCPD_LLM_PRIVATE_URL` | Base URL for private/enterprise OpenAI-compatible server (required for `--llm private`) |
| `ROCPD_LLM_PRIVATE_MODEL` | Model name for private server |
| `ROCPD_LLM_PRIVATE_API_KEY` | API key for private server (default: `"dummy"`) |
| `ROCPD_LLM_PRIVATE_HEADERS` | JSON or Python-dict of extra HTTP request headers (e.g. `{"Ocp-Apim-Subscription-Key": "..."}`) |
| `ROCPD_LLM_PRIVATE_VERIFY_SSL` | Set to `0` or `false` to disable SSL cert verification (requires `httpx`) |
| `ROCPD_LLM_LOCAL_URL` | Base URL for local Ollama endpoint (default: `http://localhost:11434/v1`) |
| `ROCPD_LLM_LOCAL_MODEL` | Model name for local Ollama (default: `codellama:13b`) |

### Reference Guide Location

Default: `/opt/rocm/share/rocprofiler-sdk/llm-reference-guide.md`

Override:
```bash
export ROCPD_LLM_REFERENCE_GUIDE=/path/to/custom-guide.md
```

## Documentation

- **[AI Analysis API Documentation](docs/AI_ANALYSIS_API.md)** - Complete API reference
- **[LLM Reference Guide Documentation](docs/LLM_REFERENCE_GUIDE.md)** - How to customize LLM behavior
- **[JSON Output Schema](docs/analysis-output.schema.json)** - Stable JSON contract (v0.1.0)
- **[Schema Changelog](docs/SCHEMA_CHANGELOG.md)** - Schema version history

## Development

### Adding New Analysis Features

1. Add analysis logic to `analyze.py` (main rocpd module)
2. Update `api.py` to expose new data in `AnalysisResult`
3. Update reference guide if LLM should use new feature
4. Add tests

### Modifying LLM Behavior

**Don't modify code.** Edit the reference guide instead:

```bash
sudo nano /opt/rocm/share/rocprofiler-sdk/llm-reference-guide.md
```

See [LLM Reference Guide Documentation](../../../docs/LLM_REFERENCE_GUIDE.md) for examples.

## Troubleshooting

### Reference Guide Not Found

```bash
# Check which path is being used
python3 -c "from rocpd.ai_analysis.llm_analyzer import get_reference_guide_path; print(get_reference_guide_path())"

# Copy from source
sudo cp share/llm-reference-guide.md /opt/rocm/share/rocprofiler-sdk/

# Or use environment variable
export ROCPD_LLM_REFERENCE_GUIDE=/path/to/guide.md
```

### LLM Authentication Errors

```bash
# Verify API key is set
echo $ANTHROPIC_API_KEY

# Test API key directly
python3 << EOF
import anthropic
client = anthropic.Anthropic(api_key="sk-ant-...")
print("API key valid!")
EOF
```

### Database Errors

```bash
# Validate database
python3 << EOF
from rocpd.ai_analysis import validate_database
from pathlib import Path

validation = validate_database(Path("output.db"))
print(f"Valid: {validation['is_valid']}")
print(f"Tier: {validation['tier']}")
print(f"Tables: {validation['tables']}")
EOF
```

## Contributing

- Follow existing code style (PEP 8)
- Add type hints
- Write docstrings (Google style)
- Add unit tests
- Update documentation

## License

MIT License - Copyright (c) 2025 Advanced Micro Devices, Inc.

## Support

- File issues on GitHub
- See [rocprofiler-sdk documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/)
- ROCm community: https://rocm.docs.amd.com/
