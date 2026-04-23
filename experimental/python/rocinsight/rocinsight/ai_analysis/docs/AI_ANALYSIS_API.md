# rocinsight AI Analysis Python API Documentation

**Version:** 0.4.0
**Module:** `rocinsight.ai_analysis`

> **Note:** This module was extracted from `rocpd` (rocprofiler-sdk) into the standalone
> `rocinsight` package.  All import paths use `rocinsight.ai_analysis` instead of
> `rocinsight.ai_analysis`.  The API, data structures, and behavior are identical.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Data Classes](#data-classes)
6. [Output Formats](#output-formats)
7. [LLM Enhancement](#llm-enhancement)
8. [Error Handling](#error-handling)
9. [Integration Examples](#integration-examples)
10. [Bug Fixes & Behavioral Changes](#bug-fixes--behavioral-changes)

---

## Overview

The rocinsight AI Analysis API provides programmatic access to AI-powered GPU performance analysis. It's designed for integration with visualization tools (like Optiq), automated analysis pipelines, and custom workflows.

**Key Features:**

- ✅ **Local-first analysis** - Works offline, no API calls required
- ✅ **Tier 0 source analysis** - Scan source code without a trace database (`analyze_source()`)
- ✅ **Optional LLM enhancement** - Natural language explanations via Anthropic Claude, OpenAI GPT, any OpenAI-compatible private server, or local Ollama
- ✅ **Multiple output formats** - Python objects, JSON, text, markdown, webview (interactive HTML)
- ✅ **Privacy-focused** - Data sanitization for LLM mode
- ✅ **User-modifiable** - Customize LLM behavior via reference guide
- ✅ **Persistent conversations** - `LLMConversation` class for multi-turn streaming sessions
- ✅ **Type-safe** - Dataclass-based API with type hints

---

## Installation

rocinsight is a standalone package — no rocprofiler-sdk build required.

```bash
# rocinsight can be installed standalone:
pip install ./projects/rocinsight

# No additional installation needed for local-only analysis

# For LLM enhancement, install provider SDKs:
pip install anthropic  # For Anthropic Claude
pip install openai     # For OpenAI GPT
```

---

## Quick Start

### Basic Analysis (Local Mode)

```python
from rocinsight.ai_analysis import analyze_database
from pathlib import Path

# Analyze a database file
result = analyze_database(Path("output.db"))

# Access results
print(result.summary.overall_assessment)
print(f"Primary bottleneck: {result.summary.primary_bottleneck}")
print(f"Confidence: {result.summary.confidence:.0%}")

# Get recommendations
for rec in result.recommendations.high_priority:
    print(f"🔴 {rec.title}")
    print(f"   {rec.description}")
    print(f"   Impact: {rec.estimated_impact}")
```

### With LLM Enhancement

```python
import os
from rocinsight.ai_analysis import analyze_database
from pathlib import Path

# Set API key
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

# Analyze with LLM enhancement
result = analyze_database(
    database_path=Path("output.db"),
    enable_llm=True,
    llm_provider="anthropic",
    custom_prompt="Why is my matmul kernel slow?"
)

# LLM-enhanced natural language explanation
print(result.llm_enhanced_explanation)
```

### JSON Output

```python
from rocinsight.ai_analysis import analyze_database_to_json
from pathlib import Path

# Generate JSON output
json_output = analyze_database_to_json(
    database_path=Path("output.db"),
    output_json_path=Path("analysis.json")  # Optional: save to file
)

# JSON string is also returned
print(json_output)
```

### Webview (Interactive HTML)

```python
from rocinsight.ai_analysis import analyze_database
from pathlib import Path

result = analyze_database(Path("output.db"))
Path("analysis.html").write_text(result.to_webview())
# Open analysis.html in any browser - no server required
```

Or via CLI (file extension applied automatically):

```bash
rocinsight analyze -i output.db --format webview -d ./output -o analysis
# Produces: ./output/analysis.html
```

---

## API Reference

### Main Functions

#### `analyze_database()`

Main entry point for performance analysis.

```python
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
) -> AnalysisResult:
```

**Parameters:**

- `database_path` (Path): Path to rocprofiler-sdk database file (.rpd or .db)
- `custom_prompt` (str, optional): Natural language question to guide analysis
  - Example: `"Why is kernel X slow?"`
- `enable_llm` (bool): Enable LLM-powered enhancements (default: False)
- `llm_provider` (str, optional): LLM provider ("anthropic" or "openai")
- `llm_api_key` (str, optional): API key (or use environment variable)
- `output_format` (OutputFormat): Output format (default: PYTHON_OBJECT)
- `verbose` (bool): Enable verbose logging (default: False)
- `top_kernels` (int): Number of top kernels to analyze (default: 10)

**Returns:**

- `AnalysisResult`: Complete analysis results object

**Raises:**

- `DatabaseNotFoundError`: Database file doesn't exist
- `DatabaseCorruptedError`: Database schema is invalid
- `MissingDataError`: Required tables missing
- `LLMAuthenticationError`: LLM API key invalid (if enable_llm=True)

**Example:**

```python
from rocinsight.ai_analysis import analyze_database, OutputFormat
from pathlib import Path

result = analyze_database(
    database_path=Path("output.db"),
    custom_prompt="Focus on memory bottlenecks",
    enable_llm=True,
    llm_provider="anthropic",
    verbose=True,
    top_kernels=20
)
```

---

#### `analyze_database_to_json()`

Analyze database and return JSON output.

```python
def analyze_database_to_json(
    database_path: Path,
    output_json_path: Optional[Path] = None,
    **kwargs
) -> str:
```

**Parameters:**

- `database_path` (Path): Path to rocprofiler-sdk database file
- `output_json_path` (Path, optional): Save JSON to this file
- `**kwargs`: Additional arguments passed to `analyze_database()`

**Returns:**

- `str`: JSON string

**Example:**

```python
from rocinsight.ai_analysis import analyze_database_to_json
from pathlib import Path

json_str = analyze_database_to_json(
    database_path=Path("output.db"),
    output_json_path=Path("analysis.json"),
    enable_llm=True,
    llm_provider="anthropic"
)
```

---

#### `get_recommendations()`

Get filtered recommendations from analysis.

```python
def get_recommendations(
    database_path: Path,
    priority_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    **kwargs
) -> List[Recommendation]:
```

**Parameters:**

- `database_path` (Path): Path to rocprofiler-sdk database file
- `priority_filter` (str, optional): Filter by priority ("high", "medium", "low")
- `category_filter` (str, optional): Filter by category ("memory", "compute", etc.)
- `**kwargs`: Additional arguments passed to `analyze_database()`

**Returns:**

- `List[Recommendation]`: Filtered recommendations

**Example:**

```python
from rocinsight.ai_analysis import get_recommendations
from pathlib import Path

# Get only high-priority recommendations
high_priority_recs = get_recommendations(
    database_path=Path("output.db"),
    priority_filter="high"
)

for rec in high_priority_recs:
    print(f"{rec.title}: {rec.estimated_impact}")
```

---

#### `validate_database()`

Validate database without performing full analysis.

```python
def validate_database(database_path: Path) -> Dict[str, Any]:
```

**Parameters:**

- `database_path` (Path): Path to rocprofiler-sdk database file

**Returns:**

- `Dict`: Validation results with keys:
  - `is_valid` (bool): Database is valid
  - `tier` (int): Analysis tier (1=trace, 2=counters, 3=pc_sampling)
  - `has_kernels` (bool): Has kernel data
  - `has_memory_copies` (bool): Has memory copy data
  - `has_counters` (bool): Has hardware counters
  - `has_pc_sampling` (bool): Has PC sampling data
  - `tables` (List[str]): List of table names

**Example:**

```python
from rocinsight.ai_analysis import validate_database
from pathlib import Path

validation = validate_database(Path("output.db"))

print(f"Valid: {validation['is_valid']}")
print(f"Analysis tier: {validation['tier']}")
print(f"Has counters: {validation['has_counters']}")
```

---

#### `analyze_source()`

Analyze source code directory (Tier 0) and return a profiling plan. No database required.

```python
def analyze_source(
    source_dir: Path,
    *,
    custom_prompt: Optional[str] = None,
    enable_llm: bool = False,
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    verbose: bool = False,
) -> SourceAnalysisResult:
```

**Parameters:**

- `source_dir` (Path): Directory containing GPU source code (`.hip`, `.cpp`, `.cu`, `.cl`, `.py`, `.h`, `.hpp`)
- `custom_prompt` (str, optional): Natural language question to guide LLM analysis
- `enable_llm` (bool): Enable LLM-powered explanation of the profiling plan (default: False)
- `llm_provider` (str, optional): LLM provider ("anthropic" or "openai")
- `llm_api_key` (str, optional): API key (or use environment variable)
- `verbose` (bool): Enable verbose logging (default: False)

**Returns:**

- `SourceAnalysisResult`: Profiling plan with detected kernels, patterns, risk areas, and suggested commands

**Raises:**

- `SourceDirectoryNotFoundError`: Source directory doesn't exist
- `SourceAnalysisError`: Error during source scanning

**Example:**

```python
from rocinsight.ai_analysis import analyze_source
from pathlib import Path

result = analyze_source(Path("./my_app/src"))
print(f"Programming model: {result.programming_model}")
print(f"Kernels found: {result.kernel_count}")
print(f"Suggested first command:\n  {result.suggested_first_command}")

for rec in result.recommendations:
    print(f"[{rec['priority']}] {rec['category']}: {rec['issue']}")
```

**CLI equivalent:**

```bash
rocinsight analyze --source-dir ./my_app/src
rocinsight analyze --source-dir ./my_app/src --format json -d ./out -o plan  # → plan.json

# Combined with trace database
rocinsight analyze -i output.db --source-dir ./my_app/src
```

---

### Recommendation Deduplication

The engine automatically detects what was already collected in the profiled run and
suppresses redundant suggestions:

| Already in database | Commands suppressed |
|---|---|
| `kernels` rows | `rocprofv3 --kernel-trace` |
| `memory_copies` rows | `rocprofv3 --memory-copy-trace` |
| `kernels` + `regions` rows | All `--sys-trace`-equivalent flags |
| `pmc_events` counter `X` | `--pmc X` in any `rocprofv3` command |

**PMC counter example**: if the trace was collected with
`--pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES`, a "Low occupancy" recommendation that
would have suggested `--pmc SQ_WAVES SQ_WAVE_CYCLES TA_TA_BUSY` will be trimmed to
`--pmc SQ_WAVE_CYCLES TA_TA_BUSY` (only the uncollected counters). If *all* suggested
counters are already present the entire `rocprofv3` command is dropped.

`rocprof-compute` commands are **never** dropped — they always represent new deep
hardware counter analysis beyond what `rocprofv3` captures.

---

## Data Classes

### `AnalysisResult`

Main result object containing all analysis data.

**Attributes:**

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
    llm_enhanced_explanation: Optional[str]  # Only if enable_llm=True
```

**Methods:**

- `to_dict() -> Dict[str, Any]`: Convert to dictionary
- `to_json(indent: int = 2) -> str`: Serialize to JSON
- `to_text() -> str`: Generate plain text report
- `to_markdown() -> str`: Generate markdown report
- `to_webview() -> str`: Generate self-contained interactive HTML report

**Example:**

```python
result = analyze_database(Path("output.db"))

# Convert to different formats
json_str = result.to_json()
text_report = result.to_text()
markdown_report = result.to_markdown()

# Access structured data
print(f"Kernel time: {result.execution_breakdown.kernel_time_pct:.1f}%")
print(f"Primary bottleneck: {result.summary.primary_bottleneck}")
```

---

### `Recommendation`

Single optimization recommendation.

```python
@dataclass
class Recommendation:
    id: str
    priority: str  # "high", "medium", "low"
    category: str  # "memory", "compute", "occupancy", etc.
    title: str
    description: str
    estimated_impact: str
    next_steps: List[str]
```

**Example:**

```python
for rec in result.recommendations.high_priority:
    print(f"ID: {rec.id}")
    print(f"Title: {rec.title}")
    print(f"Category: {rec.category}")
    print(f"Impact: {rec.estimated_impact}")
    print("Next steps:")
    for step in rec.next_steps:
        print(f"  - {step}")
```

---

### `SourceAnalysisResult`

Tier 0 analysis result from static source code scanning (returned by `analyze_source()`).

**Attributes:**

```python
@dataclass
class SourceAnalysisResult:
    source_dir: str
    analysis_timestamp: str
    programming_model: str  # "HIP", "HIP+ROCm_Libraries", "OpenCL", "PyTorch_HIP", etc.

    files_scanned: int
    files_skipped: int

    detected_kernels: List[Dict]   # {name, file, line, launch_type}
    kernel_count: int

    detected_patterns: List[Dict]  # {pattern_id, severity, category, description, count, locations}
    risk_areas: List[str]

    already_instrumented: bool     # True if ROCTx markers detected
    roctx_marker_count: int

    recommendations: List[Dict]    # Same structure as generate_recommendations() output
    suggested_counters: List[str]  # Recommended --pmc counters for this codebase
    suggested_first_command: str   # First rocprofv3 command to run

    llm_explanation: Optional[str]  # Only if enable_llm=True
```

**Example:**

```python
result = analyze_source(Path("./my_app"))

# Programming model detection
print(result.programming_model)    # "HIP+ROCm_Libraries"

# Discovered kernels
for k in result.detected_kernels:
    print(f"  {k['name']} in {k['file']}:{k['line']}")

# Risk patterns
for p in result.detected_patterns:
    print(f"[{p['severity'].upper()}] {p['category']}: {p['description']}")

# Suggested profiling workflow
print(result.suggested_first_command)
# e.g.: rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -- ./app
```

---

### Other Data Classes

- `AnalysisMetadata`: Metadata about analysis (timestamps, versions, etc.)
- `ProfilingInfo`: Profiling session info (duration, mode, GPUs)
- `AnalysisSummary`: High-level summary (assessment, bottleneck, findings)
- `ExecutionBreakdown`: Time distribution (kernel, memcpy, API overhead)
- `RecommendationSet`: Prioritized recommendations (high/medium/low)
- `AnalysisWarning`: Warning messages

See inline docstrings for complete documentation.

---

## Output Formats

### Python Object (Default)

Returns `AnalysisResult` dataclass with full type safety.

```python
result = analyze_database(Path("output.db"))
print(result.summary.overall_assessment)
```

### JSON

Machine-readable structured data. Output file extension: `.json`.

```python
from rocinsight.ai_analysis import analyze_database, OutputFormat

result = analyze_database(
    Path("output.db"),
    output_format=OutputFormat.JSON
)

json_str = result.to_json(indent=2)
```

**JSON Output conforms to `analysis-output.schema.json` (v0.1.0):**

```json
{
  "schema_version": "0.1.0",
  "metadata": {
    "rocpd_version": "6.3.0",
    "analysis_version": "0.1.0",
    "database_file": "/path/to/output.db",
    "analysis_timestamp": "2026-02-07T14:30:00Z"
  },
  "execution_breakdown": {
    "kernel_time_pct": 40.0,
    "memcpy_time_pct": 55.0,
    "api_overhead_pct": 5.0,
    "idle_time_pct": 0.0,
    "total_runtime_ns": 5000000000
  },
  "hotspots": [
    {
      "rank": 1,
      "name": "conv2d_kernel",
      "calls": 100,
      "total_duration_ns": 2000000000,
      "avg_duration_ns": 20000000,
      "pct_of_total": 40.0
    }
  ],
  "memory_analysis": { ... },
  "hardware_counters": { ... },
  "recommendations": [
    {
      "priority": "HIGH",
      "category": "Low Occupancy",
      "issue": "Average wave occupancy is low",
      "suggestion": "Increase occupancy by reducing VGPR usage",
      "estimated_impact": "15-20% performance improvement",
      "actions": ["Use rocprof-compute to measure occupancy", ...],
      "commands": [...]
    }
  ],
  "warnings": [...]
}
```

> See `docs/analysis-output.schema.json` for the normative schema definition and
> `docs/SCHEMA_CHANGELOG.md` for version history.

### Text

Human-readable plain text report. Output file extension: `.txt`.

```python
result = analyze_database(Path("output.db"))
text_report = result.to_text()
print(text_report)
```

### Markdown

Markdown-formatted report with syntax highlighting. Output file extension: `.md`.

```python
result = analyze_database(Path("output.db"))
markdown_report = result.to_markdown()
Path("report.md").write_text(markdown_report)
```

### Webview (Interactive HTML)

Self-contained single-file HTML report with light/dark theme, sortable tables, interactive
recommendation cards, status-colored KPI cards, and SVG performance gauges. No external
dependencies — works fully offline. Output file extension: `.html`.

```python
result = analyze_database(Path("output.db"))
html_report = result.to_webview()
Path("report.html").write_text(html_report)
```

**CLI usage:**

```bash
# Produces output/analysis.html automatically
rocinsight analyze -i output.db --format webview -d ./output -o analysis
```

**Features of the HTML report:**

- **Light/Dark theme toggle**: Persisted in `localStorage`; defaults to AMD dark. Header
  always uses AMD gradient branding regardless of active theme.
- **Status summary badges**: Critical/Warning/Low/Info recommendation counts shown in the
  sticky header — key issues visible without scrolling.
- **Metric pills row**: Runtime (ms), kernel dispatch count, analysis tier, generation
  timestamp, and DB file path in a compact row below the header.
- **Status-colored KPI cards**: Kernel %, bottleneck type, total runtime, and tier cards
  with colored top border (green/amber/red) reflecting health status.
- **Priority icons on recommendations**: 🔴 HIGH, 🟠 MEDIUM, 🟡 LOW, ℹ INFO icons on each card.
- **Overview panel**: Assessment text (blockquote style), status KPI grid, key findings list.
- **Execution breakdown**: Gradient segment bars + grid-aligned legend rows.
- **Recommendations**: Collapsible cards color-coded by priority (HIGH auto-expanded);
  one-click copy of profiling commands; section-level Critical/Warning count badges.
- **Hotspot table**: Sortable by any column; rows with >20% of total time highlighted.
- **Memory transfers**: Per-direction table (H2D, D2H, D2D, P2P).
- **Hardware counters**: GPU utilization and wave occupancy gauges (Tier 2); gauges have
  background fill and hover border effect.
- **FAB scroll-to-top**: Floating action button appears after scrolling 250 px.
- **Staggered animations**: Section cards fade in with `@keyframes fadeInUp` on load.
- **Embedded data**: Full JSON payload included for programmatic inspection.
- **Hover tooltips**: Every graph, gauge, bar, table column, and counter row shows a
  floating tooltip on hover explaining what the metric means, why it matters, good/bad
  thresholds, and how to address issues. Coverage includes:
  - *Gauges*: counter formula (e.g. `GRBM_GUI_ACTIVE ÷ GRBM_COUNT`), target thresholds,
    current status assessment
  - *Breakdown bars*: what each category measures, optimization guidance
  - *Overview stats*: per-bottleneck type explanation with specific fix advice,
    Tier 1 vs Tier 2 distinction with upgrade command
  - *Hotspot columns*: semantics of Calls, Total/Avg/Min time, % Total
  - *Memory directions*: H2D/D2H/D2D/P2P with PCIe vs HBM bandwidth context
  - *Counter rows*: educational content for 20+ known AMD GPU counters
    (GRBM_*, SQ_*, TCP/TCC cache, FETCH_SIZE, WRITE_SIZE, etc.);
    unknown counters receive a generic fallback message

---

## LLM Enhancement

### Overview

LLM enhancement provides natural language explanations of performance data. It's **optional** and **privacy-focused**.

### How It Works

1. **Local analysis runs first** (always)
2. **Data is sanitized** (kernel names → [KERNEL_1], grid sizes → [REDACTED])
3. **Reference guide loaded** (the "fence" - defines analysis rules)
4. **LLM called with sanitized data + reference guide**
5. **Natural language explanation returned**

### Enabling LLM Enhancement

**Option 1: Environment Variable**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
from rocinsight.ai_analysis import analyze_database

result = analyze_database(
    Path("output.db"),
    enable_llm=True,
    llm_provider="anthropic"
)
```

**Option 2: Pass API Key Directly**

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
  - Environment variable: `ANTHROPIC_API_KEY`
  - Default model: `claude-sonnet-4-20250514`

- **OpenAI GPT**
  - Provider: `"openai"`
  - Environment variable: `OPENAI_API_KEY`
  - Default model: `gpt-4-turbo-preview`
  - **Model compatibility**: newer models (gpt-5, o1, o3, gpt-4o-2024-11-20+) require
    `max_completion_tokens` instead of `max_tokens`. This is handled automatically —
    `max_completion_tokens` is tried first and falls back to `max_tokens` if needed.

- **Private/enterprise server** (any OpenAI-compatible endpoint)
  - Provider: `"private"` (`--llm private`)
  - Required env var: `ROCINSIGHT_LLM_PRIVATE_URL` — base URL (e.g. `https://llm-api.example.com/OpenAI`)
  - Required: `ROCINSIGHT_LLM_PRIVATE_MODEL` or `--llm-private-model`
  - Optional: `ROCINSIGHT_LLM_PRIVATE_API_KEY` (default: `"dummy"` for header-authenticated servers)
  - Optional: `ROCINSIGHT_LLM_PRIVATE_HEADERS` — JSON object of extra request headers;
    must be a JSON object (`{...}`), not an array or scalar — a `ValueError` is raised
    if the parsed value is not a dict; the `user` header is auto-set to `os.getlogin()`
    unless already provided
  - Optional: `ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL=0` — disable SSL certificate verification (requires `httpx`)

  ```bash
  export ROCINSIGHT_LLM_PRIVATE_URL="https://llm-api.example.com/OpenAI"
  export ROCINSIGHT_LLM_PRIVATE_HEADERS='{"Ocp-Apim-Subscription-Key": "abc123", "api-version": "preview"}'
  rocinsight analyze -i output.db --llm private --llm-private-model gpt-4o
  ```

- **Local Ollama**
  - Provider: `--llm-local ollama`
  - Env var: `ROCINSIGHT_LLM_LOCAL_URL` (default: `http://localhost:11434/v1`)
  - Env var: `ROCINSIGHT_LLM_LOCAL_MODEL` (default: `codellama:13b`)

**Override the model at runtime** (anthropic/openai providers):

```bash
export ROCINSIGHT_LLM_MODEL="claude-opus-4-6"   # Use a different Anthropic model
export ROCINSIGHT_LLM_MODEL="gpt-4o"            # Use a different OpenAI model
```

### Custom Prompts

Guide the LLM with specific questions:

```python
result = analyze_database(
    Path("output.db"),
    enable_llm=True,
    llm_provider="anthropic",
    custom_prompt="Why is my convolution kernel slow? Focus on memory access patterns."
)

print(result.llm_enhanced_explanation)
```

### Data Sanitization

When LLM mode is enabled, sensitive data is automatically redacted:

| Data Type | Original | Sanitized |
|-----------|----------|-----------|
| Kernel names | `conv2d_forward_kernel` | `[KERNEL_1]` |
| Grid sizes | `[256, 256, 1]` | `[GRID_SIZE]` |
| Workgroup sizes | `[256, 1, 1]` | `[WORKGROUP_SIZE]` |
| File paths | `/home/user/app.cpp` | `[REDACTED]` |

**Preserved Data** (aggregated/classified):
- Bottleneck classifications (compute-bound, memory-bound)
- Aggregated metrics (time percentages, utilization %)
- GPU architecture (gfx908, gfx90a, gfx942, gfx950, gfx1030, gfx1100)

---

## Error Handling

### Exception Hierarchy

```python
AnalysisError (base)
├── DatabaseNotFoundError
├── DatabaseCorruptedError
├── MissingDataError
├── UnsupportedGPUError
├── LLMAuthenticationError
├── LLMRateLimitError
├── ReferenceGuideNotFoundError
├── SourceDirectoryNotFoundError   # analyze_source(): directory doesn't exist
└── SourceAnalysisError            # analyze_source(): error during scanning
```

### Example Error Handling

```python
from rocinsight.ai_analysis import (
    analyze_database,
    DatabaseNotFoundError,
    MissingDataError,
    LLMAuthenticationError
)
from pathlib import Path

try:
    result = analyze_database(
        Path("output.db"),
        enable_llm=True,
        llm_provider="anthropic"
    )

except DatabaseNotFoundError as e:
    print(f"Database not found: {e}")

except MissingDataError as e:
    print(f"Missing data: {e}")
    print(f"Missing tables: {e.missing_tables}")
    print("Suggestion: Collect additional profiling data")

except LLMAuthenticationError as e:
    print(f"LLM authentication failed: {e}")
    print("Check your API key and environment variables")

except Exception as e:
    print(f"Unexpected error: {e}")
```

### Graceful Degradation

**Authentication and rate-limit errors propagate** — if `enable_llm=True` and your key is
invalid or exhausted, `LLMAuthenticationError` / `LLMRateLimitError` will be raised so you
know immediately rather than silently getting local-only results.

Other transient LLM failures (network timeouts, unexpected API errors) produce a warning
and fall back to local-only results without raising:

```python
try:
    result = analyze_database(
        Path("output.db"),
        enable_llm=True,
        llm_provider="anthropic"
    )
except LLMAuthenticationError:
    print("Invalid API key — check ANTHROPIC_API_KEY")
    raise

# If a transient error occurred, llm_enhanced_explanation will be None
if result.llm_enhanced_explanation:
    print("LLM enhancement available")
else:
    print("Local-only analysis (LLM enhancement failed or disabled)")

# Check warnings for details on any transient failure
for warning in result.warnings:
    print(f"⚠️  {warning.message}")
```

---

## Integration Examples

### Optiq Integration

```python
# Optiq UI integration example
from rocinsight.ai_analysis import analyze_database
from pathlib import Path

def load_trace_with_ai_insights(trace_file_path: str):
    """
    Optiq function to load trace and get AI insights.
    """
    result = analyze_database(Path(trace_file_path))

    # Extract insights for UI
    insights = {
        "summary": result.summary.overall_assessment,
        "bottleneck": result.summary.primary_bottleneck,
        "confidence": result.summary.confidence,
        "top_recommendations": [
            {
                "title": rec.title,
                "description": rec.description,
                "impact": rec.estimated_impact,
                "priority": rec.priority
            }
            for rec in result.recommendations.high_priority[:3]
        ],
        "execution_breakdown": {
            "kernel_pct": result.execution_breakdown.kernel_time_pct,
            "memcpy_pct": result.execution_breakdown.memcpy_time_pct,
            "overhead_pct": result.execution_breakdown.api_overhead_pct
        }
    }

    return insights

# Usage in Optiq
insights = load_trace_with_ai_insights("/path/to/output.db")
display_ai_panel(insights)
```

### Automated Analysis Pipeline

```python
from rocinsight.ai_analysis import analyze_database, get_recommendations
from pathlib import Path
import sys

def automated_analysis_pipeline(trace_files: List[Path]):
    """
    Analyze multiple trace files and generate reports.
    """
    for trace_file in trace_files:
        print(f"Analyzing {trace_file}...")

        try:
            # Analyze
            result = analyze_database(
                trace_file,
                enable_llm=True,
                llm_provider="anthropic"
            )

            # Generate markdown report
            report_path = trace_file.with_suffix(".md")
            report_path.write_text(result.to_markdown())
            print(f"  ✅ Report saved: {report_path}")

            # Check for high-priority issues
            high_priority = result.recommendations.high_priority
            if high_priority:
                print(f"  🔴 {len(high_priority)} high-priority issues found")
                for rec in high_priority:
                    print(f"     - {rec.title}")

        except Exception as e:
            print(f"  ❌ Analysis failed: {e}")

# Run pipeline
trace_files = list(Path("./traces").glob("*.db"))
automated_analysis_pipeline(trace_files)
```

### Batch Comparison

```python
from rocinsight.ai_analysis import analyze_database
from pathlib import Path
import pandas as pd

def compare_traces(baseline_path: Path, optimized_path: Path):
    """
    Compare baseline vs optimized traces.
    """
    baseline = analyze_database(baseline_path)
    optimized = analyze_database(optimized_path)

    # Build comparison dataframe
    comparison = pd.DataFrame({
        "Metric": [
            "Kernel Time %",
            "Memory Copy %",
            "API Overhead %",
            "Primary Bottleneck",
            "Confidence"
        ],
        "Baseline": [
            f"{baseline.execution_breakdown.kernel_time_pct:.1f}%",
            f"{baseline.execution_breakdown.memcpy_time_pct:.1f}%",
            f"{baseline.execution_breakdown.api_overhead_pct:.1f}%",
            baseline.summary.primary_bottleneck,
            f"{baseline.summary.confidence:.0%}"
        ],
        "Optimized": [
            f"{optimized.execution_breakdown.kernel_time_pct:.1f}%",
            f"{optimized.execution_breakdown.memcpy_time_pct:.1f}%",
            f"{optimized.execution_breakdown.api_overhead_pct:.1f}%",
            optimized.summary.primary_bottleneck,
            f"{optimized.summary.confidence:.0%}"
        ]
    })

    print(comparison.to_markdown(index=False))

# Usage
compare_traces(Path("baseline.db"), Path("optimized.db"))
```

---

## See Also

- [LLM Reference Guide Documentation](LLM_REFERENCE_GUIDE.md) - How to customize LLM behavior
- [CLI Documentation](../README.md) - Using `rocinsight analyze` command
- [rocprofiler-sdk Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/)

---

### `LLMConversation` — Persistent Multi-Turn Streaming Session

`LLMConversation` provides a stateful multi-turn LLM session with streaming output,
automatic compaction, and disk archiving. It is used internally by `InteractiveSession`
and is also available as a public API for custom workflows.

```python
from rocinsight.ai_analysis import LLMConversation

conv = LLMConversation(
    provider="anthropic",      # "anthropic" | "openai" | "private" | "local"
    api_key=None,              # or pass directly; falls back to env vars
    model=None,                # or override default model
    compact_every=10,          # compact history every N turns (default 10)
    keep_recent_turns=6,       # keep this many turns after compaction
    history_path=None,         # optional Path for JSONL disk archive
)

# Set the system prompt once (include the reference guide / "fence" here)
from rocinsight.ai_analysis import load_reference_guide
conv.initialize("You are an AMD GPU expert.\n\n" + load_reference_guide())

# Stream a response token-by-token
response = conv.send(
    "What is the bottleneck in this trace?",
    on_token=lambda t: print(t, end="", flush=True),
)

# Serialize / restore across sessions
state = conv.to_dict()                              # does NOT include api_key
conv2 = LLMConversation.from_dict(state, api_key="sk-ant-...")
```

**Constructor parameters:**

| Parameter | Default | Description |
|---|---|---|
| `provider` | — | `"anthropic"`, `"openai"`, `"private"`, or `"local"` |
| `api_key` | `None` | API key; falls back to `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `ROCINSIGHT_LLM_PRIVATE_API_KEY` |
| `model` | `None` | Model override; falls back to `ROCINSIGHT_LLM_MODEL` then built-in default |
| `compact_every` | `10` | Trigger LLM-based history compaction every N turns |
| `keep_recent_turns` | `6` | Number of recent turns preserved verbatim after compaction |
| `history_path` | `None` | JSONL file path for append-only message archive |

**Methods:**

- `initialize(system_prompt: str)` — Set system prompt (call once before `send()`)
- `send(user_message, *, max_tokens=4096, on_token=None) -> str` — Append user turn, stream response
- `to_dict() -> dict` — Serialize state (api_key excluded)
- `from_dict(d, *, api_key=None, model=None) -> LLMConversation` — Restore from serialized state

**Properties:** `turn_count: int`, `messages: List[dict]`

---

### `load_reference_guide()` — Load the LLM Fence

Returns the full content of the LLM reference guide (the "fence") as a string.
Useful when building a custom system prompt for `LLMConversation.initialize()`.

```python
from rocinsight.ai_analysis import load_reference_guide

guide = load_reference_guide()
# guide is the full markdown text of share/llm-reference-guide.md

conv.initialize("You are an expert AMD GPU engineer.\n\n" + guide)
```

The guide is loaded from (in order):
1. `ROCINSIGHT_LLM_REFERENCE_GUIDE` environment variable path
2. Module-relative `share/llm-reference-guide.md`
3. `/opt/rocm/share/rocprofiler-sdk/llm-reference-guide.md`

---

### Context-Aware LLM Guide Loading

`LLMAnalyzer` accepts an optional `AnalysisContext` to reduce the reference guide
tokens sent per call. Build the context from already-computed analysis results:

```python
from rocinsight.ai_analysis import AnalysisContext
from rocinsight.ai_analysis.llm_analyzer import LLMAnalyzer

ctx = AnalysisContext(
    tier=2,                        # 0=source-only, 1=trace, 2=counters
    has_counters=True,
    bottleneck_type="compute",     # triggers compiler section
    custom_prompt="why is my kernel slow?",
)

analyzer = LLMAnalyzer(provider="anthropic", api_key="...", verbose=True)
result = analyzer.analyze_with_llm(data, context=ctx)
```

When `context=None` (default), the full guide is used — backward compatible.

**`_select_tags` behavior (which guide sections are included):**

| Scenario | Tags included | Notes |
|---|---|---|
| Tier 1, clear bottleneck (e.g. `memory_transfer`) | `{"always"}` | Bottleneck known; LLM only needs formatting rules. ~40% token savings vs. old behavior. |
| Tier 1, unclear bottleneck (`mixed` or `None`) | `{"always", "tier1"}` | Same as before |
| Tier 2 (counters present) | `{"always", "tier2"}` | Drops `tier1`; was `{"always", "tier1", "tier2"}` |
| Tier 0 (source-only) | `{"always", "tier1", "source"}` | Unchanged |
| `compiler` tag | Added when bottleneck is `compute` OR prompt mentions compiler/flag/build/register | No longer added for `memory` bottleneck |

**Bottleneck-aware `_format_data_for_llm`** (data trimming sent to LLM):

| Bottleneck | Kernels sent | Counter fields | Memory ops section |
|---|---|---|---|
| `memory_transfer` | Top 3, no counter fields | omitted | included (full) |
| `compute` | Top 3, with counter fields | included | omitted |
| `latency` / `api` | Top 3, no counter fields | omitted | omitted |
| `mixed` or `None` | Top 5, all fields | included | included |

Token savings by scenario:
- Tier 1 trace-only: ~47% fewer tokens
- Tier 1 with clear bottleneck: ~40% additional savings on guide sections
- Tier 0 source-only: ~51% fewer tokens
- Tier 2 with latency bottleneck: ~18% fewer tokens

See `docs/LLM_GUIDE_SECTIONS.md` for the full tag vocabulary and how to add
new sections or tags.

---

## Support

For issues, questions, or feature requests:
- File an issue on GitHub
- See [CONTRIBUTING.md](../CONTRIBUTING.md)
- ROCm documentation: https://rocm.docs.amd.com/

---

## Bug Fixes & Behavioral Changes

This section documents behavioral changes made during code review that affect
how callers interact with the API. Changes are grouped by category.

### LLM Layer

**`LLMAnalyzer()` construction no longer raises `LLMAuthenticationError`**

Previously, constructing `LLMAnalyzer(provider="anthropic")` without setting
`ANTHROPIC_API_KEY` would raise `LLMAuthenticationError` immediately. This blocked
use cases where the analyzer is constructed ahead of time and the API key is
supplied later (e.g., via a configuration reload).

The key validation is now **deferred** — `LLMAuthenticationError` is raised only
when an actual API call is made (`analyze_with_llm()`, `_call_anthropic()`, etc.).
Construction always succeeds as long as `provider` is valid.

```python
# This now works even without an API key set
from rocinsight.ai_analysis.llm_analyzer import LLMAnalyzer
analyzer = LLMAnalyzer(provider="anthropic")  # no longer raises

# The error fires here instead, when the call is actually made
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  # set key before calling
result = analyzer.analyze_with_llm(data)
```

**`LLMAnalyzer(model=...)` is now honored**

Previously, the `model` parameter was stored but the `ROCINSIGHT_LLM_MODEL` environment
variable was checked first at call time, silently overriding any explicit `model=`
argument. The priority is now:

1. `model=` constructor argument (highest priority)
2. `ROCINSIGHT_LLM_MODEL` environment variable
3. Built-in default (`DEFAULT_ANTHROPIC_MODEL` or `DEFAULT_OPENAI_MODEL`)

**`analyze_source()` now passes `AnalysisContext(tier=0)` to the LLM automatically**

When `enable_llm=True`, `analyze_source()` constructs an `AnalysisContext(tier=0,
custom_prompt=...)` and passes it to `analyze_source_with_llm()`. This ensures the
LLM reference guide is filtered to Tier 0-relevant sections (reducing token cost by
~51%) and that compiler optimization guidance is included.

Callers who create `LLMAnalyzer` directly and call `analyze_source_with_llm()`
should also pass `context=AnalysisContext(tier=0)` for the same benefit.

**Timeout parameter added to all LLM API calls**

All Anthropic and OpenAI API calls now include `timeout=120` (seconds). Previously,
LLM calls could hang indefinitely on slow or unavailable network connections. If the
call takes longer than 120 seconds a network timeout exception is raised and wrapped
as a non-fatal warning (local analysis continues).

**LLM truncation retry for reasoning models (`_call_openai`)**

Reasoning models (gpt-5, o1, o3) consume `max_completion_tokens=4096` on internal
thinking tokens, leaving no budget for the actual output. When `_call_openai` receives
empty content with `finish_reason=="length"`, `analyze_with_llm` automatically retries:

1. System prompt filtered to only `<!-- rocinsight-context: always -->` sections
2. `max_completion_tokens` raised to `16384` to give reasoning models room for thinking + output

If the retry also produces empty content, a clean user-facing error is raised — no
internal sentinel values leak to callers. If content is non-empty but truncated
(`finish_reason=="length"`), the partial content is returned with a warning attached
to `result.warnings` rather than raising an exception.

### Output & Serialization

**`-d <dir>` without `-o <name>` auto-generates output filename**

When a directory is specified with `-d` but no explicit `-o` name is given, the output
filename is now derived from the database basename:

```bash
# merged_opt.db → merged_opt.html  (with --format webview)
rocinsight analyze -i merged_opt.db --format webview -d ./reports
```

Previously the analysis was printed to the terminal even when `-d` was specified.
`-o` can still be used to override the filename explicitly.

**`AnalysisResult.to_json()` now raises `RuntimeError` when `_raw` is absent**

Previously, calling `to_json()` on an `AnalysisResult` constructed manually (not via
`analyze_database()`) would silently return non-schema-conformant JSON — a plain
`asdict()` serialization missing `schema_version`, `hotspots`, and other required
fields.

It now raises `RuntimeError("Raw analysis data not available. ...")` immediately,
making the problem visible. Use `to_dict()` for non-schema-conformant dict output,
or use `analyze_database()` (which populates `_raw`) to get schema-conformant JSON.

```python
# Manual construction — to_json() now raises:
result = AnalysisResult(...)
result.to_json()          # raises RuntimeError — use to_dict() instead
result.to_dict()          # works — returns plain asdict() dict

# Via analyze_database() — to_json() works:
result = analyze_database(Path("output.db"))
result.to_json()          # works — schema-conformant, schema_version="0.1.0"
```

**`analyze_memory_copies()` bandwidth now uses actual transfer sizes**

Previously the `size` column in the `memory_copies` table was not reliably
populated and bandwidth calculations returned 0. The column is now read and
`bandwidth_bytes_per_sec` (and `bandwidth_gbps`) are computed from real transfer
sizes when available. The "Low memory bandwidth" recommendation (< 10 GB/s threshold)
can now fire based on actual measurements.

### Analysis Correctness

**`overhead_percent` is now guaranteed to be ≥ 0**

In some trace databases where kernel + memcpy time slightly exceeds the computed
total runtime (due to timestamp rounding), `overhead_percent` could be negative.
`compute_time_breakdown()` now applies `max(0.0, raw_overhead_pct)` before
returning the result. The field is always non-negative in the output.

**Bottleneck classification no longer triggers `compute` from `has_counters` alone**

Previously, the `_build_summary()` bottleneck classifier in `api.py` could produce
`primary_bottleneck="compute"` based on `kernel_pct > 70 AND has_counters=True`,
even when `kernel_pct` was well below 70%. The condition now uses the correct
threshold check: `kernel_pct > 70` is evaluated first, then `has_counters` is used
only to raise the confidence from 0.60 to 0.80 — not to change the bottleneck type.

**`analyze_source_code()` raises `SourceDirectoryNotFoundError` for missing directories**

The `analyze_source_code()` function in `analyze.py` (CLI path) now raises
`SourceDirectoryNotFoundError` (not a generic `Exception`) when the `source_dir`
argument does not exist or is not a directory. This matches the behavior of the
Python API's `analyze_source()`.

### Interactive Session (LLM Providers)

**`"private"` provider now correctly routed in `_apply_suggestions_via_llm` and `_llm_rewrite_file`**

Previously, both `InteractiveSession._apply_suggestions_via_llm` and
`WorkflowSession._llm_rewrite_file` dispatched any unrecognized provider to
`_call_local()` (Ollama). This caused the `"private"` provider to attempt a connection
to `http://localhost:11434/v1` and fail with a connection error instead of calling the
configured enterprise server.

Both methods now explicitly handle `"private"` by routing to `_call_private()`.

**`InteractiveSession` uses `LLMConversation` for persistent multi-turn context**

The previous `SessionContext` dataclass (compact per-session summary: analyses, suggestions, commands)
has been replaced by a persistent `LLMConversation` object that holds the full message history.
All LLM calls within a session (`[o]`, `[a]` annotations, code rewrites) share the same conversation
so the LLM accumulates full context rather than receiving a condensed summary block.

Key behavioral changes:
- History is compacted via `--llm-compact-every N` (default 10 turns) using an LLM-generated summary, not a rule-based snippet
- Source files are tracked in `_sent_source_files`; a file already sent in this session is not re-transmitted
- Conversation state (`conv.to_dict()`) is serialized into the session JSON on `[s]` save
- On `--resume-session`, the conversation is restored with `LLMConversation.from_dict()`

### Source Scanner

**`SourceAnalyzer` adds a truncation warning to `risk_areas` when `_MAX_FILES` is hit**

When the number of source files in the scanned directory exceeds `_MAX_FILES` (500),
scanning stops early. The scanner now appends a human-readable warning to
`plan.risk_areas` noting how many files were skipped and suggesting a more targeted
`--source-dir` path. Previously the truncation was silent.

```python
plan = SourceAnalyzer(Path("./huge_repo")).analyze()
# If > 500 files found:
assert any("truncat" in r.lower() for r in plan.risk_areas)
```

### WorkflowSession — Cycle Prevention and Tier 3 Escalation

**Collection fingerprint expanded to all trace flags**

The PMC-dedup logic that prevents infinite `[r] → re-profile → same INFO` loops now
fingerprints **all named trace collection flags** in addition to individual `--pmc`
counter names:

```
--sys-trace  --hip-trace  --kernel-trace  --memory-copy-trace  --hsa-trace  --stats
```

Previously only `--pmc` counters were tracked, causing the session to cycle between
sys-trace and counter-collection runs indefinitely.

**All-history comparison (not just last run)**

The dedup check now compares the suggested command's fingerprint against the **union**
of everything collected across all previous trace runs:

```python
already_fp = frozenset().union(*(
    _collection_fingerprint(t.command) for t in self._state.trace_history
))
if suggested_fp and suggested_fp.issubset(already_fp):
    ai_rec_cmd = None  # every suggested collection already performed
```

**Tier 3 escalation when Tier 1/2 exhausted**

When all Tier 1/2 data has been collected and there is nothing new to suggest, Phase 5
now shows a "go deeper" menu instead of just printing "stuck":

- TraceLens interval + kernel-category analysis: already embedded in the Phase 4 report.
- `[d]` builds a PC sampling command and sets it as the Phase 7 option `[3]`:
  ```
  ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1 rocprofv3 --pc-sampling \
    -d /tmp/rocpd_trace/run_<ts> -o results -- <app>
  ```

**ENV=VALUE command prefix support in Phase 3**

`_phase3_run_profiler` now strips leading `KEY=VALUE` tokens from the command string
and injects them into the subprocess environment via `env=` rather than `shell=True`:

```
# This works directly — ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1 is extracted
# and added to the child process env before rocprofv3 is exec'd.
ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1 rocprofv3 --pc-sampling ...
```

### WorkflowSession — AI Edit Revert

**`_revert_last_edit(failure_reason="")` helper**

Restores the most recently AI-modified file from its `.bak` backup and removes the
`_EditRecord` from `edit_history`. Accepts an optional `failure_reason` string.

When `failure_reason` is non-empty and an `LLMConversation` is active, two messages
are injected directly into the conversation history (a `user` message describing the
failure and an `assistant` acknowledgement):

```python
feedback = (
    f"IMPORTANT: The previous code edit to {file} was reverted "
    f"because it caused errors.\n\nFailure details:\n{failure_reason}\n\n"
    f"Do NOT suggest the same pattern again..."
)
conv._messages.append({"role": "user", "content": feedback})
conv._messages.append({"role": "assistant", "content": "Understood. ..."})
```

This teaches the LLM what failed without requiring a separate API call.

**Phase 3 (run profiler) — `[v]` revert on profiling failure**

When the profiling command exits non-zero and `edit_history` is non-empty, the retry
menu now includes `[v] Revert last AI edit and retry`. The exit code is passed as the
failure reason so the LLM conversation records it.

**Phase 6 (recompile wait) — accumulate and pass error text**

The recompile-wait loop accumulates all lines the user types as potential compilation
errors. When the user types `revert`/`undo`/`v`, the accumulated error lines are passed
to `_revert_last_edit(failure_reason=...)` so the LLM conversation receives the exact
compiler output. Example:

```
Changes applied. Please recompile your application.
Type 'done' when compiled, 'revert' to undo the AI edit,
'abort' to exit, or paste compilation errors.
> error: use of undeclared identifier '__builtin_amdgcn_sin'
  Error noted. Type 'done' when fixed or 'revert' to undo the edit.
> revert
  ✓ Reverted: inefficient_demo.cpp  (backup kept at inefficient_demo.cpp.bak)
```

### LLM Fence — Invalid HIP Intrinsics

**`__builtin_amdgcn_sin` / `__builtin_amdgcn_cos` added to the prohibited list**

The reference guide now explicitly bans these non-existent HIP device functions with a
`❌` rule. The `__builtin_amdgcn_*` namespace covers hardware-specific operations
(lane reads, DS swizzle) but **not** transcendental math. Suggesting them causes:

```
error: use of undeclared identifier '__builtin_amdgcn_sin'
```

The guide documents the correct HIP math API: use `sinf()`, `cosf()`, `sqrtf()`, etc.
— amdclang++ maps these to OCML hardware-optimized implementations automatically.


### WorkflowSession — Phase 1b Quick Workload Analysis

**New pre-Phase-2 step selects the best starter profiling command**

Before presenting the profiling command to the user in Phase 2, `WorkflowSession` now
runs `_phase1b_quick_workload_analysis()` which combines two analysis paths:

**1. App-command heuristics (`_classify_app_command`)**

Inspects the binary name and arguments to detect workload type:

| Detected workload | `workload_type` | Extra flags added |
|---|---|---|
| Python + ML framework (torch/jax/tf/paddle) | `python_ml` | `--hip-trace` |
| Python + LLM inference (vllm/llama/gpt/…) | `llm_inference` | `--hip-trace` |
| Python without ML framework | `python_generic` | `--hip-trace` |
| Compiled HIP/ROCm binary | `hip_compute` | none |
| MPI/Slurm launcher | `mpi_multi` | warning only |

Multi-process patterns (torchrun, DDP, DeepSpeed, NCCL) trigger a warning about
worker-process GPU kernel capture limitations regardless of workload type.

**2. Tier 0 source analysis**

If `--source-dir` paths are provided, `SourceAnalyzer.analyze()` is called on the
first path. The flags from `plan.suggested_first_command` (the highest-priority
recommendation) replace the heuristic flags. The `-d <dir>` and `-o <name>` components
are updated to a fresh timestamped directory before the command is shown.

**Precedence and fallback:**

```
Source analysis flags  >  Heuristic extra flags  >  default set
(if source dir given)     (always appended)         (--sys-trace --kernel-trace
                                                      --memory-copy-trace --stats)
```

**Return value:** The method returns the full suggested command string. `run()` falls
back to `_build_profiling_command()` (pure default) only if the method returns `None`,
which only happens if both paths raise exceptions.

### --resume-session Scope (InteractiveSession only)

`--resume-session` restores a previously saved `InteractiveSession` by ID. It applies
**only** to the menu-driven `InteractiveSession` (triggered by
`rocinsight analyze -i db.db --interactive` **without** a `"<app_command>"` string).

`WorkflowSession` (7-phase workflow, triggered by `rocinsight analyze --interactive "<app>"`)
starts fresh each invocation. It does not support session resume.

**How resume works:**

1. The session ID (format: `YYYY-MM-DD_HH-MM-SS_<source_dir_basename>`) is passed to
   `InteractiveSession(resume_session_id=...)`.
2. `_init_session(resume_id)` loads the session JSON from `~/.rocinsight/sessions/`.
3. `_restore_or_create_conv(loaded)` reconstructs the `LLMConversation` from the
   serialized `loaded.conversation` dict via `LLMConversation.from_dict()`.
4. `_sent_source_files` is restored from `loaded.sent_source_files`.

**Auto-detect (no `--resume-session` needed):** `_init_session` also calls
`self._store.find_by_source_dir(self._source_dir)` and, if matching sessions exist,
prompts the user to choose one. This means repeat invocations against the same
`--source-dir` will automatically offer resume without needing the session ID.

**Session ID discovery:**

```bash
ls ~/.rocinsight/sessions/*.json | xargs -I{} python3 -c \
    "import json; d=json.load(open('{}'));print(d['session_id'],'|',d['source_dir'])"
```

---

### WorkflowSession — Session Checkpoints

Each AI source-file edit creates a git-worktree checkpoint so the user can roll back to
any prior state and blacklist approaches that caused regressions.

#### Overview

```
Phase 6 AI edit
  └─► git commit all modified files
  └─► git update-ref refs/rocinsight/<session_id>/cp-N  (GC-pinned ref, not a branch)
  └─► git worktree add --detach ~/.rocinsight/sessions/<session_id>/cp-N
  └─► CheckpointRecord appended to WorkflowState.checkpoints
        ├── cp_id, commit_hash, ref_name, worktree_path
        ├── files_modified, file_snapshots (full file contents for offline restore)
        ├── run_index       ← set in Phase 3 after profiling succeeds
        ├── performance_delta_pct  ← set in Phase 4 after analysis history appended
        └── blacklisted, blacklist_category, blacklist_description
```

When the session exits (normally or via Ctrl+C), `_teardown_checkpoints()` removes all
worktrees. Refs (`refs/rocinsight/…`) are kept so the commits survive GC until the user
explicitly runs a cleanup command.

#### Dataclasses

**`CheckpointRecord`** (in `interactive.py`):

| Field | Type | Description |
|---|---|---|
| `cp_id` | `int` | Sequential checkpoint index (0-based) |
| `commit_hash` | `str` | Full git commit SHA |
| `ref_name` | `str` | `refs/rocinsight/<session_id>/cp-<N>` |
| `worktree_path` | `str` | Absolute path to the detached worktree |
| `timestamp` | `str` | ISO-8601 timestamp |
| `files_modified` | `List[str]` | Repo-relative paths of files in this edit batch |
| `edit_summary` | `str` | First non-blank line of the LLM suggestion (≤80 chars) |
| `file_snapshots` | `Dict[str, str]` | Full file contents keyed by relative path |
| `run_index` | `Optional[int]` | Which trace run followed this edit (set in Phase 3) |
| `performance_delta_pct` | `Optional[float]` | Runtime change % vs prior run (set in Phase 4) |
| `blacklisted` | `bool` | Whether this approach has been blacklisted |
| `blacklist_category` | `str` | Equal to `edit_summary` (used for deduplication) |
| `blacklist_description` | `str` | Human-readable description injected into LLM prompt |

**`WorkflowState` additions:**

| Field | Type | Description |
|---|---|---|
| `repo_root` | `str` | Absolute path to git repo root (empty when no git) |
| `baseline_commit` | `str` | HEAD at session start — rollback target `cp_id=-1` |
| `checkpoints` | `List[CheckpointRecord]` | All checkpoints in this session |
| `active_checkpoint` | `Optional[int]` | Currently restored checkpoint (or `None`) |
| `blacklisted_approaches` | `List[str]` | Persistent list of blacklist descriptions; **not truncated by rollback** |

#### GitCheckpointManager

All git operations are isolated in `GitCheckpointManager`:

```python
gcm = GitCheckpointManager(repo_root="/path/to/repo", session_id="2026-03-13_myapp")

# Detect repo (static — does not require a known repo_root)
repo_root = GitCheckpointManager.detect_repo(cwd="/path/to/project")

# Core checkpoint operations
hash_ = gcm.commit_files(files=["src/kernel.cpp"], message="rocpd: checkpoint 0")
gcm.tag_checkpoint(commit_hash=hash_, cp_id=0)          # creates refs/rocinsight/.../cp-0
gcm.add_worktree(commit_hash=hash_, cp_id=0)             # git worktree add --detach
gcm.remove_worktree(worktree_path="/path/to/wt")

# Introspection
gcm.get_head()                                           # current HEAD SHA
gcm.files_in_commit(commit_hash)                        # list of relative paths
gcm.list_worktrees()                                     # all registered worktrees
gcm.restore_files_from_commit(commit_hash, files)        # git checkout <hash> -- <files>
```

`commit_files` uses `-c user.email=rocinsight@local -c user.name=rocpd` overrides and
`--no-verify` to work in any git environment regardless of hooks or missing config.

#### Rollback

Triggered by `[b]` in the Phase 5 recommendations menu (shown only when checkpoints
exist). `_show_checkpoint_picker()` displays a table of all checkpoints with performance
delta and edit summary:

```
  Checkpoints
  ──────────────────────────────────────────────────────────
  [-1]  Baseline (no AI edits)
  [ 0]  Reduce memcpy by using zero-copy buffers             Run #1   -12.3%
  [ 1]  Optimize wave occupancy via LDS padding              Run #2   +4.1%  ← regression
  [ 2]  Unroll inner loop and vectorize memory accesses      Run #3   -8.7%
```

Regression checkpoints (+delta) are flagged and the user is prompted to blacklist them
before the rollback is applied. The blacklist description is appended to
`WorkflowState.blacklisted_approaches` (never truncated by rollback) so future LLM
calls avoid the same approach.

**Rollback strategy:**

1. **git fast path**: `git checkout <hash> -- <file>` for each file in the target
   checkpoint. Falls back to snapshot path on any `CheckpointError`.
2. **Snapshot fallback**: Writes `file_snapshots` contents directly. Works in any
   environment including those where git is not available post-session-start.
3. **Baseline rollback** (`cp_id = -1`): Restores to `baseline_commit` via git, or
   writes all accumulated snapshots in reverse order as a last resort.

After rollback, `WorkflowState.checkpoints` is truncated to `checkpoints[:target+1]`
and `_save_session()` is called unconditionally.

#### Blacklist Injection

When `_build_blacklist_block()` returns a non-empty string, it is prepended to the LLM
suggestion prompt in Phase 6 before `_llm_rewrite_file()` is called:

```
# Blacklisted approaches (do NOT use these):

- Reduce memcpy by using zero-copy buffers (caused +4.1% regression on run #2)
- ...
```

The blacklist is built from `WorkflowState.blacklisted_approaches` (persistent) so it
survives rollbacks that truncate the `checkpoints` list. Entries are deduplicated by
exact string match.

#### Session lifecycle

```
WorkflowSession.run()
  ├─ Phase 1: validate sources
  ├─ _init_checkpoints()       ← detect git, record baseline (dirty tree OK)
  ├─ _prune_stale_worktrees()  ← remove orphaned worktrees with no session JSON
  ├─ Phase 1b … Phase 7 loop
  └─ finally:
       _teardown_checkpoints() ← remove all worktrees (refs kept for GC protection)
       _save_session()
```

**Dirty working tree**: No issue. `commit_files` stages only the specific files modified
by each AI edit (`git add -- <file>`), so other in-progress changes in the working tree
are never touched or included in checkpoint commits.

**No-git graceful fallback**: When git is not detected or any checkpoint operation
fails, `self._gcm` is set to `None` and checkpoints are silently skipped. All other
session functionality continues normally.
