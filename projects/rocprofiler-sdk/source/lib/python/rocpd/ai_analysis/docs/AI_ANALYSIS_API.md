# rocpd AI Analysis Python API Documentation

**Version:** 0.2.0
**Module:** `rocpd.ai_analysis`

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

The rocpd AI Analysis API provides programmatic access to AI-powered GPU performance analysis. It's designed for integration with visualization tools (like Optiq), automated analysis pipelines, and custom workflows.

**Key Features:**

- ✅ **Local-first analysis** - Works offline, no API calls required
- ✅ **Tier 0 source analysis** - Scan source code without a trace database (`analyze_source()`)
- ✅ **Optional LLM enhancement** - Natural language explanations via Anthropic Claude or OpenAI GPT
- ✅ **Multiple output formats** - Python objects, JSON, text, markdown, webview (interactive HTML)
- ✅ **Privacy-focused** - Data sanitization for LLM mode
- ✅ **User-modifiable** - Customize LLM behavior via reference guide
- ✅ **Type-safe** - Dataclass-based API with type hints

---

## Installation

The AI analysis module is included with rocprofiler-sdk 6.3.0 or later.

```bash
# rocprofiler-sdk is typically installed at:
/opt/rocm/lib/python3.12/site-packages/rocpd/

# No additional installation needed for local-only analysis

# For LLM enhancement, install provider SDKs:
pip install anthropic  # For Anthropic Claude
pip install openai     # For OpenAI GPT
```

---

## Quick Start

### Basic Analysis (Local Mode)

```python
from rocpd.ai_analysis import analyze_database
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
from rocpd.ai_analysis import analyze_database
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
from rocpd.ai_analysis import analyze_database_to_json
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
from rocpd.ai_analysis import analyze_database
from pathlib import Path

result = analyze_database(Path("output.db"))
Path("analysis.html").write_text(result.to_webview())
# Open analysis.html in any browser - no server required
```

Or via CLI (file extension applied automatically):

```bash
rocpd analyze -i output.db --format webview -d ./output -o analysis
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

- `database_path` (Path): Path to rocpd database file (.rpd or .db)
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
from rocpd.ai_analysis import analyze_database, OutputFormat
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

- `database_path` (Path): Path to rocpd database file
- `output_json_path` (Path, optional): Save JSON to this file
- `**kwargs`: Additional arguments passed to `analyze_database()`

**Returns:**

- `str`: JSON string

**Example:**

```python
from rocpd.ai_analysis import analyze_database_to_json
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

- `database_path` (Path): Path to rocpd database file
- `priority_filter` (str, optional): Filter by priority ("high", "medium", "low")
- `category_filter` (str, optional): Filter by category ("memory", "compute", etc.)
- `**kwargs`: Additional arguments passed to `analyze_database()`

**Returns:**

- `List[Recommendation]`: Filtered recommendations

**Example:**

```python
from rocpd.ai_analysis import get_recommendations
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

- `database_path` (Path): Path to rocpd database file

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
from rocpd.ai_analysis import validate_database
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
from rocpd.ai_analysis import analyze_source
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
rocpd analyze --source-dir ./my_app/src
rocpd analyze --source-dir ./my_app/src --format json -d ./out -o plan  # → plan.json

# Combined with trace database
rocpd analyze -i output.db --source-dir ./my_app/src
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
from rocpd.ai_analysis import analyze_database, OutputFormat

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
rocpd analyze -i output.db --format webview -d ./output -o analysis
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
from rocpd.ai_analysis import analyze_database

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

**Override the model at runtime** (both providers):

```bash
export ROCPD_LLM_MODEL="claude-opus-4-6"   # Use a different Anthropic model
export ROCPD_LLM_MODEL="gpt-4o"            # Use a different OpenAI model
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
from rocpd.ai_analysis import (
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
from rocpd.ai_analysis import analyze_database
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
from rocpd.ai_analysis import analyze_database, get_recommendations
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
from rocpd.ai_analysis import analyze_database
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
- [CLI Documentation](../README.md) - Using `rocpd analyze` command
- [rocprofiler-sdk Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/)

---

### Context-Aware LLM Guide Loading

`LLMAnalyzer` accepts an optional `AnalysisContext` to reduce the reference guide
tokens sent per call. Build the context from already-computed analysis results:

```python
from rocpd.ai_analysis import AnalysisContext
from rocpd.ai_analysis.llm_analyzer import LLMAnalyzer

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

Token savings by scenario:
- Tier 1 trace-only: ~47% fewer tokens
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
from rocpd.ai_analysis.llm_analyzer import LLMAnalyzer
analyzer = LLMAnalyzer(provider="anthropic")  # no longer raises

# The error fires here instead, when the call is actually made
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  # set key before calling
result = analyzer.analyze_with_llm(data)
```

**`LLMAnalyzer(model=...)` is now honored**

Previously, the `model` parameter was stored but the `ROCPD_LLM_MODEL` environment
variable was checked first at call time, silently overriding any explicit `model=`
argument. The priority is now:

1. `model=` constructor argument (highest priority)
2. `ROCPD_LLM_MODEL` environment variable
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

### Output & Serialization

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
