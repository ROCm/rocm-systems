# rocpd AI Analysis Python API Documentation

**Version:** 0.1.0
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

---

## Overview

The rocpd AI Analysis API provides programmatic access to AI-powered GPU performance analysis. It's designed for integration with visualization tools (like Optiq), automated analysis pipelines, and custom workflows.

**Key Features:**

- ✅ **Local-first analysis** - Works offline, no API calls required
- ✅ **Optional LLM enhancement** - Natural language explanations via Anthropic Claude or OpenAI GPT
- ✅ **Multiple output formats** - Python objects, JSON, text, markdown, webview (interactive HTML)
- ✅ **Privacy-focused** - Data sanitization for LLM mode
- ✅ **User-modifiable** - Customize LLM behavior via reference guide
- ✅ **Type-safe** - Dataclass-based API with type hints

---

## Installation

The AI analysis module is included with rocprofiler-sdk 6.3.0+.

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

**JSON Schema:**

```json
{
  "metadata": {
    "rocpd_version": "6.3.0",
    "analysis_version": "0.1.0",
    "database_file": "/path/to/output.db",
    "analysis_timestamp": "2026-02-07T14:30:00Z",
    "custom_prompt": null
  },
  "profiling_info": {
    "total_duration_ns": 5000000000,
    "profiling_mode": "sys_trace_with_counters",
    "analysis_tier": 2
  },
  "summary": {
    "overall_assessment": "...",
    "primary_bottleneck": "memory_bound",
    "confidence": 0.85,
    "key_findings": [...]
  },
  "execution_breakdown": {
    "kernel_time_pct": 40.0,
    "memcpy_time_pct": 55.0,
    "api_overhead_pct": 5.0
  },
  "recommendations": {
    "high_priority": [...],
    "medium_priority": [...],
    "low_priority": [...]
  },
  "warnings": [...],
  "llm_enhanced_explanation": "..." // if enable_llm=True
}
```

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
  - Model: `claude-sonnet-4-20250514`

- **OpenAI GPT**
  - Provider: `"openai"`
  - Environment variable: `OPENAI_API_KEY`
  - Model: `gpt-4-turbo-preview`

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
- GPU architecture (gfx90a, gfx942)

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
└── ReferenceGuideNotFoundError
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

LLM enhancement failures don't prevent local analysis:

```python
result = analyze_database(
    Path("output.db"),
    enable_llm=True,
    llm_provider="anthropic"
)

# If LLM fails, you still get local analysis results
if result.llm_enhanced_explanation:
    print("LLM enhancement available")
else:
    print("Local-only analysis (LLM enhancement failed or disabled)")

# Check warnings
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

## Support

For issues, questions, or feature requests:
- File an issue on GitHub
- See [CONTRIBUTING.md](../CONTRIBUTING.md)
- ROCm documentation: https://rocm.docs.amd.com/
