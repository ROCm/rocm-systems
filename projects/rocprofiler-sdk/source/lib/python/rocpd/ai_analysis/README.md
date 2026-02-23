# rocpd AI Analysis Module

AI-powered GPU performance analysis for AMD ROCm profiling data.

## Overview

This module provides both CLI and Python API access to AI-powered analysis of GPU profiling traces. It analyzes rocpd database files and generates human-readable insights, bottleneck identification, and actionable optimization recommendations.

### Key Features

- **Local-first analysis** - Works offline, no API calls required by default
- **Optional LLM enhancement** - Natural language explanations via Anthropic Claude or OpenAI GPT
- **User-modifiable "fence"** - Customize LLM behavior by editing reference guide
- **Privacy-focused** - Data sanitization for LLM mode (kernel names, grid sizes redacted)
- **Multiple output formats** - Python objects, JSON, text, markdown, webview (interactive HTML)
- **Type-safe API** - Dataclass-based with type hints

## Quick Start

### CLI Usage

```bash
# Basic analysis (local mode)
rocpd analyze -i output.db

# With LLM enhancement
export ANTHROPIC_API_KEY="sk-ant-..."
rocpd analyze -i output.db --llm anthropic

# With custom prompt
rocpd analyze -i output.db --llm anthropic --prompt "Why is my matmul kernel slow?"

# JSON output (produces analysis.json)
rocpd analyze -i output.db --format json -d ./output -o analysis

# Markdown output (produces analysis.md)
rocpd analyze -i output.db --format markdown -d ./output -o analysis

# Interactive HTML webview (produces analysis.html)
rocpd analyze -i output.db --format webview -d ./output -o analysis
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
├── __init__.py              # Public API exports
├── api.py                   # Main API functions and data classes
├── llm_analyzer.py          # LLM integration with "fence" implementation
├── exceptions.py            # Exception classes
├── share/
│   └── llm-reference-guide.md  # LLM "fence" - user-modifiable reference guide
├── docs/
│   ├── AI_ANALYSIS_API.md      # API documentation
│   └── LLM_REFERENCE_GUIDE.md  # Fence documentation
└── README.md                # This file
```

## Architecture: The "Fence"

The LLM reference guide ("fence") is a **user-modifiable markdown file** that controls LLM behavior:

**Location:**
- `/opt/rocm/lib/python3.12/site-packages/rocpd/ai_analysis/share/llm-reference-guide.md` (default)
- Can be overridden with `ROCPD_LLM_REFERENCE_GUIDE` environment variable

**What's in the guide:**
- **ROCm Profiling Tools** - Correct tool names and commands (rocprofv3, rocprof-compute, rocprof-sys)
- **Tool Documentation Links** - Official ROCm documentation references
- **AMD GPU Hardware Specs** - MI100, MI250X, MI300X specifications
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

The module automatically detects which tier of analysis is possible based on available data:

| Tier | Data Required | Analysis Capabilities |
|------|---------------|----------------------|
| **Tier 1** | Trace data (always available) | Kernel hotspots, time breakdown, memory copy overhead |
| **Tier 2** | Trace + hardware counters (`--pmc`) | Roofline model, Speed-of-Light metrics, bottleneck classification |
| **Tier 3** | Trace + counters + PC sampling | Instruction-level hotspots |
| **Tier 4** | Trace + thread trace | Full instruction timeline, stall analysis |

**Note:** Tiers 3 and 4 are future enhancements.

## API Reference

### Main Functions

```python
# Analyze database and return result object
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

    # Methods
    def to_dict() -> Dict[str, Any]
    def to_json(indent: int = 2) -> str
    def to_text() -> str
    def to_markdown() -> str
    def to_webview() -> str  # Self-contained interactive HTML report
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
└── ReferenceGuideNotFoundError
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

## Testing

### Unit Tests

```bash
cd rocm-systems-dev/projects/rocprofiler-sdk
pytest source/lib/python/rocpd/ai_analysis/tests/
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

- `ANTHROPIC_API_KEY` - Anthropic Claude API key
- `OPENAI_API_KEY` - OpenAI GPT API key
- `ROCPD_LLM_REFERENCE_GUIDE` - Path to custom reference guide

### Reference Guide Location

Default: `/opt/rocm/share/rocprofiler-sdk/llm-reference-guide.md`

Override:
```bash
export ROCPD_LLM_REFERENCE_GUIDE=/path/to/custom-guide.md
```

## Documentation

- **[AI Analysis API Documentation](../../../docs/AI_ANALYSIS_API.md)** - Complete API reference
- **[LLM Reference Guide Documentation](../../../docs/LLM_REFERENCE_GUIDE.md)** - How to customize LLM behavior
- **[rocpd README](../README.md)** - Main rocpd documentation

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
