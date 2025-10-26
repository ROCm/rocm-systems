# Jupyter Integration for ROCProfiler-Compute

## Overview

This document describes the Jupyter notebook integration for rocprofiler-compute, which allows users to analyze GPU performance data interactively in Jupyter notebooks with a simple Python API.

## Implementation Summary

### What Was Created

1. **Main Module**: `src/rocprof_compute_jupyter.py`
   - Provides a Jupyter-compatible interface for rocprofiler-compute analysis
   - Implements a simple API: `rc.open()` and `rc.analysis()`
   - Wraps existing analysis infrastructure from `analysis_base.py` and `analysis_webui.py`

2. **Example Notebook**: `examples/jupyter_analysis_example.ipynb`
   - Comprehensive examples of using the Jupyter interface
   - Demonstrates basic and advanced usage patterns
   - Shows how to work with DataFrames for custom analysis

3. **Documentation**: `docs/how-to/jupyter_analysis.rst`
   - Complete user guide for the Jupyter interface
   - API reference
   - Troubleshooting guide
   - Comparison with GUI mode

## Key Features

### Simple API

The interface provides a module-level API that matches the requested design:

```python
import rocprof_compute_jupyter as rc

# Load performance data
rc.open('/path/to/perf_data')

# Display analysis
rc.analysis()
```

### Core Functions

1. **`rc.open(perf_data_dir, **kwargs)`**
   - Loads performance data from a directory
   - Initializes the analysis engine
   - Supports configuration options (time_unit, normalization, filters, etc.)

2. **`rc.analysis(filter_kernel=None, filter_gpu=None, filter_dispatch=None, show_basic_only=False)`**
   - Displays analysis results in the notebook
   - Supports filtering by kernel, GPU, and dispatch
   - Shows interactive Plotly charts and pandas DataFrames

3. **`rc.get_dataframe(table_id)`**
   - Returns specific data tables as pandas DataFrames
   - Enables custom analysis and visualization

4. **`rc.list_tables()`**
   - Lists all available data tables with descriptions

### Architecture

The implementation follows these design principles:

1. **Reuse Existing Infrastructure**
   - Extends `OmniAnalyze_Base` class
   - Reuses data loading from `file_io` module
   - Leverages existing chart building from `utils.gui`

2. **Jupyter-Specific Adaptations**
   - Uses `IPython.display` for inline output
   - Returns Plotly figures for interactive visualization
   - Provides direct DataFrame access for custom analysis

3. **Module-Level State**
   - Maintains global `_current_analysis` object
   - Allows simple function calls without object management
   - Matches the requested API design

## Differences from GUI Mode

### GUI Mode (`--gui`)
- Standalone Dash web application
- Runs on localhost with web server
- Real-time interactive filtering via web UI
- No programming required

### Jupyter Mode
- Runs within Jupyter notebooks
- Programmatic API for analysis
- Direct access to underlying data
- Supports custom analysis and visualization
- Better for reproducible workflows

## Usage Examples

### Basic Analysis

```python
import rocprof_compute_jupyter as rc

# Load data
rc.open('/path/to/workload_dir')

# Show all metrics
rc.analysis()
```

### Filtered Analysis

```python
# Filter by specific kernels
rc.analysis(filter_kernel=['0', '1'])

# Filter by GPU
rc.analysis(filter_gpu=[0])

# Combined filters
rc.analysis(filter_kernel=['0'], filter_gpu=[0])
```

### Custom Analysis

```python
# Get kernel statistics
kernel_df = rc.get_dataframe(1)

# Find top kernels
top_kernels = kernel_df.nlargest(5, 'Mean_Duration')

# Custom visualization
import matplotlib.pyplot as plt
top_kernels.plot(x='Kernel_Name', y='Mean_Duration', kind='bar')
plt.show()
```

### Advanced Configuration

```python
rc.open(
    '/path/to/workload_dir',
    time_unit='us',           # Use microseconds
    normal_unit='per_cycle',  # Normalize by cycles
    max_stat_num=20,          # Show top 20 stats
    decimal=3                 # 3 decimal places
)
```

## Technical Details

### Class Structure

```
JupyterAnalysis (extends OmniAnalyze_Base)
├── __init__(): Initialize with minimal argparse.Namespace
├── pre_processing(): Load and prepare data
├── run_analysis(): Required by base class (no-op in Jupyter)
├── display_results(): Main display logic for Jupyter
└── _display_dataframe(): Helper for rendering tables/charts
```

### Module-Level Functions

```
open() → Initializes JupyterAnalysis instance
analysis() → Calls display_results() on current instance
get_dataframe() → Returns DataFrame from current instance
list_tables() → Lists available tables
```

### Data Flow

1. User calls `rc.open(path)`
2. Creates `JupyterAnalysis` instance
3. Initializes SOCs and architecture configs
4. Runs `sanitize()` and `pre_processing()`
5. Loads all performance data into DataFrames

6. User calls `rc.analysis()`
7. Applies any filters
8. Loads table data with filters
9. Displays results using IPython.display
10. Shows Plotly charts or pandas DataFrames

## Dependencies

The Jupyter interface requires:
- `jupyter` or `jupyterlab`
- `ipython`
- `plotly` (for interactive charts)
- `pandas` (already required by rocprof-compute)

All other dependencies are already part of rocprofiler-compute.

## Integration Points

The implementation integrates with existing code at these points:

1. **`analysis_base.py`**: Extends `OmniAnalyze_Base`
2. **`file_io.py`**: Uses data loading functions
3. **`parser.py`**: Uses data parsing and filtering
4. **`utils.gui`**: Reuses chart building functions
5. **`schema.py`**: Uses Workload and ArchConfig schemas
6. **SOC modules**: Initializes architecture-specific objects

## Future Enhancements

Potential improvements for future versions:

1. **Widget Support**: Add ipywidgets for interactive filtering
2. **Export Functions**: Save analysis results to files
3. **Comparison Mode**: Compare multiple profiling runs
4. **Custom Metrics**: Allow users to define custom metrics
5. **Caching**: Cache loaded data for faster re-analysis
6. **Progress Indicators**: Show progress for long operations

## Testing

To test the implementation:

1. Profile a sample application:
   ```bash
   rocprof-compute profile -w test_workload -- ./sample_app
   ```

2. Create a Jupyter notebook:
   ```python
   import rocprof_compute_jupyter as rc
   rc.open('test_workload')
   rc.analysis()
   ```

3. Verify:
   - Data loads successfully
   - Charts display correctly
   - Filters work as expected
   - DataFrames are accessible

## Conclusion

The Jupyter integration provides a powerful, flexible interface for rocprofiler-compute analysis that complements the existing CLI and GUI modes. It enables:

- Interactive exploration in notebooks
- Programmatic access to performance data
- Custom analysis and visualization
- Reproducible analysis workflows
- Easy sharing and collaboration

The implementation reuses existing infrastructure while providing a clean, simple API that matches the requested design pattern.
