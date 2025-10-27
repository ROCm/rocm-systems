# ROCm Profiling Data (rocpd) - Pure Python Implementation

The rocpd Python package provides a scriptable API for analyzing, summarizing, filtering, and merging tracing data
collected with the ROCm profiling tools suite.

**Implementation**: Pure Python (no C++ compilation required)
**Python Support**: Python 3.6+
**Dependencies**: sqlite3 (built-in), otf2 (PyPI, optional), perfnetto (PyPI, optional)

---

## Table of Contents

- [Background](#background)
- [Overview](#overview)
- [Architecture](#architecture)
- [Module Structure](#module-structure)
- [Usage Examples](#usage-examples)
- [CLI Interface](#cli-interface)
- [Pure Python Implementation Details](#pure-python-implementation-details)
- [API Reference](#api-reference)

---

## Background

In the past, the ROCm profiling tools (e.g. rocprofv3, rocprofiler-systems, etc.) have directly written data to
various output formats such as CSV, JSON, Perfetto, OTF2, etc. This approach has a significant number of flaws:

### No standardization in the CSV and JSON output formats

The ROCm profiling groups considers the standardization of the CSV and JSON output formats for all the tools
as a waste of time. Neither of these data formats scale well when large amounts of profiling data is collected
Due to the inherent overhead of parsing textual data as opposed to binary, the archane simplicity of the
CSV format, and the (general) requirement to parse/load the entire JSON file in order to perform any meaningful
data processing.

### Inability to unify output collected across multiple processes and nodes

Supporting the unification of output collected across multiple processes and nodes is a difficult endeavor.
The complexity of communicating profiling information between processes, especially when the processes exist
on separate nodes connected through a network, at best, requires integration with the job launchers and/or
explicit support for the job launchers. The general expectation for profiling tools is for them to work
regardless of the user application's choice of process-level parallelism (e.g. MPI, fork, spawn,
Python multi-processing, UPC, etc.) and job scheduler (e.g. SLURM, flux, PBS/Torque, LSD, etc.).
Adding explicit integration/support for this many flavors of parallelism and jobs schedulers is untenable.
The most consistent aspect of multi-node jobs is a shared filesystem: it is considered a necessity for the
user experience. Without a shared filesystem, the user would be responsible for transferring the application's
input and output to/from the specific nodes the job scheduler decided to give them. Thus, the most reliable
output for in-process profiling tools is adopting the approach of generating (at least) one output file per process.

In order to unify the output colleted across multiple processes, the one-output-per-process approach
requires either (A) a post-processing step which combines the various outputs into a single output,
(B) an output format which utilizes a single "metadata" file which links together the individual
outputs, or (C) a visualizer which supports opening multiple files at once. The ROCm profiling group
considers Option A are the most flexible and reliable approach since Option B does require a small
amount of inter-process communication to write the "metadata" file and Option C imposes a rigid
restriction on the choice of visualizer.

### Data filtering at the data collection stage

In rocprofiler-systems and rocprofv3 with the direct output to Perfetto approach, if the tool collects
2 GB of tracing data per-process in a multi-node job with 16 processes, Perfetto will struggle
to visualize each individual 2 GB trace and fail to load a combined 32 GB trace. In this situation, the
user must re-run the application and collect less data -- all of that tracing data from the previous run
is effectively lost. However, if rocprofiler-systems and rocprofv3 were to adopt an intermediate output
format approach and the Perfetto visualization is generated from this intermediate output format,
the user would have a multitude of options to remedy this issue. For example, the user could filter out
data (e.g. drop HSA functions from the trace), instruct the Perfetto generator to skip adding Perfetto
debug annotations on the trace events, combine the 32 GB of data and split it into 32 separate visualizations
based on time instead of processes, etc.

### Absence of automated analysis

Certain formats such as Perfetto are great for visualization. However, they lack any automated analysis
of the data. For example, a flat profile is an extremely useful companion when visually analyzing a trace
and other forms of automated analysis can quickly and easily do anomaly detection.

---

## Overview

rocpd is a Python package which understands a standardized SQLite3 schema. This Python package
provides a centralized place for a multitude of post-process analysis capabilities. The capabilities
include, but are not limited to, analyzing, summarizing, filtering, merging, and generating visualizations of
tracing data. This design allows tools such as rocprofv3, rocprofiler-systems, rocprofiler-compute, etc. to
focus on minimizing overhead during data collection and adding new data collection features. These tools simply
need to write one SQL database per process which adheres to the agreed upon rocpd SQL schema and rocpd will
handle the analysis and visualization of the data.

rocpd uses a unique approach to view multiple on-disk databases as a single-database when performing queries.
Python applications using rocpd **must** load the on-disk databases by constructing a `rocpd.importer.RocpdImportData`
object with a list of the database filepaths or by using the `rocpd.connect` function which returns a
`rocpd.importer.RocpdImportData` object.

---

## Architecture

### Pure Python Design

rocpd is implemented in **100% pure Python** with no C++ compilation required. This provides several benefits:

1. **Version Agnostic**: Single package works with all Python 3.6+ versions
2. **No Compilation**: No need for python3-dev packages or C++ compiler
3. **Easy Debugging**: Pure Python code is easier to inspect and debug
4. **Lightweight**: Small package size (~150KB)
5. **Portable**: Works on any platform with Python 3.6+

### Core Components

```
rocpd/
├── __init__.py                 # Main API entry point
├── __main__.py                 # CLI entry point (python -m rocpd)
├── importer.py                 # Database connection and multi-DB support
├── libpyrocpd_compat.py        # Pure Python replacement for C++ bindings
├── schema.py                   # Database schema definitions
├── query.py                    # SQL query utilities
├── output_config.py            # Output configuration management
├── csv.py                      # CSV export functionality
├── otf2.py                     # OTF2 trace export (requires otf2 package)
├── pftrace.py                  # Perfetto trace export (requires perfnetto)
├── summary.py                  # Summary and statistics generation
└── time_window.py              # Time-based filtering
```

### Data Flow

```
┌─────────────────┐
│ rocprofv3 Tool  │
│ (or similar)    │
└────────┬────────┘
         │ Collects profiling data
         ▼
┌─────────────────┐
│ SQLite3 DB(s)   │  ← One DB per process/rank
│ (rocpd schema)  │
└────────┬────────┘
         │ rocpd.connect()
         ▼
┌─────────────────┐
│ RocpdImportData │  ← Virtual unified database
└────────┬────────┘
         │
         ├─→ Query (SQL)
         ├─→ Export CSV
         ├─→ Export OTF2
         ├─→ Export Perfetto
         └─→ Generate Summary
```

---

## Module Structure

### `__init__.py` - Main API

Entry point for the rocpd package. Provides:
- `connect()` - Connect to one or more rocpd databases
- `format_path()` - Format output paths with placeholders (%pid%, %rank%, etc.)
- `version_info` - Package version information

### `importer.py` - Database Connection

**Class**: `RocpdImportData`

Manages connections to one or more SQLite databases and provides a unified interface for querying.

**Key Features**:
- Connects to multiple SQLite databases simultaneously
- Provides a single view across all databases
- Supports all sqlite3.Connection methods (execute, cursor, etc.)
- Handles database schema validation

**Example**:
```python
import rocpd

# Connect to single database
data = rocpd.connect("trace.db")

# Connect to multiple databases (multi-process run)
data = rocpd.connect(["rank0.db", "rank1.db", "rank2.db"])

# Execute queries
for row in data.execute("SELECT * FROM kernels"):
    print(row)
```

### `libpyrocpd_compat.py` - Core Data Types

Pure Python replacement for the old C++ PyBind11 bindings. Provides:

**Enumerations**:
- `agent_indexing` - Agent indexing modes (node, node_and_agent)
- `buffer_category` - Buffer categories (device, host, other)
- `copy_kind` - Memory copy types (none, host_to_host, host_to_device, etc.)
- `domain` - Profiling domains (kernel_dispatch, hip_api, hsa_api, etc.)

**Data Classes**:
- `output_config` - Output configuration
- `agent` - GPU agent information
- `correlation_id` - Correlation ID data
- `counter_value` - Performance counter values
- `kernel_dispatch` - Kernel dispatch records
- `memory_allocation` - Memory allocation records
- `memory_copy` - Memory copy records
- `string_entry` - String table entries

**Functions**:
- `read_agents()` - Read agent info from database
- `read_correlation_ids()` - Read correlation IDs
- `read_counter_values()` - Read counter values
- `read_kernel_dispatches()` - Read kernel dispatch records
- `read_memory_allocations()` - Read memory allocation records
- `read_memory_copies()` - Read memory copy records
- `read_strings()` - Read string table entries

**Python 3.6 Compatibility**:
The module includes a compatibility shim for Python 3.6 (which lacks dataclasses):
```python
try:
    from dataclasses import dataclass, field
except ImportError:
    # Fallback for Python 3.6
    def dataclass(cls):
        return cls
```

### `schema.py` - Database Schema

Defines the rocpd SQLite database schema:
- Table definitions
- Column types
- Relationships
- Validation

### `query.py` - Query Utilities

Helper functions for common database queries:
- Time range filtering
- Data aggregation
- Join operations
- Statistics calculation

### `output_config.py` - Output Configuration

Manages output file configuration:
- Output path formatting
- File naming conventions
- Format selection
- Configuration validation

### `csv.py` - CSV Export

Exports rocpd data to CSV format:
- Agent information CSV
- Kernel dispatch CSV
- Memory copy CSV
- Memory allocation CSV
- API trace CSV

### `otf2.py` - OTF2 Export

Exports rocpd data to OTF2 trace format for visualization in tools like Vampir.

**Dependency**: Requires `otf2` package from PyPI:
```bash
pip install otf2>=3.0
```

### `pftrace.py` - Perfetto Export

Exports rocpd data to Perfetto trace format for visualization in Perfetto UI.

**Dependency**: Requires `perfnetto` package from PyPI:
```bash
pip install perfnetto>=0.1
```

### `summary.py` - Summary Generation

Generates statistical summaries of profiling data:
- Per-kernel statistics
- Per-API statistics
- Domain summaries
- Rank summaries
- Time-based aggregations

### `time_window.py` - Time Filtering

Provides time-based filtering of profiling data:
- Select time ranges
- Filter by timestamp
- Window-based analysis

### `__main__.py` - CLI Interface

Command-line interface for rocpd. Run with `python -m rocpd`.

---

## Usage Examples

### Basic Connection and Query

```python
import rocpd

# Connect to database
data = rocpd.connect("output_results.db")

# Execute SQL queries directly
agents = data.execute("SELECT * FROM rocpd_info_agent").fetchall()
print(f"Found {len(agents)} agents")

# Use high-level API
agents = rocpd.libpyrocpd.read_agents(data)
for agent in agents:
    print(f"Agent {agent.id}: {agent.name}")
```

### Multi-Process Data

```python
import rocpd

# Connect to multiple databases from MPI run
databases = [
    "output_rank0_results.db",
    "output_rank1_results.db",
    "output_rank2_results.db",
    "output_rank3_results.db"
]

data = rocpd.connect(databases)

# Query unified view
kernels = rocpd.libpyrocpd.read_kernel_dispatches(data)
print(f"Total kernels across all ranks: {len(kernels)}")
```

### Export to OTF2

```python
import rocpd

# Connect to database
data = rocpd.connect("trace.db")

# Export to OTF2 (requires: pip install otf2)
from rocpd.otf2 import export_otf2
export_otf2(data, output_path="trace.otf2")
```

### Export to Perfetto

```python
import rocpd

# Connect to database
data = rocpd.connect("trace.db")

# Export to Perfetto (requires: pip install perfnetto)
from rocpd.pftrace import export_perfetto
export_perfetto(data, output_path="trace.pftrace")
```

### Generate Summary

```python
import rocpd

# Connect to database
data = rocpd.connect("trace.db")

# Generate summary statistics
from rocpd.summary import generate_summary
summary = generate_summary(data, domain_summary=True, summary_by_rank=True)

# Output summary to CSV
summary.to_csv("summary.csv", format="csv")
```

### Custom SQL Queries

```python
import rocpd

data = rocpd.connect("trace.db")

# Find kernels with longest execution time
query = """
    SELECT kernel_name, duration_ns
    FROM kernels
    ORDER BY duration_ns DESC
    LIMIT 10
"""

for kernel_name, duration in data.execute(query):
    print(f"{kernel_name}: {duration/1e6:.2f} ms")
```

---

## CLI Interface

rocpd provides a command-line interface via `python -m rocpd`.

### Convert Command

Convert rocpd database to other formats:

```bash
# Export to OTF2
python -m rocpd convert -f otf2 -i trace.db -o output.otf2

# Export to Perfetto
python -m rocpd convert -f pftrace -i trace.db -o output.pftrace

# Export to CSV
python -m rocpd convert -f csv -i trace.db -d output_dir/

# Apply kernel renaming
python -m rocpd convert -f otf2 --kernel-rename -i trace.db

# Multi-process input
python -m rocpd convert -f otf2 -i rank0.db rank1.db rank2.db -o combined.otf2
```

### Summary Command

Generate summary statistics:

```bash
# Basic summary
python -m rocpd summary -i trace.db -o summary.csv

# Domain summary
python -m rocpd summary --domain-summary -i trace.db

# Per-rank summary
python -m rocpd summary --summary-by-rank -i trace.db

# Both domain and rank summaries
python -m rocpd summary --domain-summary --summary-by-rank -f csv -i trace.db
```

### Options

Common options:
- `-i, --input` - Input database file(s)
- `-o, --output` - Output file or directory
- `-f, --format` - Output format (otf2, pftrace, csv)
- `-d, --output-directory` - Output directory path
- `--kernel-rename` - Apply kernel name demangling
- `--domain-summary` - Generate per-domain summary
- `--summary-by-rank` - Generate per-rank summary

---

## Pure Python Implementation Details

### Migration from PyBind11

rocpd was previously implemented using PyBind11 C++ bindings, which required:
- Compilation for each Python version (3.8, 3.9, 3.10, 3.11, 3.12...)
- C++ compiler and python3-dev packages
- Version-specific .so files with SOABI tags
- Complex build infrastructure

The new pure Python implementation:
- Works with all Python 3.6+ versions from a single build
- Requires no compilation or dev packages
- Uses only .py source files
- Simplified build and installation

### Key Implementation Strategies

#### 1. SQLite3 Built-in Module

Instead of using C++ to query SQLite databases, we use Python's built-in `sqlite3` module:

```python
import sqlite3

# Old PyBind11 approach: C++ sqlite3 bindings
# New approach: Pure Python sqlite3
connection = sqlite3.connect("trace.db")
cursor = connection.execute("SELECT * FROM kernels")
```

#### 2. Pure Python Data Structures

Instead of C++ classes wrapped with PyBind11, we use Python dataclasses (with Python 3.6 fallback):

```python
# Old: C++ struct exposed via PyBind11
# New: Pure Python dataclass
@dataclass
class kernel_dispatch:
    dispatch_id: int
    kernel_name: str
    duration_ns: int
    # ...
```

#### 3. PyPI Packages for Complex Formats

Instead of C++ libraries for OTF2 and Perfetto:
- **OTF2**: Use `otf2` package from PyPI
- **Perfetto**: Use `perfnetto` package from PyPI

This eliminates the need to compile and link against C++ libraries.

#### 4. SQL-Based Data Access

All data access goes through SQL queries instead of custom C++ iterators:

```python
# Read all kernel dispatches
def read_kernel_dispatches(data, condition=""):
    query = f"SELECT * FROM rocpd_kernel {condition}"
    results = []
    for row in data.execute(query):
        results.append(kernel_dispatch(*row))
    return results
```

### Dependencies

**Built-in** (no installation needed):
- `sqlite3` - Database access
- `argparse` - CLI parsing
- `os`, `sys`, `pathlib` - File operations

**Optional** (from PyPI):
- `otf2>=3.0` - For OTF2 export
- `perfnetto>=0.1` - For Perfetto export

Install optional dependencies:
```bash
pip install otf2 perfnetto
```

### Performance Considerations

The pure Python implementation is efficient because:
1. **Bulk Operations**: Most operations are bulk SQL queries (fast C implementation in sqlite3)
2. **Lazy Loading**: Data is only loaded when accessed
3. **Iterator Pattern**: Large result sets use iterators to avoid loading all data into memory
4. **Database Indexes**: SQLite indexes provide fast lookups

For most use cases, the pure Python implementation performs comparably to the C++ version because the bottleneck is database I/O, not Python execution.

---

## API Reference

### High-Level API

#### `rocpd.connect(input)`
Connect to one or more rocpd databases.

**Parameters**:
- `input` (str or list): Database file path(s)

**Returns**: `RocpdImportData` object

**Example**:
```python
data = rocpd.connect("trace.db")
data = rocpd.connect(["rank0.db", "rank1.db"])
```

#### `rocpd.format_path(path, **kwargs)`
Format output path with placeholder substitution.

**Parameters**:
- `path` (str): Path template with placeholders
- `**kwargs`: Values for placeholder substitution

**Placeholders**:
- `%pid%` - Process ID
- `%rank%` - MPI rank
- `%hostname%` - Hostname

**Example**:
```python
path = rocpd.format_path("output_%rank%.db", rank=0)
# Returns: "output_0.db"
```

### Data Reading Functions

All in `rocpd.libpyrocpd`:

#### `read_agents(data, condition="")`
Read agent (GPU) information.

**Returns**: List of `agent` objects

#### `read_correlation_ids(data, condition="")`
Read correlation ID records.

**Returns**: List of `correlation_id` objects

#### `read_counter_values(data, condition="")`
Read performance counter values.

**Returns**: List of `counter_value` objects

#### `read_kernel_dispatches(data, condition="")`
Read kernel dispatch records.

**Returns**: List of `kernel_dispatch` objects

#### `read_memory_allocations(data, condition="")`
Read memory allocation records.

**Returns**: List of `memory_allocation` objects

#### `read_memory_copies(data, condition="")`
Read memory copy records.

**Returns**: List of `memory_copy` objects

#### `read_strings(data, condition="")`
Read string table entries.

**Returns**: List of `string_entry` objects

### Database Connection

#### `RocpdImportData` Class

Provides database connection and query interface.

**Methods**:
- `execute(query, params=())` - Execute SQL query
- `cursor()` - Get database cursor
- `commit()` - Commit transaction
- `rollback()` - Rollback transaction
- `close()` - Close connection

All standard `sqlite3.Connection` methods are supported.

---

## Version Information

```python
import rocpd
print(rocpd.version_info)
# {'version': '1.0.0', 'binding_type': 'pure_python', ...}
```

---

## Installation

rocpd is installed as part of rocprofiler-sdk:

```bash
# Build and install
cd rocprofiler-sdk
cmake -B build -S . -DROCPROFILER_BUILD_TESTS=ON
cmake --build build
cmake --install build --prefix /opt/rocm

# Add to Python path
export PYTHONPATH=/opt/rocm/lib/python/site-packages:$PYTHONPATH
```

### Verify Installation

```bash
python3 -c "import rocpd; print(rocpd.version_info)"
```

---

## License

rocpd is part of rocprofiler-sdk and follows the same license terms.

---

## See Also

- [rocprofiler-sdk Documentation](https://github.com/ROCm/rocprofiler-sdk)
- [rocprofv3 User Guide](../../docs/rocprofv3/)
- [Python Migration Summary](../PYTHON_MIGRATION_SUMMARY.md)
