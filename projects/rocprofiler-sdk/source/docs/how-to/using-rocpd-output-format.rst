.. meta::
    :description: "ROCprofiler-SDK rocpd output format documentation - comprehensive guide for SQLite3 database storage, format conversion utilities, and multi-format export capabilities for GPU profiling data analysis."
    :keywords: "ROCprofiler-SDK, rocpd, SQLite3, profiling database, format conversion, CSV export, JSON export, PFTrace, OTF2, GPU profiling, trace analysis"

.. _using-rocpd-output-format:

=========================
Using rocpd Output Format
=========================

``rocprofv3`` provides comprehensive support for multiple output formats to accommodate diverse analysis workflows:

- **rocpd** (SQLite3 Database) - Default format providing structured data storage
- **CSV** (Comma-Separated Values) - Tabular format for spreadsheet applications and data analysis tools
- **JSON** (JavaScript Object Notation) - Structured format optimized for programmatic analysis and integration
- **PFTrace** (Perfetto Protocol Buffers) - Binary trace format for high-performance visualization using Perfetto
- **OTF2** (Open Trace Format 2) - Standardized trace format for interoperability with third-party analysis tools

The ``rocpd`` output format serves as the primary data repository for ``rocprofv3`` profiling sessions. This format leverages SQLite3's ACID-compliant database engine to provide robust, structured storage of comprehensive profiling datasets. The relational schema enables efficient querying and manipulation of profiling data through standard SQL interfaces, facilitating complex analytical operations and custom reporting workflows.

Features
++++++++

- **Comprehensive Data Model**: Consolidates all profiling artifacts including execution traces, performance counters, hardware metrics, and contextual metadata within a single SQLite3 database file (`.db` extension).
- **Standards-Compliant Access**: Supports querying through industry-standard SQL interfaces including command-line tools (``sqlite3`` CLI), programming language bindings (Python ``sqlite3`` module, C/C++ SQLite API), and database management applications.
- **Advanced Analytics Integration**: Facilitates sophisticated post-processing workflows through custom analytical scripts, automated reporting systems, and integration with third-party visualization and analysis frameworks that provide SQLite3 connectivity.

Generating rocpd Output
+++++++++++++++++++++++

To generate profiling data in the default rocpd format:

.. code-block:: bash

   rocprofv3 --hip-trace -- <application>

Alternatively, explicitly specify the rocpd output format using the ``--output-format`` parameter:

.. code-block:: bash

   rocprofv3 --hip-trace --output-format rocpd -- <application>

The profiling session generates output files following the naming convention ``%hostname%/%pid%_results.db``, where ``%hostname%`` represents the system hostname and ``%pid%`` corresponds to the process identifier of the profiled application.

Converting rocpd to Alternative Formats
+++++++++++++++++++++++++++++++++++++

The ``rocpd`` database format supports conversion to alternative output formats for specialized analysis and visualization workflows.

The ``rocpd`` conversion utility is distributed as part of the ROCm installation package, located in ``/opt/rocm-<version>/bin``, and provides both executable and Python module interfaces for programmatic integration.

Invoke the ``rocpd convert`` command with appropriate parameters to transform database files into target formats.

**CSV Format Conversion:**

.. code-block:: bash

   /opt/rocm/bin/rocpd convert -i <input-file>.db --output-format csv

**Python Interpreter Compatibility:**

When encountering Python interpreter version conflicts, specify the appropriate Python executable explicitly:

.. code-block:: bash

   python3.10 $(which rocpd) convert -f csv -i <input-file>.db

The CSV conversion process generates output files in the ``rocpd-output-data/out_hip_api_trace.csv`` path relative to the current working directory.

**OTF2 Format Conversion:**

.. code-block:: bash

   /opt/rocm/bin/rocpd convert -i <input-file>.db --output-format otf2

**Perfetto Trace Format Conversion:**

.. code-block:: bash

   /opt/rocm/bin/rocpd convert -i <input-file>.db --output-format pftrace

rocpd convert Command-Line Options
++++++++++++++++++++++++++++++++++

.. code-block:: none

   usage: rocpd convert [-h] -i INPUT [INPUT ...] -f {csv,pftrace,otf2} [{csv,pftrace,otf2} ...]
                        [-o OUTPUT_FILE] [-d OUTPUT_PATH] [--kernel-rename]
                        [--agent-index-value {absolute,relative,type-relative}]
                        [--perfetto-backend {inprocess,system}]
                        [--perfetto-buffer-fill-policy {discard,ring_buffer}]
                        [--perfetto-buffer-size KB] [--perfetto-shmem-size-hint KB]
                        [--group-by-queue]
                        [--start START | --start-marker START_MARKER]
                        [--end END | --end-marker END_MARKER]
                        [--inclusive INCLUSIVE]

Options
-------

**Required Arguments:**

- ``-i INPUT [INPUT ...]``, ``--input INPUT [INPUT ...]``  
  Specifies input database file paths. Accepts multiple SQLite3 database files separated by whitespace for batch processing operations.

- ``-f {csv,pftrace,otf2} [{csv,pftrace,otf2} ...]``, ``--output-format {csv,pftrace,otf2} [{csv,pftrace,otf2} ...]``  
  Defines target output format(s). Supports concurrent conversion to multiple formats: ``csv`` (Comma-Separated Values), ``pftrace`` (Perfetto Protocol Buffers), ``otf2`` (Open Trace Format 2).

**I/O Configuration:**

- ``-o OUTPUT_FILE``, ``--output-file OUTPUT_FILE``  
  Configures the base filename for generated output files (default: ``out``).

- ``-d OUTPUT_PATH``, ``--output-path OUTPUT_PATH``  
  Specifies the target directory for output file generation (default: ``./rocpd-output-data``).

**Kernel Identification Options:**

- ``--kernel-rename``  
  Substitutes kernel function names with corresponding ROCTx marker annotations for enhanced semantic context.

**Device Identification Configuration:**

- ``--agent-index-value {absolute,relative,type-relative}``  
  Controls device identification methodology in converted output:
  
  - ``absolute``: Utilizes hardware node identifiers (e.g., Agent-0, Agent-2, Agent-4), bypassing container group abstractions.
  - ``relative``: Employs logical node identifiers (e.g., Agent-0, Agent-1, Agent-2), incorporating container group context. *(Default)*
  - ``type-relative``: Applies device-type-specific logical identifiers (e.g., CPU-0, GPU-0, GPU-1), with independent numbering sequences per device class.

**Perfetto Trace Configuration:**

- ``--perfetto-backend {inprocess,system}``  
  Configures Perfetto data collection architecture. The ``system`` backend requires active ``traced`` and ``perfetto`` daemon processes, while ``inprocess`` operates autonomously (default: ``inprocess``).

- ``--perfetto-buffer-fill-policy {discard,ring_buffer}``  
  Defines buffer overflow handling strategy: ``discard`` drops new records when capacity is exceeded, ``ring_buffer`` overwrites oldest records (default: ``discard``).

- ``--perfetto-buffer-size KB``  
  Sets the trace buffer capacity in kilobytes for Perfetto output generation (default: 1,048,576 KB / 1 GB).

- ``--perfetto-shmem-size-hint KB``  
  Specifies shared memory allocation hint for Perfetto inter-process communication in kilobytes (default: 64 KB).

- ``--group-by-queue``  
   Displays the HSA queues to which these kernel and memory operations were submitted. By default, ``rocprofv3`` shows the HIP streams to which the kernel and memory copy operations were submitted

**Temporal Filtering Configuration:**

- ``--start START``  
  Defines trace window start boundary using percentage notation (e.g., ``50%``) or absolute nanosecond timestamps (e.g., ``781470909013049``).

- ``--start-marker START_MARKER``  
  Specifies named marker event identifier to establish trace window start boundary.

- ``--end END``  
  Defines trace window end boundary using percentage notation (e.g., ``75%``) or absolute nanosecond timestamps (e.g., ``3543724246381057``).

- ``--end-marker END_MARKER``  
  Specifies named marker event identifier to establish trace window end boundary.

- ``--inclusive INCLUSIVE``  
  Controls event inclusion criteria: ``True`` includes events with either start or end timestamps within the specified window; ``False`` requires both timestamps within the window (default: ``True``).

**Command-Line Help:**

- ``-h``, ``--help``  
  Displays comprehensive command syntax, parameter descriptions, and usage examples.

Examples
++++++++

**Single Database Conversion to Perfetto Format:**

.. code-block:: bash

   /opt/rocm/bin/rocpd convert -i db1.db --output-format pftrace

**Multi-Database Conversion with Temporal Filtering:**

Convert multiple databases to Perfetto format, specifying custom output directory and filename, with temporal window constraint to the final 70% of the trace duration:

.. code-block:: bash

   /opt/rocm/bin/rocpd convert -i db1.db db2.db --output-format pftrace -d "./output/" -o "twoFileTraces" --start 30% --end 100%

**Batch Conversion to Multiple Formats:**

Process six database files simultaneously, generating both CSV and Perfetto trace outputs with custom output configuration:

.. code-block:: bash

   /opt/rocm/bin/rocpd convert -i db{0..5}.db --output-format csv pftrace -d "~/output_folder/" -o "sixFileTraces"

**Comprehensive Format Conversion:**

Convert multiple databases to all supported formats (CSV, OTF2, and Perfetto trace) in a single operation:

.. code-block:: bash

   /opt/rocm/bin/rocpd convert -i db{3,4}.db --output-format csv otf2 pftrace

Dedicated Conversion Tools
++++++++++++++++++++++++++

ROCprofiler-SDK provides specialized conversion utilities for efficient format-specific operations. These tools offer streamlined interfaces for single-format conversions and are particularly useful in automated workflows and scripts.

rocpd2csv - CSV Export Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Converts rocpd SQLite3 databases to Comma-Separated Values (CSV) format for spreadsheet analysis and data processing workflows.

**Location:** ``/opt/rocm/bin/rocpd2csv``

**Syntax:**

.. code-block:: bash

   rocpd2csv -i INPUT [INPUT ...] [OPTIONS]

**Key Features:**

- **Structured Data Export:** Converts hierarchical database content to tabular CSV format
- **Multi-Database Support:** Aggregates data from multiple database files into unified CSV output
- **Time Window Filtering:** Apply temporal filters to limit exported data range
- **Configurable Output:** Customize output file naming and directory structure

**Usage Examples:**

.. code-block:: bash

   # Basic CSV conversion
   rocpd2csv -i profile_data.db

   # Convert multiple databases with custom output path
   rocpd2csv -i db1.db db2.db db3.db -d ~/analysis_output/ -o combined_profile

   # Apply time window filtering (export middle 70% of execution)
   rocpd2csv -i large_profile.db --start 15% --end 85%

**Common Output Files:**
- ``out_hip_api_trace.csv`` - HIP API call trace data
- ``out_kernel_trace.csv`` - GPU kernel execution information
- ``out_counter_collection.csv`` - Hardware performance counter data

rocpd2otf2 - Open Trace Format 2 Export
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Generates OTF2 (Open Trace Format 2) files for high-performance trace analysis using tools like Vampir, Tau, and Score-P viewers.

**Location:** ``/opt/rocm/bin/rocpd2otf2``

**Syntax:**

.. code-block:: bash

   rocpd2otf2 -i INPUT [INPUT ...] [OPTIONS]

**Key Features:**

- **HPC-Standard Format:** Produces traces compatible with scientific computing analysis tools
- **Hierarchical Timeline:** Preserves process/thread/queue relationships in trace structure
- **Scalable Storage:** Efficient binary format for large-scale profiling data
- **Agent Indexing:** Configurable GPU agent indexing strategies (absolute, relative, type-relative)

**Usage Examples:**

.. code-block:: bash

   # Generate OTF2 trace archive
   rocpd2otf2 -i gpu_workload.db

   # Multi-process trace with custom indexing
   rocpd2otf2 -i mpi_rank_*.db --agent-index-value type-relative -o mpi_trace

   # Time-windowed trace export
   rocpd2otf2 -i long_execution.db --start-marker "computation_begin" --end-marker "computation_end"

**Output Structure:**
- ``trace.otf2`` - Main trace archive containing timeline data
- ``trace.def`` - Trace definition file with metadata
- Supporting files for multi-stream trace data

rocpd2pftrace - Perfetto Trace Export
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Converts rocpd databases to Perfetto protocol buffer format for interactive visualization using the Perfetto UI (ui.perfetto.dev).

**Location:** ``/opt/rocm/bin/rocpd2pftrace``

**Syntax:**

.. code-block:: bash

   rocpd2pftrace -i INPUT [INPUT ...] [OPTIONS]

**Key Features:**

- **Interactive Visualization:** Optimized for modern web-based trace viewers
- **Real-time Analysis:** Supports streaming analysis workflows
- **GPU Timeline Integration:** Specialized visualization of GPU execution patterns
- **Configurable Backend:** Supports both in-process and system-wide tracing backends

**Backend Configuration Options:**

.. code-block:: bash

   # In-process backend (default)
   rocpd2pftrace -i profile.db --perfetto-backend inprocess

   # System-wide tracing backend
   rocpd2pftrace -i system_profile.db --perfetto-backend system \
                 --perfetto-buffer-size 64MB --perfetto-shmem-size-hint 32MB

**Buffer Management:**

.. code-block:: bash

   # Ring buffer mode (overwrites old data)
   rocpd2pftrace -i continuous_profile.db --perfetto-buffer-fill-policy ring_buffer

   # Discard mode (stops recording when full)
   rocpd2pftrace -i bounded_profile.db --perfetto-buffer-fill-policy discard

**Usage Examples:**

.. code-block:: bash

   # Basic Perfetto trace generation
   rocpd2pftrace -i application.db

   # High-throughput configuration
   rocpd2pftrace -i heavy_workload.db --perfetto-buffer-size 128MB \
                 --perfetto-buffer-fill-policy ring_buffer

   # Multi-queue analysis
   rocpd2pftrace -i multi_stream.db --group-by-queue -o queue_analysis

**Visualization Workflow:**
1. Generate ``.perfetto-trace`` file using ``rocpd2pftrace``
2. Open https://ui.perfetto.dev in web browser
3. Load generated trace file for interactive analysis

rocpd2summary - Statistical Analysis Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Generates comprehensive statistical summaries and performance analysis reports from rocpd profiling data.

**Location:** ``/opt/rocm/bin/rocpd2summary``

**Syntax:**

.. code-block:: bash

   rocpd2summary -i INPUT [INPUT ...] [OPTIONS]

**Key Features:**

- **Multi-Format Output:** Supports console, CSV, HTML, JSON, Markdown, and PDF report generation
- **Comprehensive Statistics:** Kernel execution times, API call frequencies, memory transfer analysis
- **Domain-Specific Analysis:** Separate summaries for HIP, ROCr, Markers, and other trace domains
- **Rank-Based Analysis:** Per-process and per-rank performance breakdowns for MPI applications
- **Configurable Scope:** Selective inclusion/exclusion of analysis categories

**Output Format Options:**

.. code-block:: bash

   # Console output (default)
   rocpd2summary -i profile.db

   # CSV format for data analysis
   rocpd2summary -i profile.db --format csv -o performance_metrics

   # HTML report with visualization
   rocpd2summary -i profile.db --format html -d ~/reports/

   # Multiple output formats
   rocpd2summary -i profile.db --format csv html json

**Analysis Categories:**

.. code-block:: bash

   # Include all available domains
   rocpd2summary -i profile.db --region-categories HIP ROCR MARKERS KERNEL

   # Focus on GPU kernel analysis only
   rocpd2summary -i profile.db --region-categories KERNEL

   # Exclude markers to speed up processing
   rocpd2summary -i profile.db --region-categories HIP ROCR KERNEL

**Advanced Analysis Options:**

.. code-block:: bash

   # Include domain-specific statistics
   rocpd2summary -i multi_gpu.db --domain-summary

   # Per-rank analysis for MPI applications
   rocpd2summary -i mpi_profile_*.db --summary-by-rank --format html

   # Time-windowed summary analysis
   rocpd2summary -i long_run.db --start 25% --end 75% --format csv

**Report Content:**
- **Kernel Statistics:** Execution time distributions, call frequencies, grid/block sizes
- **API Timing:** HIP/ROCr function call durations and frequencies
- **Memory Analysis:** Transfer patterns, bandwidth utilization, allocation statistics  
- **Device Utilization:** GPU occupancy patterns and idle time analysis
- **Synchronization Overhead:** Barrier and synchronization point analysis

**Output Files:**
- ``summary_kernel.{format}`` - GPU kernel execution summary
- ``summary_hip_api.{format}`` - HIP API call statistics
- ``summary_rocr_api.{format}`` - ROCr runtime API analysis
- ``summary_memory.{format}`` - Memory operation statistics

Tool Integration and Workflow Examples
+++++++++++++++++++++++++++++++++++++++

**Multi-Format Analysis Pipeline:**

.. code-block:: bash

   # Generate all analysis formats for comprehensive review
   rocpd2csv -i profile.db -o analysis_data
   rocpd2summary -i profile.db --format html -o performance_report
   rocpd2pftrace -i profile.db -o interactive_trace

**Automated Performance Monitoring:**

.. code-block:: bash

   #!/bin/bash
   # Performance analysis automation script
   
   PROFILE_DB="$1"
   OUTPUT_DIR="analysis_$(date +%Y%m%d_%H%M%S)"
   
   mkdir -p "$OUTPUT_DIR"
   
   # Generate CSV data for automated analysis
   rocpd2csv -i "$PROFILE_DB" -d "$OUTPUT_DIR" -o raw_data
   
   # Create summary reports
   rocpd2summary -i "$PROFILE_DB" --format csv html \
                 -d "$OUTPUT_DIR" -o performance_summary
   
   # Generate interactive trace for detailed investigation
   rocpd2pftrace -i "$PROFILE_DB" -d "$OUTPUT_DIR" -o interactive_trace



