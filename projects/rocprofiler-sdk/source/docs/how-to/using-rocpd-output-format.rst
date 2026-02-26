.. meta::
    :description: "ROCprofiler-SDK rocpd output format documentation - comprehensive guide for SQLite3 database storage, format conversion utilities, and multi-format export capabilities for GPU profiling data analysis."
    :keywords: "ROCprofiler-SDK, rocpd, SQLite3, profiling database, format conversion, CSV export, JSON export, PFTrace, OTF2, GPU profiling, trace analysis"

.. _using-rocpd-output-format:

=========================
Using rocpd output format
=========================

To accommodate diverse analysis workflows, ``rocprofv3`` provides comprehensive support for multiple output formats:

- **rocpd** (SQLite3 database) - Default format providing structured data storage.
- **CSV** (Comma-Separated Values) - Tabular format for spreadsheet applications and data analysis tools.
- **JSON** (JavaScript Object Notation) - Structured format optimized for programmatic analysis and integration.
- **PFTrace** (Perfetto protocol buffers) - Binary trace format for high-performance visualization using Perfetto.
- **OTF2** (Open Trace Format 2) - Standardized trace format for interoperability with third-party analysis tools.

The ``rocpd`` output format serves as the primary data repository for ``rocprofv3`` profiling sessions. This format leverages SQLite3's ACID-compliant database engine to provide robust, structured storage of comprehensive profiling datasets. The relational schema enables efficient querying and manipulation of profiling data through standard SQL interfaces, facilitating complex analytical operations and custom reporting workflows.

Features
--------

- **Comprehensive data model:** Consolidates all profiling artifacts including execution traces, performance counters, hardware metrics, and contextual metadata within a single SQLite3 database file (``.db`` extension).
- **Standards-compliant access:** Supports querying through industry-standard SQL interfaces including command-line tools (``sqlite3`` CLI), programming language bindings (Python ``sqlite3`` module, C/C++ SQLite API), and database management applications.
- **Advanced analytics integration:** Facilitates sophisticated post-processing workflows through custom analytical scripts, automated reporting systems, and integration with third-party visualization and analysis frameworks that provide SQLite3 connectivity.

Generating rocpd output
-----------------------

To generate profiling data in the default ``rocpd`` format, use:

.. code-block:: bash

   rocprofv3 --hip-trace -- <application>

Or, explicitly specify the ``rocpd`` output format using the ``--output-format`` parameter:

.. code-block:: bash

   rocprofv3 --hip-trace --output-format rocpd -- <application>

The profiling session generates output files following the naming convention ``%hostname%/%pid%_results.db``, where:

- ``%hostname%``: The system hostname.

- ``%pid%``: The process identifier of the profiled application.

Converting rocpd to alternative formats
---------------------------------------

The ``rocpd`` database format supports conversion to alternative output formats for specialized analysis and visualization workflows.

The ``rocpd`` conversion utility is distributed as part of the ROCm installation package, located in ``/opt/rocm-<version>/bin``, and provides both executable and Python module interfaces for programmatic integration.

To transform database files into target formats, run the ``rocpd convert`` command with appropriate parameters.

- **CSV format conversion**

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i <input-file>.db --output-format csv

  The CSV conversion process generates a consolidated trace output file ``rocpd-output-data/out_regions_trace.csv`` path relative to the current working directory.

This consolidated approach replaces the previous API-specific CSV files (``out_hip_api_trace.csv``, ``out_hsa_api_trace.csv``, ``out_marker_api_trace.csv``, etc.) to provide comprehensive coverage of all traced regions, including MPI functions, pthread functions, and other regions captured by rocprofiler-systems beyond the core ROCm APIs.

- **OTF2 format conversion**

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i <input-file>.db --output-format otf2

- **Perfetto trace format conversion**

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i <input-file>.db --output-format pftrace

**Python interpreter compatibility:**

On encountering Python interpreter version conflicts, specify the appropriate Python executable explicitly:

.. code-block:: bash

  python3.10 $(which rocpd) convert -f csv -i <input-file>.db

rocpd convert command-line options
+++++++++++++++++++++++++++++++++++

The command-line options as displayed using ``rocpd convert --help`` are listed here:

.. code-block:: none

   usage: rocpd convert [-h] -i INPUT [INPUT ...] -f {csv,pftrace,otf2} [{csv,pftrace,otf2} ...]
                        [-o OUTPUT_FILE] [-d OUTPUT_PATH] [--automerge-limit LIMIT]
                        [--kernel-rename]
                        [--agent-index-value {absolute,relative,type-relative}]
                        [--perfetto-backend {inprocess,system}]
                        [--perfetto-buffer-fill-policy {discard,ring_buffer}]
                        [--perfetto-buffer-size KB] [--perfetto-shmem-size-hint KB]
                        [--group-by-queue]
                        [--start START | --start-marker START_MARKER]
                        [--end END | --end-marker END_MARKER]
                        [--inclusive INCLUSIVE]

The following table provides a detailed listing of the ``rocpd convert`` command-line options:

.. # COMMENT: The following lines define a line break for use in the table below.
.. |br| raw:: html

    <br />

.. list-table::
  :header-rows: 1

  * - Category
    - Option
    - Description

  * - Required arguments
    - | ``-i INPUT [INPUT ...]``, ``--input INPUT [INPUT ...]`` |br| |br| |br| |br|
      | ``-f {csv,pftrace,otf2} [{csv,pftrace,otf2} ...]``, ``--output-format {csv,pftrace,otf2} [{csv,pftrace,otf2} ...]``
    - | Specifies input database file paths. Accepts multiple SQLite3 database files separated by whitespace for batch processing operations. |br| |br|
      | Defines target output formats. Supports concurrent conversion to multiple formats such as CSV, PFTrace, and OTF2.

  * - I/O configuration
    - | ``-o OUTPUT_FILE``, ``--output-file OUTPUT_FILE`` |br| |br|
      | ``-d OUTPUT_PATH``, ``--output-path OUTPUT_PATH`` |br| |br|
      | ``--automerge-limit LIMIT``
    - | Configures the base filename for generated output files (default: ``out``). |br| |br|
      | Specifies the target directory for output file generation (default: ``./rocpd-output-data``). |br| |br|
      | Controls the database auto-merge limit. When the number of input databases exceeds this limit, they are automatically merged into a ``.rpdb`` package to stay within SQLite3's attach limit (max 10). Default: 1, maximum: 8.

  * - Kernel identification Options
    - ``--kernel-rename``
    - Substitutes kernel function names with the corresponding ROCTx marker annotations for enhanced semantic context.

  * - Device identification configuration
    - ``--agent-index-value {absolute,relative,type-relative}``
    - Controls device identification methodology in the converted output. Here are the values:

      - ``absolute``: Utilizes hardware node identifiers such as Agent-0, Agent-2, and Agent-4, while bypassing container group abstractions.
      - ``relative``: Employs logical node identifiers such as Agent-0, Agent-1, and Agent-2, while incorporating container group context. This is the Default value.
      - ``type-relative``: Applies device-type-specific logical identifiers such as CPU-0, GPU-0, and GPU-1, with independent numbering sequence per device class.

  * - Perfetto trace configuration
    - | ``--perfetto-backend {inprocess,system}`` |br| |br| |br| |br| |br| |br|
      | ``--perfetto-buffer-fill-policy {discard,ring_buffer}`` |br| |br| |br| |br| |br|
      | ``--perfetto-buffer-size KB`` |br| |br| |br| |br|
      | ``--perfetto-shmem-size-hint KB`` |br| |br| |br| |br|
      | ``--group-by-queue``
    - | Configures Perfetto data collection architecture. The value ``system`` requires active ``traced`` and ``perfetto`` daemon processes, while ``inprocess`` operates autonomously. The default value is ``inprocess``. |br| |br|
      | Defines buffer overflow handling strategy. The value ``discard`` drops new records when capacity is exceeded and ``ring_buffer`` overwrites oldest records. The default value is ``discard``. |br| |br|
      | Sets the trace buffer capacity (in kilobytes) for Perfetto output generation. The default value is 1,048,576 KB or 1 GB. |br| |br|
      | Specifies shared memory allocation hint (in kilobytes) for Perfetto interprocess communication. The default value is 64 KB. |br| |br|
      | Organizes trace data by HIP stream abstractions rather than low-level HSA queue identifiers, providing higher-level application context for kernel and memory transfer operations.

  * - Temporal filtering configuration
    - | ``--start START`` |br| |br| |br| |br| |br|
      | ``--start-marker START_MARKER`` |br| |br| |br|
      | ``--end END`` |br| |br| |br| |br| |br|
      | ``--end-marker END_MARKER`` |br| |br| |br|
      | ``--inclusive INCLUSIVE``
    - | Defines trace window start boundary using percentage notation such as ``50%`` or absolute nanosecond timestamps such as ``781470909013049``. |br| |br|
      | Specifies named marker event identifier to establish trace window start boundary. |br| |br|
      | Defines trace window end boundary using percentage notation such as ``75%`` or absolute nanosecond timestamps such as ``3543724246381057``. |br| |br|
      | Specifies named marker event identifier to establish trace window end boundary. |br| |br|
      | Controls event inclusion criteria. The value ``True`` includes events with either start or end timestamps within the specified window while ``False`` requires both timestamps within the window. The default value is ``True``.

  * - Command-line Help
    - ``-h``, ``--help``
    - Displays comprehensive command syntax, parameter descriptions, and usage examples.

Types of conversion
--------------------

Here are the types of conversion supported by ``rocpd``:

- Single database conversion to Perfetto format

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i db1.db --output-format pftrace

- Multi-Database conversion with temporal filtering

  The following example converts multiple databases to Perfetto format while specifying custom output directory and filename with temporal window constraint set to the final 70% of the trace duration:

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i db1.db db2.db --output-format pftrace -d ./output/ -o twoFileTraces --start 30% --end 100%

- Batch conversion into multiple formats

  The following example processes six database files simultaneously, generating both CSV and Perfetto trace outputs with custom output configuration:

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i db{0..5}.db -f csv pftrace -d ~/output_folder/ -o sixFileTraces

- Comprehensive format conversion

  The following example converts multiple databases into all supported formats (CSV, OTF2, and Perfetto trace) in a single operation:

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i db{3,4}.db -f csv otf2 pftrace

- Convert .rpdb package

  The following example converts a packaged database collection to Perfetto format:

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i my_profiling_data.rpdb -f pftrace

**Automatic database merging:**

When multiple database files are provided as input, ``rocpd convert`` automatically manages database file counts to stay within SQLite3's attach limit:

- SQLite3 has a maximum limit of 10 attached databases
- If the number of input databases exceeds the automerge limit (default: 1), the tool automatically merges them into a temporary ``.rpdb`` package
- The automerge limit can be controlled with the ``--automerge-limit`` parameter (max: 8, conservatively set below SQLite3's limit of 10)
- For explicit control over merging, use the ``rocpd merge`` or ``rocpd package`` commands before analysis (see :ref:`managing-multiple-databases`)

Example with automerge control:

.. code-block:: bash

    # Allow up to 4 databases without automatic merging
    rocpd convert -i db1.db db2.db db3.db db4.db --automerge-limit 4 -f pftrace

Example with automerge control using an index.yaml package file:

.. code-block:: bash

    # Don't automerge, use the index.yaml package file to attach databases as is
    rocpd convert -i index.yaml --automerge-limit 6 -f csv

**Automatic merging in action:**

When multiple databases exceed the automerge limit, you'll see output like this:

.. code-block:: console

    $ rocpd convert -i database1.db database2.db -f pftrace
    Found 2 database files.
    More than 1 database files found. It is recommended to merge and package databases
    Original number of DBs: 2, Target number of DBs to merge per batch: 2
    Adding database1.db
    Tables found: 20
    Adding database2.db
    Tables found: 20
    Merge completed successfully! Output saved to: rocpd-20260205-011104.rpdb/merged_db_0_e7e1a495-8185-4875-bdcd-6570e9b2fa81.db
    Time: 1.02 sec
    Reduced to 1 database files.
    Converting database(s) to pftrace format:
    ...
    Done. Exiting...

The rocpd convert command automatically:

- Detected 2 input databases exceeding the limit (default: 1)
- Created a timestamped ``.rpdb`` folder (``rocpd-20260205-011104.rpdb``)
- Merged the databases to stay within SQLite3's attach limit (max 10 attached databases)
- Proceeded with the requested conversion operation

The merged database remains in the ``.rpdb`` folder for future use, eliminating the need to re-merge for subsequent operations.

For more information on explicit database merging and packaging, see :ref:`managing-multiple-databases`.

.. _conversion-tools:

Dedicated conversion tools
---------------------------

ROCprofiler-SDK provides specialized conversion utilities for efficient format-specific operations. The following tools offer streamlined interfaces for single-format conversions, which are particularly useful in automated workflows and scripts.

rocpd2csv - CSV export tool
++++++++++++++++++++++++++++

**Purpose:** To convert ``rocpd`` SQLite3 databases to CSV format for spreadsheet analysis and data processing workflows.

**Location:** ``/opt/rocm/bin/rocpd2csv``

**Syntax:**

.. code-block:: bash

  rocpd2csv -i INPUT [INPUT ...] [OPTIONS]

**Key features:**

- Structured data export: Converts hierarchical database content to tabular CSV format.

- Multidatabase support: Aggregates data from multiple database files into a unified CSV output.

- Time window filtering: Applies temporal filters to limit exported data range.

- Configurable output: Facilitates customized output file naming and directory structure.

**Usage examples:**

.. code-block:: bash

    # Basic CSV conversion
    rocpd2csv -i profile_data.db

    # Convert multiple databases with custom output path
    rocpd2csv -i db1.db db2.db db3.db -d ~/analysis_output/ -o combined_profile

    # Apply time window filtering (export middle 70% of execution)
    rocpd2csv -i large_profile.db --start 15% --end 85%

**Common output files:**

- ``out_hip_api_trace.csv``: HIP API call trace data.

- ``out_kernel_trace.csv``: GPU kernel execution information.

- ``out_counter_collection.csv``: Hardware performance counter data.

rocpd2otf2 - OTF2 export tool
++++++++++++++++++++++++++++++

**Purpose:** To generate OTF2 files for high-performance trace analysis using tools such as Vampir, Tau, and Score-P viewers.

**Location:** ``/opt/rocm/bin/rocpd2otf2``

**Syntax:**

.. code-block:: bash

    rocpd2otf2 -i INPUT [INPUT ...] [OPTIONS]

**Key features:**

- HPC-standard format: Produces traces compatible with scientific computing analysis tools.

- Hierarchical timeline: Preserves process, thread, or queue relationships in trace structure.

- Scalable storage: Efficient binary format for large-scale profiling data.

- Agent indexing: Configurable GPU agent indexing strategies (absolute, relative, or type-relative).

**Usage examples:**

.. code-block:: bash

    # Generate OTF2 trace archive
    rocpd2otf2 -i gpu_workload.db

    # Multi-process trace with custom indexing
    rocpd2otf2 -i mpi_rank_*.db --agent-index-value type-relative -o mpi_trace

    # Time-windowed trace export
    rocpd2otf2 -i long_execution.db --start-marker "computation_begin" --end-marker "computation_end"

**Output structure:**

- ``trace.otf2``: Main trace archive containing timeline data.

- ``trace.def``: Trace definition file with metadata.

- Supporting files for multistream trace data.

rocpd2pftrace - Perfetto trace export
++++++++++++++++++++++++++++++++++++++

**Purpose:** To convert ``rocpd`` databases to Perfetto protocol buffer format for interactive visualization using the `Perfetto UI <https://ui.perfetto.dev/>`_.

**Location:** ``/opt/rocm/bin/rocpd2pftrace``

**Syntax:**

.. code-block:: bash

    rocpd2pftrace -i INPUT [INPUT ...] [OPTIONS]

**Key features:**

- Interactive visualization: Optimized for modern web-based trace viewers.

- Real-time analysis: Supports streaming analysis workflows.

- GPU timeline integration: Specialized visualization of GPU execution patterns.

- Configurable backend: Supports both in-process and system-wide tracing backends.

**Backend configuration options:**

.. code-block:: bash

    # In-process backend (default)
    rocpd2pftrace -i profile.db --perfetto-backend inprocess

    # System-wide tracing backend
    rocpd2pftrace -i system_profile.db --perfetto-backend system \
                    --perfetto-buffer-size 64MB --perfetto-shmem-size-hint 32MB

**Buffer management options:**

.. code-block:: bash

    # Ring buffer mode (overwrites old data)
    rocpd2pftrace -i continuous_profile.db --perfetto-buffer-fill-policy ring_buffer

    # Discard mode (stops recording when full)
    rocpd2pftrace -i bounded_profile.db --perfetto-buffer-fill-policy discard

**Usage examples:**

.. code-block:: bash

    # Basic Perfetto trace generation
    rocpd2pftrace -i application.db

    # High-throughput configuration
    rocpd2pftrace -i heavy_workload.db --perfetto-buffer-size 128MB \
                    --perfetto-buffer-fill-policy ring_buffer

    # Multi-queue analysis
    rocpd2pftrace -i multi_stream.db --group-by-queue -o queue_analysis

**Visualization workflow:**

Follow these steps to visualize traces on Perfetto:

1. Generate ``.perfetto-trace`` file using ``rocpd2pftrace``.

2. Open https://ui.perfetto.dev in web browser.

3. Load generated trace file for interactive analysis.

rocpd2summary - statistical analysis tool
++++++++++++++++++++++++++++++++++++++++++

**Purpose:** To generate comprehensive statistical summaries and performance analysis reports from ``rocpd`` profiling data.

**Location:** ``/opt/rocm/bin/rocpd2summary``

**Syntax:**

.. code-block:: bash

    rocpd2summary -i INPUT [INPUT ...] [OPTIONS]

**Key features:**

- Multiformat output: Supports console, CSV, HTML, JSON, Markdown, and PDF report generation.

- Comprehensive statistics: Statistics include kernel execution times, API call frequencies, and memory transfer analysis.

- Domain-specific analysis: Generates separate summaries for HIP, ROCr, Markers, and other trace domains.

- Rank-based analysis: Per-process and per-rank performance breakdowns for MPI applications.

- Configurable scope: Selective inclusion or exclusion of analysis categories.

**Output format options:**

.. code-block:: bash

    # Console output (default)
    rocpd2summary -i profile.db

    # CSV format for data analysis
    rocpd2summary -i profile.db --format csv -o performance_metrics

    # HTML report with visualization
    rocpd2summary -i profile.db --format html -d ~/reports/

    # Multiple output formats
    rocpd2summary -i profile.db --format csv html json

.. _analysis-categories:

**Analysis categories:**

Here is how you can generate summary for specific trace domains:

.. code-block:: bash

    # By default, all available domains will be processed and you can identify which domain regions are included in your profiled data
    rocpd2summary -i profile.db

    # Only include HIP and HSA regions (and skip others) to speed up analysis
    rocpd2summary -i profile.db --region-categories HIP HSA

    # Exclude all domain categories so that only the kernels and memory copies are analyzed, to speed up analysis
    rocpd2summary -i profile.db --region-categories NONE


**Advanced analysis options:**

The following commands demonstrate the usage of advanced analysis options such as ``domain-summary``, ``summary-by-rank``, and ``start/end``:

.. code-block:: bash

    # Include domain-specific statistics
    rocpd2summary -i multi_gpu.db --domain-summary

    # Per-rank analysis for MPI applications
    rocpd2summary -i mpi_profile_*.db --summary-by-rank --format html

    # Time-windowed summary analysis
    rocpd2summary -i long_run.db --start 25% --end 75% --format csv

**Report content:**

- Kernel statistics: Execution time distributions, call frequencies, and grid or block sizes.

- API timing: HIP or HSA function call durations and frequencies.

- Memory analysis: Transfer patterns, bandwidth utilization, and allocation statistics.

- Device utilization: GPU occupancy patterns and idle time analysis.

- Synchronization overhead: Barrier and synchronization point analysis.

**Output files:**

- ``kernels_summary.{format}`` - GPU kernel execution summary.

- ``hip_summary.{format}`` - HIP API call statistics.

- ``hsa_summary.{format}`` - HSA runtime API analysis.

- ``memory_summary.{format}`` - Memory operation statistics.

- ``markers_summary.{format}`` - Marker event analysis.

Generating performance summary using rocpd
-------------------------------------------

The ``rocpd summary`` command provides statistical analysis and performance summaries, similar to the summary functionality available in ``rocprofv3``. This command generates comprehensive reports from ``rocpd`` database files, offering the same analytical capabilities that were previously available through ``rocprofv3 --summary`` but on the structured database format.

**Purpose:** To generate statistical summaries and performance reports from ``rocpd`` database files, providing similar functionality as ``rocprofv3`` builtin summary capabilities.

**Location:** ``/opt/rocm/bin/rocpd summary``

**Syntax:**

.. code-block:: bash

    rocpd summary -i INPUT [INPUT ...] [OPTIONS]

**Key features:**

- Compatible analysis: Provides the same summary statistics and reports as ``rocprofv3 --summary``.

- Database-driven: Operates on structured ``rocpd`` database files for consistent and reproducible analysis.

- Multidatabase aggregation: Combines and analyzes data from multiple profiling sessions, ranks, or nodes in a single operation. This is further explained in :ref:`multidatabase-summary`.

- Comparative analysis: Uses ``--summary-by-rank`` to compare performance across different ranks, nodes, or execution contexts.

- Flexible output: Generates summaries in multiple formats such as console, CSV, HTML, and JSON.

- Selective reporting: Allows generating reports based on specific performance domains and categories.

.. _multidatabase-summary:

Multidatabase summary analysis
+++++++++++++++++++++++++++++++

The ``rocpd summary`` command excels at aggregating multiple database files, providing capabilities not available with single-session analysis.

Here are the benefits of using ``rocpd summary`` for multidatabase summary:

- **Unified summary reports:**

  .. code-block:: bash

    # Aggregate multiple databases into single comprehensive summary
    rocpd summary -i session1.db session2.db session3.db --format html -o unified_summary

    # Combine all MPI rank databases for overall application analysis
    rocpd summary -i rank_*.db -f csv -o mpi_application_summary

    # Time-series aggregation across multiple profiling runs
    rocpd summary -i daily_profile_*.db -f json -o weekly_performance_trends

    # Generate summary from .rpdb package
    rocpd summary -i simulation_run_001.rpdb -f html -o mpi_performance_summary

- **Rankwise comparative analysis:**

  The ``--summary-by-rank`` option provides detailed comparative analysis, helping you to identify performance variations, load balancing issues, and optimization opportunities across different execution contexts:

  .. code-block:: bash

    # Compare performance across MPI ranks
    rocpd summary -i rank_0.db rank_1.db rank_2.db rank_3.db --summary-by-rank -f html -o rank_comparison

    # Analyze multi-node performance characteristics
    rocpd summary -i node_*.db --summary-by-rank -f csv -o node_performance_analysis

    # Compare GPU device performance in multi-GPU applications
    rocpd summary -i gpu_0.db gpu_1.db gpu_2.db gpu_3.db --summary-by-rank -f json -o gpu_scaling_analysis

Use cases for multidatabase summary analysis
#############################################

- **MPI application performance analysis:**

  .. code-block:: bash

    # Profile distributed MPI application
    mpirun -np 8 rocprofv3 --hip-trace --output-format rocpd -- mpi_simulation

    # Generate unified summary for overall application performance
    rocpd summary -i results_rank_*.db --format html -o application_overview

    # Identify load balancing issues with rank-by-rank comparison
    rocpd summary -i results_rank_*.db --summary-by-rank --format csv -o load_balance_analysis

- **Multi-GPU scaling studies:**

  .. code-block:: bash

    # Profile scaling from 1 to 4 GPUs (control GPUs via HIP_VISIBLE_DEVICES)
    HIP_VISIBLE_DEVICES=0 rocprofv3 --hip-trace --output-format rocpd -o scaling_1gpu -- gpu_benchmark
    HIP_VISIBLE_DEVICES=0,1 rocprofv3 --hip-trace --output-format rocpd -o scaling_2gpu -- gpu_benchmark
    HIP_VISIBLE_DEVICES=0,1,2,3 rocprofv3 --hip-trace --output-format rocpd -o scaling_4gpu -- gpu_benchmark

    # Aggregate scaling analysis
    rocpd summary -i scaling_*.db --format html -o gpu_scaling_summary

    # Compare efficiency across different GPU counts
    rocpd summary -i scaling_*.db --summary-by-rank --format json -o scaling_efficiency

- **Performance regression testing:**

  .. code-block:: bash

    # Profile baseline and optimized versions
    rocprofv3 --hip-trace --output-format rocpd -o baseline.db -- application_v1
    rocprofv3 --hip-trace --output-format rocpd -o optimized.db -- application_v2

    # Generate unified performance comparison
    rocpd summary -i baseline.db optimized.db --summary-by-rank --format html -o regression_analysis

- **Cross-platform performance comparison:**

  .. code-block:: bash

    # Profile on different hardware platforms
    rocprofv3 --hip-trace --output-format rocpd -o platform_A.db -- benchmark
    rocprofv3 --hip-trace --output-format rocpd -o platform_B.db -- benchmark

    # Compare platform performance characteristics
    rocpd summary -i platform_*.db --summary-by-rank --format csv -o platform_comparison

- **Domain-specific reporting:**

  .. code-block:: bash

    # Cross-rank summary for MPI applications with domain focus
    rocpd summary -i rank_*.db --summary-by-rank --region-categories KERNEL HIP --format html

- **Time-windowed analysis:**

  .. code-block:: bash

    rocpd summary -i profile_*.db --start 25% --end 75% --summary-by-rank

- **Domain-specific comparative analysis:**

  .. code-block:: bash

    rocpd summary -i node_*.db --domain-summary --summary-by-rank --region-categories HIP ROCR

In summary, here is how the output generated using ``rocpd summary`` enhances performance analysis:

- **Unified summaries:** Provides aggregate statistics across all input databases, showing combined performance metrics.

- **Rankwise summaries:** Generates separate statistical reports for each input database, enabling direct comparison of performance characteristics.

- **Comparative metrics:** Highlights performance variations, identifies outliers, and reveals load balancing opportunities.

Integration with rocprofv3 workflow
++++++++++++++++++++++++++++++++++++

The ``rocpd summary`` command maintains full compatibility with ``rocprofv3`` summary analysis, while extending capabilities to multidatabase scenarios. Users familiar with ``rocprofv3 --summary`` will find identical statistical outputs and report formats when using ``rocpd summary`` on database files, with the added benefit of cross-session analysis capabilities.

**Automatic database merging:**

When multiple database files are provided as input, ``rocpd summary`` automatically manages database file counts to stay within SQLite3's attach limit:

- SQLite3 has a maximum limit of 10 attached databases
- If the number of input databases exceeds the automerge limit (default: 1), the tool automatically merges them into a temporary ``.rpdb`` package
- The automerge limit can be controlled with the ``--automerge-limit`` parameter (max: 8, conservatively set below SQLite3's limit of 10)
- For explicit control over merging, use the ``rocpd merge`` or ``rocpd package`` commands before analysis (see :ref:`managing-multiple-databases`)

Example with automerge control:

.. code-block:: bash

    # Allow up to 4 databases without automatic merging
    rocpd summary -i db1.db db2.db db3.db db4.db --automerge-limit 4 -f html

For explicit control over database organization, see :ref:`managing-multiple-databases`.

Aggregating multiprofiling data using rocpd
--------------------------------------------

One of the key advantages of the ``rocpd`` format is its ability to aggregate and analyze data from multiple profiling sessions, ranks, or nodes within a unified framework. This capability enables comprehensive analysis workflows, which was not possible with earlier output formats.

Here are the use cases of data aggregation using ``rocpd``:

- **Multidatabase analysis capabilities.**

  Unlike the Perfetto output format used in earlier versions, ``rocpd`` databases can be seamlessly combined for cross-session analysis:

  .. code-block:: bash

    # Aggregate analysis across multiple profiling sessions
    rocpd query -i session1.db session2.db session3.db \
                --query "SELECT name, AVG(duration) FROM kernels GROUP BY name"

    # Cross-rank performance comparison for MPI applications
    rocpd summary -i rank_0.db rank_1.db rank_2.db rank_3.db --summary-by-rank

    # Multi-node scaling analysis
    rocpd query -i node_*.db \
                --query "SELECT COUNT(*) as total_kernels, SUM(duration) as total_time FROM kernels"

- **Distributed computing workflows.**

  ``rocpd`` can effectively aggregate data in a distributed computing environment. Here are the examples for single and multi-GPU analysis:

  - MPI application analysis:

    .. code-block:: bash

        # Profile MPI application across multiple ranks
        mpirun -np 4 rocprofv3 --hip-trace --output-format rocpd -- mpi_application

        # Generate aggregated performance summary
        rocpd summary -i results_rank_*.db --summary-by-rank --format html -o mpi_performance_report

        # Analyze load balancing across ranks
        rocpd query -i results_rank_*.db \
                    --query "SELECT pid, COUNT(*) as kernel_count, AVG(duration) as avg_duration FROM kernels GROUP BY pid"

  - Multi-GPU scaling analysis:

    .. code-block:: bash

        # Profile application with multiple GPU devices (GPUs visible to the app via HIP_VISIBLE_DEVICES)
        HIP_VISIBLE_DEVICES=0,1,2,3 rocprofv3 --hip-trace --output-format rocpd -o multi_gpu -- multi_gpu_app

        # Aggregate device utilization analysis
        rocpd query -i multi_gpu*.db \
                    --query "SELECT agent_abs_index as device_id, COUNT(*) as operations, SUM(duration) as total_time FROM kernels GROUP BY device_id"

        # Cross-device performance comparison
        rocpd summary -i multi_gpu_results.db --domain-summary

- **Temporal aggregation.**

  - Time-series analysis:

    .. code-block:: bash

        # Collect profiles over time for performance monitoring
        for hour in {1..24}; do
            rocprofv3 --hip-trace --output-format rocpd -o "profile_hour_$hour.db" -- application
        done

        # Analyze performance trends over time
        rocpd query -i profile_hour_*.db \
                    --query "SELECT AVG(duration) as avg_kernel_time, COUNT(*) as kernel_count FROM kernels" \
                    --format csv -o performance_trends

  - Comparative analysis:

    .. code-block:: bash

        # Compare baseline vs optimized performance
        rocpd query -i baseline.db optimized.db \
                    --query "SELECT kernel, AVG(duration) as avg_time FROM kernels GROUP BY name ORDER BY avg_time DESC"

        # Generate comparative summary reports
        rocpd summary -i baseline.db optimized.db --format html -o comparison_report

Here are the benefits of data aggregation using ``rocpd``:

- Unified analysis: Combines data from different execution contexts, hardware configurations, and time periods.

- Scalability insights: Analyzes performance scaling across multiple nodes, ranks, or GPU devices.

- Trend analysis: Tracks performance evolution over time or across different software versions.

- Load balancing: Identifies performance bottlenecks and load distribution issues in distributed applications.

- Cross-platform comparison: Compares performance across different hardware platforms using unified database schema.

The aggregation capabilities of ``rocpd`` format enable sophisticated analysis workflows that provide deeper insight into application performance characteristics across diverse computing environments.

.. _managing-multiple-databases:

Managing multiple database files with rocpd
--------------------------------------------

When working with distributed profiling sessions, multi-rank MPI applications, or multi-node GPU workloads, ``rocpd`` provides specialized tools for managing and consolidating multiple database files. The ``rocpd merge`` and ``rocpd package`` commands streamline database management, enabling efficient organization and analysis of large-scale profiling datasets.

rocpd merge - Database merging tool
++++++++++++++++++++++++++++++++++++

**Purpose:** To merge multiple SQLite3 databases into a single unified database file for simplified analysis and reduced file management overhead.

**Location:** ``/opt/rocm/bin/rocpd merge``

**Syntax:**

.. code-block:: bash

    rocpd merge -i INPUT [INPUT ...] [OPTIONS]

**Key features:**

- Database consolidation: Combines multiple ``rocpd`` databases into a single unified database file.

- Schema integrity: Validates that all input databases share the same schema version before merging.

- Comprehensive merging: Preserves all database objects including tables, views, indexes, and triggers.

- Data aggregation: Creates UNION views that automatically aggregate data from all merged sources.

- Integrity handling: Re-enables SQLite foreign key enforcement for subsequent operations on the merged database.

**Command-line options:**

The following table provides a detailed listing of the ``rocpd merge`` command-line options:

.. # COMMENT: The following lines define a line break for use in the table below.
.. |br| raw:: html

    <br />

.. list-table::
  :header-rows: 1

  * - Category
    - Option
    - Description

  * - Required arguments
    - ``-i INPUT [INPUT ...]``, ``--input INPUT [INPUT ...]``
    - Specifies input database file paths. Accepts multiple SQLite3 database files separated by whitespace. Supports wildcard patterns and ``.rpdb`` folder inputs.

  * - Output configuration
    - | ``-o OUTPUT_FILE``, ``--output-file OUTPUT_FILE`` |br| |br|
      | ``-d OUTPUT_PATH``, ``--output-path OUTPUT_PATH``
    - | Sets the base output filename (default: ``merged``). The ``.db`` extension is added automatically if not provided. |br| |br|
      | Specifies the output directory path (default: ``./rocpd-output-data``).

  * - Command-line help
    - ``-h``, ``--help``
    - Displays comprehensive command syntax, parameter descriptions, and usage examples.

**Usage examples:**

- Basic database merge:

  .. code-block:: bash

    # Merge three databases into a single file
    rocpd merge -i db0.db db1.db db2.db

    # Output: ./rocpd-output-data/merged.db

- Merge with custom output location:

  .. code-block:: bash

    # Merge databases with custom output directory and filename
    rocpd merge -i db0.db db1.db db2.db -d merged3DBs -o combined_results

    # Output: merged3DBs/combined_results.db

- Merge multi-node profiling data:

  .. code-block:: bash

    # Consolidate all databases from a node directory
    rocpd merge -i node0/*.db -d node0_output -o largeMerged

    # Output: node0_output/largeMerged.db

- Merge MPI rank databases:

  .. code-block:: bash

    # Merge all rank databases from distributed MPI run
    rocpd merge -i results_rank_*.db -d mpi_merged -o unified_mpi_profile

    # Output: mpi_merged/unified_mpi_profile.db

.. warning::

   **Consider alternatives to merging large databases**

   Merging multiple large databases creates a single, very large output file that may be difficult to manage, transfer, or analyze. For large profiling datasets, consider these alternatives:

   - **Use** ``rocpd package`` **instead**: Package databases with metadata files that reference them in their current locations, avoiding the creation of a single large file.
   - **Selective merging**: Merge only subsets of databases by node or rank, then package the merged subsets.
   - **Direct analysis**: Many ``rocpd`` commands (``convert``, ``query``, ``summary``) can work directly with multiple database files or ``.rpdb`` packages without requiring a merge operation.

   **Example - Package instead of merge for large datasets:**

   .. code-block:: bash

       # Instead of merging (which creates one large file):
       # rocpd merge -i node*/rank*.db -o huge_merged.db

       # Package them to reference in-place (no large file created):
       rocpd package -i node*/rank*.db -d my_large_dataset

       # Then analyze the package:
       rocpd summary -i my_large_dataset.rpdb -f html --automerge-limit 8

**Use cases:**

- MPI application consolidation: Merge databases from all MPI ranks into a single unified database for comprehensive performance analysis.

- Multi-node profiling: Consolidate profiling data collected across multiple compute nodes for centralized analysis.

- Database file reduction: Reduce the number of database files when working with analysis tools that have file count limitations.

- Simplified analysis: Create a single database file for easier sharing, archival, or integration with external analysis tools.

**Important considerations:**

- Schema version compatibility: All input databases must have matching schema versions. Merging fails if version mismatches are detected.

- Table name uniqueness: The merge operation assumes globally unique table names across all input databases.

- Merged database size: The output database size equals the sum of all input database sizes plus index overhead.

- Large dataset management: For very large databases, consider using ``rocpd package`` instead to avoid creating an unwieldy single file.

rocpd package - Database packaging tool
++++++++++++++++++++++++++++++++++++++++

**Purpose:** To create organized database collections with metadata files, enabling efficient management and distribution of multiple profiling databases.

**Location:** ``/opt/rocm/bin/rocpd package``

**Syntax:**

.. code-block:: bash

    rocpd package -i INPUT [INPUT ...] [OPTIONS]

**Key features:**

- Metadata generation: Creates ``.rpdb`` folders containing ``index.yaml`` metadata files that reference database collections.

- Consolidation support: Optionally moves or copies database files into a centralized ``.rpdb`` folder structure.

- Flexible input handling: Accepts database files, directories, wildcard patterns, and existing ``.rpdb`` folders as input.

- In-place referencing: Can create metadata files that reference databases in their original locations without moving them.

**Command-line options:**

The following table provides a detailed listing of the ``rocpd package`` command-line options:

.. list-table::
  :header-rows: 1

  * - Category
    - Option
    - Description

  * - Required arguments
    - ``-i INPUT [INPUT ...]``, ``--input INPUT [INPUT ...]``
    - Input paths to database files or directories. Supports multiple inputs, wildcard patterns (e.g., ``*.db``), directories containing databases, and existing ``.rpdb`` folders.

  * - Package configuration
    - | ``-c``, ``--consolidate`` |br| |br| |br|
      | ``--copy`` |br| |br|
      | ``-d OUTPUT_PATH``, ``--output-path OUTPUT_PATH``
    - | Consolidates database files by moving them into the output ``.rpdb`` folder and generates metadata file. |br| |br|
      | When used with ``--consolidate``, copies database files instead of moving them. |br| |br|
      | Specifies the output folder name. Without ``--consolidate``, creates metadata in current directory. With ``--consolidate``, creates a ``.rpdb`` folder with the specified name.

  * - Command-line help
    - ``-h``, ``--help``
    - Displays comprehensive command syntax, parameter descriptions, and usage examples.

**The .rpdb package format:**

The ``.rpdb`` (ROCProfiler Database) package is a standardized folder structure for organizing multiple profiling databases:

- Folder structure: A directory with the ``.rpdb`` extension containing database files and metadata.

- Metadata file: Contains an ``index.yaml`` file in YAML format with database inventory and configuration.

- Portability: The entire ``.rpdb`` folder can be moved, archived, or shared as a self-contained profiling dataset.

- Tool integration: ``.rpdb`` folders are recognized as valid input by all ``rocpd`` commands (``convert``, ``query``, ``summary``, ``merge``).

**Metadata file structure (index.yaml):**

.. code-block:: yaml

    rocprofiler-sdk:
      rocpd:
        rocpd_package_version: "1.0"
        path: "."
        files:
          - "database1.db"
          - "database2.db"
          - "database3.db"

.. note::

   **When to use** ``rocpd package`` **instead of** ``rocpd merge``:

   - **Large databases**: Packaging avoids creating a single large merged file while maintaining organized access to all data
   - **Distributed storage**: When databases reside on different folders withing the same filesystem, packaging can reference them in-place
   - **Iterative analysis**: Package databases once, then run multiple analysis operations without repeated merging overhead
   - **Flexible organization**: Easily add or remove databases from a package by updating the metadata file

**Usage examples:**

- Create metadata file for in-place databases:

  .. code-block:: bash

    # Index databases without moving them
    rocpd package -i node0/db0.db node1/db1.db node2/db2.db

    # Output: ./index.yaml (references databases in original locations)

- Package and consolidate databases:

  .. code-block:: bash

    # Copy all databases into a new .rpdb folder
    rocpd package -i node0/db0.db node1/db1.db node2/db2.db \
                   -d my_MPI_run_1 --consolidate --copy

    # Output: my_MPI_run_1.rpdb/ folder containing copies of all databases and index.yaml

- Append databases to existing package:

  .. code-block:: bash

    # Add additional databases to existing .rpdb folder
    rocpd package -i my_MPI_run_1.rpdb node5/db5.db \
                   -d my_MPI_run_1_append_5 --consolidate --copy

    # Output: my_MPI_run_1_append_5.rpdb/ folder with expanded database collection

- Consolidate with move operation:

  .. code-block:: bash

    # Move databases into .rpdb folder (original files are moved, not copied)
    rocpd package -i my_MPI_run_1.rpdb node7/db7.db \
                   -d my_MPI_run_1 --consolidate

    # Output: my_MPI_run_1.rpdb/ folder with all databases moved into it

- Package directory of databases:

  .. code-block:: bash

    # Package all databases from a directory
    rocpd package -i ./profiling_results/ -d archived_run --consolidate --copy

    # Output: archived_run.rpdb/ folder with all .db files from profiling_results/

**Use cases:**

- MPI profiling organization: Package databases from all MPI ranks into a single portable ``.rpdb`` folder for easy distribution and analysis.

- Archival and backup: Create self-contained ``.rpdb`` packages for long-term storage or sharing with collaborators.

- Distributed profiling management: Organize databases from multi-node runs with clear metadata tracking.

- Analysis workflow simplification: Create ``.rpdb`` packages that can be directly consumed by ``rocpd convert``, ``query``, and ``summary`` commands.

- Database consolidation: Reduce filesystem clutter by collecting scattered database files into organized ``.rpdb`` folders.

Integration with other rocpd commands
++++++++++++++++++++++++++++++++++++++

The ``rocpd merge`` and ``rocpd package`` commands integrate seamlessly with other ``rocpd`` functionality, enabling sophisticated profiling workflows:

**Using .rpdb folders as input:**

All ``rocpd`` commands accept ``.rpdb`` folders as input, automatically processing the metadata and loading referenced databases:

.. code-block:: bash

    # Convert .rpdb package to Perfetto format
    rocpd convert -i my_profiling_data.rpdb -f pftrace

    # Generate summary from packaged databases
    rocpd summary -i archived_run.rpdb -f html -o performance_report

    # Query packaged databases
    rocpd query -i mpi_results.rpdb --query "SELECT * FROM top_kernels LIMIT 10"

**Automatic merging in analysis workflows:**

The ``rocpd convert``, ``query``, and ``summary`` commands automatically merge databases when the file count exceeds optimal thresholds:

.. code-block:: bash

    # Convert many databases - automatic merging if count exceeds threshold
    rocpd convert -i node_*/rank_*.db -f pftrace

    # The tool automatically merges databases into manageable batches for optimal performance

**Complete profiling and analysis workflow:**

Here is an example workflow demonstrating merge and package integration:

.. code-block:: bash

    # Step 1: Profile distributed MPI application (8 ranks)
    mpirun -np 8 rocprofv3 --hip-trace --output-format rocpd -o results_rank_%rank% -- mpi_simulation

    # Step 2: Package all rank databases with consolidation
    rocpd package -i results_rank_*.db -d simulation_run_001 --consolidate --copy

    # Step 3: Generate comprehensive summary from package
    rocpd summary -i simulation_run_001.rpdb --summary-by-rank -f html \
                  -o mpi_performance_analysis

    # Step 4: Convert to Perfetto for visualizing the data
    rocpd convert -i simulation_run_001.rpdb -f pftrace

    # Step 5: Perform custom analysis with SQL queries
    rocpd query -i simulation_run_001.rpdb \
                --query "SELECT pid, COUNT(*) as kernel_count, \
                         AVG(duration) as avg_duration \
                         FROM kernels GROUP BY pid" \
                --format csv -o rank_comparison

**Alternative workflow with merge:**

.. code-block:: bash

    # Step 1: Profile distributed application
    mpirun -np 4 rocprofv3 --hip-trace --output-format rocpd -o results_rank_%rank% -- gpu_workload

    # Step 2: Merge all rank databases into single unified database
    rocpd merge -i results_rank_*.db -d merged_results -o unified_profile

    # Step 3: Analyze merged database (simpler than handling multiple files)
    rocpd convert -i merged_results/unified_profile.db -f pftrace csv

    # Step 4: Generate unified summary
    rocpd summary -i merged_results/unified_profile.db -f html

**Combining merge and package:**

.. code-block:: bash

    # Merge subsets of databases, then package the merged results
    
    # Merge databases by node
    rocpd merge -i node0/rank_*.db -d merged_by_node -o node0_merged
    rocpd merge -i node1/rank_*.db -d merged_by_node -o node1_merged
    rocpd merge -i node2/rank_*.db -d merged_by_node -o node2_merged
    
    # Package the node-level merged databases
    rocpd package -i merged_by_node/*.db -d cluster_profiling_data \
                   --consolidate --copy

    # Analyze the packaged multi-node data
    rocpd summary -i cluster_profiling_data.rpdb --summary-by-rank \
                  -f html -o cluster_performance_report

**Benefits of integrated workflows:**

- Simplified file management: Reduce complexity of handling dozens or hundreds of database files.

- Portable analysis: Create self-contained ``.rpdb`` packages for easy sharing and archival.

- Flexible analysis: Choose between merged single-database or packaged multi-database approaches based on analysis needs.

- Scalable processing: Automatic merging ensures optimal performance regardless of database file count.

- Reproducible workflows: Metadata files document database provenance and organization for reproducible analysis.

Tool integration and workflow examples
--------------------------------------

The following examples demonstrate how to use the :ref:`rocpd conversion tools <conversion-tools>` for automated performance monitoring and multiformat analysis.

- **Multiformat analysis pipeline:**

  .. code-block:: bash

    # Generate all analysis formats for comprehensive review
    rocpd2csv -i profile.db -o analysis_data
    rocpd2summary -i profile.db --format html -o performance_report
    rocpd2pftrace -i profile.db -o interactive_trace

- **Automated performance monitoring:**

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

Executing SQL queries with rocpd
---------------------------------

The ``rocpd query`` command provides comprehensive SQL-based analysis capabilities for exploring and extracting data from ``rocpd`` databases. This command enables custom analysis workflows and automated reporting for GPU profiling data analysis, and integration with external analysis pipelines.

**Purpose:** To execute custom SQL queries against ``rocpd`` databases with support for multiple output formats, automated reporting, and Email delivery.

**Location:** ``/opt/rocm/bin/rocpd query``

**Syntax:**

.. code-block:: bash

    rocpd query -i INPUT [INPUT ...] --query "SQL_STATEMENT" [OPTIONS]

**Key features:**

- Standard SQL support: Supports full SQLite3 SQL syntax including JOINs, aggregate functions, and complex WHERE clauses.

- Multidatabase aggregation: Facilitates queries across multiple database files as a unified virtual database.

- Multiple output formats: Supports output formats such as console, CSV, HTML, JSON, Markdown, PDF, and interactive dashboards.

- Script execution: Can execute complex SQL scripts with view definitions and custom functions.

- Automated reporting: Supports Email delivery with SMTP configuration and attachment management.

- Time window integration: Allows temporal filtering before query execution.

Database schema and views
++++++++++++++++++++++++++

``rocpd`` databases provide the following comprehensive views for analysis. It is generally recommended to build a query using ``data_views``:

- **Core data views:**

  .. code-block:: sql

    -- System and hardware information
    SELECT * FROM rocpd_info_agents;
    SELECT * FROM rocpd_info_node;

    -- Kernel execution data
    SELECT * FROM kernels;
    SELECT * FROM top_kernels;

    -- API trace information
    SELECT * FROM regions_and_samples WHERE category LIKE 'HIP_%';
    SELECT * FROM regions_and_samples WHERE category LIKE 'RCCL_%';

    -- Performance counters
    SELECT * FROM counters_collection;

    -- Memory operations
    SELECT * FROM memory_copies;
    SELECT * FROM memory_allocations;

    -- Process and thread information
    SELECT * FROM processes;
    SELECT * FROM threads;

    -- Marker and region data
    SELECT * FROM regions;
    SELECT * FROM regions_and_samples WHERE category LIKE 'MARKERS_%';

- **Summary and analysis views:**

  .. code-block:: sql

    -- Top performing kernels by execution time
    SELECT * FROM top_kernels LIMIT 10;

    -- Top Analysis
    SELECT * FROM top;

    -- Busy Analysis
    SELECT * FROM busy;

rocpd query examples
+++++++++++++++++++++

The following examples demonstrate the usage of ``rocpd query`` for performing some useful operations:

- **Simple data exploration:**

  .. code-block:: bash

    # List available GPU agents
    rocpd query -i profile.db --query "SELECT * FROM rocpd_info_agents"

    # Show top 10 longest-running kernels
    rocpd query -i profile.db --query "SELECT name, duration FROM kernels ORDER BY duration DESC LIMIT 10"

    # Count total number of kernel dispatches
    rocpd query -i profile.db --query "SELECT COUNT(*) as total_kernels FROM kernels"

- **Multidatabase aggregation:**

  .. code-block:: bash

    # Combine data from multiple profiling sessions
    rocpd query -i session1.db session2.db session3.db \
                --query "SELECT pid, COUNT(*) as kernel_count FROM kernels GROUP BY pid"

    # Cross-session performance comparison
    rocpd query -i baseline.db optimized.db \
                --query "SELECT name as kernel_name, AVG(duration) as avg_duration FROM kernels GROUP BY kernel_name"

- **Query .rpdb package:**

  .. code-block:: bash

    # Execute SQL query against packaged database collection
    rocpd query -i cluster_data.rpdb \
                --query "SELECT * FROM top_kernels LIMIT 10" \
                --format csv -o top_kernels_report

- **Advanced analytics:**

  .. code-block:: bash

    # Kernel performance analysis with statistics
    rocpd query -i profile.db --query "
    SELECT
        name as kernel_name,
        COUNT(*) as dispatch_count,
        MIN(duration) as min_duration,
        AVG(duration) as avg_duration,
        MAX(duration) as max_duration,
        SUM(duration) as total_duration
    FROM kernels
    GROUP BY kernel_name
    ORDER BY total_duration DESC"

- **Memory transfer analysis:**

  .. code-block:: bash

    # Memory copy analysis by direction
    rocpd query -i profile.db --query "
    SELECT
        name as kernel_name,
        src_agent_type,
        src_agent_abs_index,
        dst_agent_type,
        dst_agent_abs_index,
        COUNT(*) as transfer_count,
        SUM(size) as total_bytes,
        SUM(duration) as total_duration
    FROM memory_copies
    GROUP BY src_agent_abs_index
    ORDER BY total_bytes DESC"

Output format options
++++++++++++++++++++++

``rocpd query`` supports the following output formats:

- **Console output (default):**

  .. code-block:: bash

    # Display results in terminal
    rocpd query -i profile.db --query "SELECT * FROM top_kernels LIMIT 5"

- **CSV export for data analysis:**

  .. code-block:: bash

    # Export to CSV file
    rocpd query -i profile.db --query "SELECT * FROM kernels" --format csv -o kernel_analysis

    # Specify custom output directory
    rocpd query -i profile.db --query "SELECT * FROM kernels" --format csv -d ~/analysis/ -o kernel_data

- **HTML reports:**

  .. code-block:: bash

    # Generate HTML table
    rocpd query -i profile.db --query "SELECT * FROM top_kernels" --format html -o performance_report

- **Interactive dashboard:**

  .. code-block:: bash

    # Create interactive HTML dashboard
    rocpd query -i profile.db --query "SELECT * FROM device_utilization" --format dashboard -o utilization_dashboard

    # Use custom dashboard template
    rocpd query -i profile.db --query "SELECT * FROM kernels" --format dashboard \
                --template-path ~/templates/custom_dashboard.html -o custom_report

- **JSON for programmatic integration:**

  .. code-block:: bash

    # Export structured JSON data
    rocpd query -i profile.db --query "SELECT * FROM counters_collection" --format json -o counter_data

- **PDF reports:**

  .. code-block:: bash

    # Generate PDF report with monospace formatting
    rocpd query -i profile.db --query "SELECT name, duration FROM top_kernels" --format pdf -o kernel_report

**Automatic database merging:**

When multiple database files are provided as input, ``rocpd query`` automatically manages database file counts to stay within SQLite3's attach limit:

- SQLite3 has a maximum limit of 10 attached databases
- If the number of input databases exceeds the automerge limit (default: 1), the tool automatically merges them into a ``.rpdb`` package
- The automerge limit can be controlled with the ``--automerge-limit`` parameter (max: 8, conservatively set below SQLite3's limit of 10)
- For explicit control over merging, use the ``rocpd merge`` or ``rocpd package`` commands before analysis (see :ref:`managing-multiple-databases`)

Example with automerge control:

.. code-block:: bash

    # Allow up to 4 databases without automatic merging
    rocpd query -i db1.db db2.db db3.db db4.db --automerge-limit 4 \
                --query "SELECT COUNT(*) FROM kernels"

Example referencing an index.yaml package file with automerge control (don't automerge, use the package as is):

.. code-block:: bash

    # Don't automerge, use the index.yaml package file to attach databases as is
    rocpd query -i index.yaml --automerge-limit 6 \
                --query "SELECT * FROM top_kernels"

To learn more about database merging and packaging workflows, see :ref:`managing-multiple-databases`.

Script-based analysis
++++++++++++++++++++++

You can use ``rocpd query`` to execute complex SQL scripts with view definitions and custom analysis logic.

Here is a sample SQL script (analysis.sql):

.. code-block:: sql

    -- Create temporary views for complex analysis
    CREATE TEMP VIEW kernel_stats AS
    SELECT
        name as kernel_name,
        COUNT(*) as dispatch_count,
        AVG(duration) as avg_duration,
        STDDEV(duration) as duration_stddev
    FROM kernels
    GROUP BY kernel_name;

    CREATE TEMP VIEW performance_outliers AS
    SELECT k.*, ks.avg_duration, ks.duration_stddev
    FROM kernels k
    JOIN kernel_stats ks ON k.name = ks.name
    WHERE ABS(k.duration - ks.avg_duration) > 2 * ks.duration_stddev;

Here is how to execute the preceding script using ``rocpd query``:

.. code-block:: bash

    # Run script then execute query
    rocpd query -i profile.db --script analysis.sql \
                --query "SELECT * FROM performance_outliers" --format html -o outlier_analysis

Time window integration
++++++++++++++++++++++++

Here is how to apply temporal filtering before query execution:

.. code-block:: bash

   # Query only middle 50% of execution timeline
   rocpd query -i profile.db --start 25% --end 75% \
               --query "SELECT COUNT(*) as kernel_count FROM kernels"

   # Use marker-based time windows
   rocpd query -i profile.db --start-marker "computation_begin" --end-marker "computation_end" \
               --query "SELECT * FROM kernels ORDER BY start_time"

   # Absolute timestamp filtering
   rocpd query -i profile.db --start 1000000000 --end 2000000000 \
               --query "SELECT * FROM kernels WHERE start_time BETWEEN 1000000000 AND 2000000000"

Automated Email reporting
++++++++++++++++++++++++++

``rocpd query`` facilitates sending reports through Email. Here is how you can send reports in different formats and use various Email configuration options:

- **Basic Email delivery:**

  .. code-block:: bash

    # Send CSV report via email
    rocpd query -i profile.db --query "SELECT * FROM top_kernels" --format csv \
                --email-to analyst@company.com --email-from profiler@company.com \
                --email-subject "Weekly Performance Report"

- **Advanced Email configuration:**

  .. code-block:: bash

    # Multiple recipients with SMTP authentication
    rocpd query -i profile.db --query "SELECT * FROM device_utilization" --format html \
                --email-to "team@company.com,manager@company.com" \
                --email-from profiler@company.com \
                --email-subject "GPU Utilization Analysis" \
                --smtp-server smtp.company.com --smtp-port 587 \
                --smtp-user profiler@company.com --smtp-password $(cat ~/.smtp_pass) \
                --inline-preview --zip-attachments

- **Dashboard Email reports:**

  .. code-block:: bash

    # Send interactive dashboard via email
    rocpd query -i profile.db --query "SELECT * FROM kernels" --format dashboard \
                --template-path ~/templates/executive_summary.html \
                --email-to executives@company.com --email-from profiler@company.com \
                --email-subject "Executive Performance Dashboard" \
                --inline-preview

Integration workflows
++++++++++++++++++++++

- **Automated analysis pipeline:**

  .. code-block:: bash

    #!/bin/bash
    # Automated reporting script

    DB_FILE="$1"
    REPORT_DATE=$(date +%Y-%m-%d)

    # Generate multiple analysis reports
    rocpd query -i "$DB_FILE" --query "SELECT * FROM top_kernels LIMIT 20" \
                --format html -o "top_kernels_$REPORT_DATE"

    rocpd query -i "$DB_FILE" --query "SELECT * FROM memory_copy_summary" \
                --format csv -o "memory_analysis_$REPORT_DATE"

    rocpd query -i "$DB_FILE" --query "SELECT * FROM device_utilization" \
                --format dashboard -o "utilization_dashboard_$REPORT_DATE" \
                --email-to team@company.com --email-from automation@company.com

- **Performance regression detection:**

  .. code-block:: bash

    # Compare current performance against baseline
    rocpd query -i baseline.db current.db --script performance_comparison.sql \
                --query "SELECT * FROM performance_regression_analysis" \
                --format html -o regression_report \
                --email-to devteam@company.com --email-from ci@company.com \
                --email-subject "Performance Regression Analysis"

- **Custom analysis functions:**

  rocpd databases support custom SQL functions for advanced analysis:

  .. code-block:: bash

    # Use built-in rocpd functions
    rocpd query -i profile.db --query "
    SELECT
        name,
        rocpd_get_string(name_id, 0, nid, pid) as full_kernel_name,
        duration
    FROM kernels
    WHERE rocpd_get_string(name_id, 0, nid, pid) LIKE '%gemm%'"

rocpd query command-line reference
+++++++++++++++++++++++++++++++++++

The command-line options as displayed using ``rocpd query --help`` are listed here:

.. code-block:: none

   usage: rocpd query [-h] -i INPUT [INPUT ...] --query QUERY [--script SCRIPT]
                      [--format {console,csv,html,json,md,pdf,dashboard,clipboard}]
                      [-o OUTPUT_FILE] [-d OUTPUT_PATH] [--automerge-limit LIMIT]
                      [--email-to EMAIL_TO] [--email-from EMAIL_FROM]
                      [--email-subject EMAIL_SUBJECT] [--smtp-server SMTP_SERVER]
                      [--smtp-port SMTP_PORT] [--smtp-user SMTP_USER]
                      [--smtp-password SMTP_PASSWORD] [--zip-attachments]
                      [--inline-preview] [--template-path TEMPLATE_PATH]
                      [--start START | --start-marker START_MARKER]
                      [--end END | --end-marker END_MARKER]

The following table provides a detailed listing of the ``rocpd query`` command-line options:

.. # COMMENT: The following lines define a line break for use in the table below.
.. |br| raw:: html

    <br />

.. list-table::
  :header-rows: 1

  * - Category
    - Option
    - Description

  * - Required arguments
    - | ``-i INPUT [INPUT ...]``, ``--input INPUT [INPUT ...]`` |br| |br| |br| |br| |br|
      | ``--query QUERY``
    - | The input database file paths. Multiple databases are merged into a unified view. |br| |br|
      | The SQL SELECT statement to be executed. Enclose complex queries in quotes.

  * - Query options
    - | ``--script SCRIPT`` |br| |br| |br| |br| |br| |br|
      | ``--format {console,csv,html,json,md,pdf,dashboard,clipboard}``
    - | The SQL script file to be executed before running the main query. Useful for creating views and functions. |br| |br|
      | The output format. Dashboard format creates interactive HTML reports. The default value is ``console``.
  * - Output configuration
    - | ``-o OUTPUT_FILE``, ``--output-file OUTPUT_FILE`` |br| |br| |br|
      | ``-d OUTPUT_PATH``, ``--output-path OUTPUT_PATH`` |br| |br|
      | ``--automerge-limit LIMIT`` |br| |br| |br|
      | ``--template-path TEMPLATE_PATH``
    - | Base filename for exported files. |br| |br|
      | Output directory path. |br| |br|
      | Controls the database auto-merge limit. When the number of input databases exceeds this limit, they are automatically merged into a ``.rpdb`` package to stay below SQLite3's attach limit (max 10). Default: 1, maximum: 8. |br| |br|
      | Jinja2 template file for dashboard format customization.

  * - Email reporting
    - | ``--email-to EMAIL_TO`` |br| |br| |br| |br|
      | ``--email-from EMAIL_FROM`` |br| |br| |br| |br|
      | ``--email-subject EMAIL_SUBJECT`` |br| |br|
      | ``--smtp-server SMTP_SERVER``, ``--smtp-port SMTP_PORT`` |br| |br| |br| |br|
      | ``--smtp-user SMTP_USER``, ``--smtp-password SMTP_PASSWORD`` |br| |br|
      | ``--zip-attachments`` |br| |br| |br|
      | ``--inline-preview``
    - | Recipient Email addresses (comma-separated for multiple recipients). |br| |br|
      | Sender Email address. This is required when using Email delivery. |br| |br|
      | Email subject line. |br| |br|
      | SMTP server configuration. The default value is ``localhost:25``. |br| |br|
      | SMTP authentication credentials. |br| |br|
      | Bundles all attachments into a single ZIP file. |br| |br|
      | Embeds HTML reports as Email body content.

  * - | Time window filtering
    - | ``--start START``, ``--end END`` |br| |br| |br| |br| |br|
      | ``--start-marker START_MARKER``, ``--end-marker END_MARKER``
    - | Temporal boundaries using percentage such as 25% or absolute timestamps. |br| |br|
      | Named marker events defining time window boundaries.
