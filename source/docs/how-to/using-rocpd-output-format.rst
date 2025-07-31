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
  Organizes trace data by HIP stream abstractions rather than low-level HSA queue identifiers, providing higher-level application context for kernel and memory transfer operations.

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

