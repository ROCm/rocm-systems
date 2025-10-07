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
++++++++

- **Comprehensive data model:** Consolidates all profiling artifacts including execution traces, performance counters, hardware metrics, and contextual metadata within a single SQLite3 database file (``.db`` extension).
- **Standards-compliant access:** Supports querying through industry-standard SQL interfaces including command-line tools (``sqlite3`` CLI), programming language bindings (Python ``sqlite3`` module, C/C++ SQLite API), and database management applications.
- **Advanced analytics integration:** Facilitates sophisticated post-processing workflows through custom analytical scripts, automated reporting systems, and integration with third-party visualization and analysis frameworks that provide SQLite3 connectivity.

Generating rocpd output
++++++++++++++++++++++++

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
++++++++++++++++++++++++++++++++++++++++

The ``rocpd`` database format supports conversion to alternative output formats for specialized analysis and visualization workflows.

The ``rocpd`` conversion utility is distributed as part of the ROCm installation package, located in ``/opt/rocm-<version>/bin``, and provides both executable and Python module interfaces for programmatic integration.

To transform database files into target formats, run the ``rocpd convert`` command with appropriate parameters.

- **CSV format conversion**

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i <input-file>.db --output-format csv

  The converted CSV files are generated as ``rocpd-output-data/out_hip_api_trace.csv``, where the ``rocpd-output-data`` is relative to the current working directory.

- **OTF2 format conversion**

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i <input-file>.db --output-format otf2

- **Perfetto trace format conversion**

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i <input-file>.db --output-format pftrace

**Python interpreter compatibility**

On encountering Python interpreter version conflicts, specify the appropriate Python executable explicitly:

.. code-block:: bash

  python3.10 $(which rocpd) convert -f csv -i <input-file>.db

Command-line options for rocpd convert
+++++++++++++++++++++++++++++++++++++++

The command-line options as displayed using ``rocpd convert --help`` are listed here:

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
      | ``-d OUTPUT_PATH``, ``--output-path OUTPUT_PATH``
    - | Configures the base filename for generated output files (default: ``out``). |br| |br|
      | Specifies the target directory for output file generation (default: ``./rocpd-output-data``).

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

Examples
++++++++

Here are the various conversion types supported by ``rocpd``:

- Single database conversion to Perfetto format

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i db1.db --output-format pftrace

- Multi-Database conversion with temporal filtering

  The following example converts multiple databases to Perfetto format while specifying custom output directory and filename with temporal window constraint set to the final 70% of the trace duration:

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i db1.db db2.db --output-format pftrace -d "./output/" -o "twoFileTraces" --start 30% --end 100%

- Batch conversion into multiple formats

  The following example processes six database files simultaneously, generating both CSV and Perfetto trace outputs with custom output configuration:

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i db{0..5}.db --output-format csv pftrace -d "~/output_folder/" -o "sixFileTraces"

- Comprehensive format conversion

  The following example converts multiple databases into all supported formats (CSV, OTF2, and Perfetto trace) in a single operation:

  .. code-block:: bash

    /opt/rocm/bin/rocpd convert -i db{3,4}.db --output-format csv otf2 pftrace
