.. meta::
    :description: "ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software."
    :keywords: "ROCprofiler-SDK, ROCProfiler-SDK output formats, rocpd, SQLite3, CSV, JSON, PFTrace, OTF2"

.. _using-rocpd-output-format:

=========================
Using rocpd Output Format
=========================

``rocprofv3`` supports the following output formats:

- **rocpd** (SQLite3 Database, Default)
- **CSV**
- **JSON** (Custom format for programmatic analysis only)
- **PFTrace** (Perfetto trace for visualization with Perfetto)
- **OTF2** (Open Trace Format for visualization with compatible third-party tools)

The ``rocpd`` output format is the default for ``rocprofv3``. It stores profiling results in a SQLite3 database, providing a structured and efficient way to analyze and post-process profiling data. This format allows users to query and manipulate profiling data using SQL, making it easy to extract specific information or perform complex analyses.

Features
++++++++

- **Rich Data Model**: Stores all collected profiling data, including traces, counters, and metadata, in a single `.db` (SQLite3) file.
- **Programmatic Access**: Can be queried using standard SQL tools or libraries (e.g., `sqlite3` CLI, Python's `sqlite3` module).
- **Post-Processing**: Enables advanced analysis and visualization using custom scripts or third-party tools that support SQLite3.

Generating rocpd Output
+++++++++++++++++++++++

To generate output in rocpd format, simply use:

.. code-block:: bash

   rocprofv3 --hip-trace -- <application>

Or use the ``--output-format`` option with ``rocpd``:

.. code-block:: bash

   rocprofv3 --hip-trace --output-format rocpd -- <application>

The output will be saved as ``%hostname%/%pid%_results.db``, where ``%hostname%`` is the name of the host machine and ``%pid%`` is the process ID of the application being profiled.

Converting rocpd to Other Formats
+++++++++++++++++++++++++++++++++

The ``rocpd`` output format can be converted to other formats for further analysis or visualization.  
First, ensure the ``rocpd`` Python module is available in your environment:

.. code-block:: bash

   export PYTHONPATH=<install-path>/lib/pythonX.Y/site-packages:$PYTHONPATH

where ``<install-path>`` is the ROCm installation path (usually ``/opt/rocm-<major.minor.patch>``), and ``X.Y`` is your Python version.

Once the ``rocpd`` module is available, use the ``rocpd convert`` command to convert the output to other formats.

Convert to CSV format:

.. code-block:: bash

   python3 -m rocpd convert -i <input-file>.db --output-format csv

The converted CSV will be saved as ``rocpd-output-data/out_hip_api_trace.csv`` in the current working directory.

Convert to OTF2 format:

.. code-block:: bash

   python3 -m rocpd convert -i <input-file>.db --output-format otf2

Convert to PFTrace format:

.. code-block:: bash

   python3 -m rocpd convert -i <input-file>.db --output-format pftrace

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
  Input path and filename to one or more database(s), separated by spaces.

- ``-f {csv,pftrace,otf2} [{csv,pftrace,otf2} ...]``, ``--output-format {csv,pftrace,otf2} [{csv,pftrace,otf2} ...]``  
  Specify one or more output formats. Supported: ``csv``, ``pftrace``, ``otf2``.

**I/O Options:**

- ``-o OUTPUT_FILE``, ``--output-file OUTPUT_FILE``  
  Sets the base output file name (default: ``out``).

- ``-d OUTPUT_PATH``, ``--output-path OUTPUT_PATH``  
  Sets the output directory (default: ``./rocpd-output-data``).

**Kernel Naming Options:**

- ``--kernel-rename``  
  Use ROCTx marker names instead of kernel names.

**Generic Options:**

- ``--agent-index-value {absolute,relative,type-relative}``  
  Device identification format in output:
  
  - ``absolute``: Uses node_id (e.g., Agent-0, Agent-2, Agent-4), ignoring cgroups.
  - ``relative``: Uses logical_node_id (e.g., Agent-0, Agent-1, Agent-2), considering cgroups. *(Default)*
  - ``type-relative``: Uses logical_node_type_id (e.g., CPU-0, GPU-0, GPU-1), numbering resets for each device type.

**Perfetto Trace (pftrace) Options:**

- ``--perfetto-backend {inprocess,system}``  
  Perfetto data collection backend. ``system`` mode requires running ``traced`` and ``perfetto`` daemons (default: ``inprocess``).

- ``--perfetto-buffer-fill-policy {discard,ring_buffer}``  
  Policy for handling new records when buffer is full (default: ``discard``).

- ``--perfetto-buffer-size KB``  
  Buffer size for perfetto output in KB (default: 1 GB).

- ``--perfetto-shmem-size-hint KB``  
  Perfetto shared memory size hint in KB (default: 64 KB).

- ``--group-by-queue``  
  Display HIP streams that kernels and memory copy operations are submitted to, rather than HSA queues.

**Time Window Options:**

- ``--start START``  
  Start time as percentage or nanoseconds from trace file (e.g., ``50%`` or ``781470909013049``).

- ``--start-marker START_MARKER``  
  Named marker event to use as window start point.

- ``--end END``  
  End time as percentage or nanoseconds from trace file (e.g., ``75%`` or ``3543724246381057``).

- ``--end-marker END_MARKER``  
  Named marker event to use as window end point.

- ``--inclusive INCLUSIVE``  
  ``True``: include events if START or END in window; ``False``: only if BOTH in window (default: ``True``).

**Help:**

- ``-h``, ``--help``  
  Show help message and exit.

Examples
++++++++

Convert one database to Perfetto trace:

.. code-block:: bash

   python3 -m rocpd convert -i db1.db --output-format pftrace

Convert two databases to Perfetto trace, set output path and filename, and limit to last 70% of trace:

.. code-block:: bash

   python3 -m rocpd convert -i db1.db db2.db --output-format pftrace -d "./output/" -o "twoFileTraces" --start 30% --end 100%

Convert six databases to CSV and Perfetto trace formats:

.. code-block:: bash

   python3 -m rocpd convert -i db{0..5}.db --output-format csv pftrace -d "~/output_folder/" -o "sixFileTraces"

Convert two databases to CSV, OTF2, and Perfetto trace formats:

.. code-block:: bash

   python3 -m rocpd convert -i db{3,4}.db --output-format csv otf2 pftrace

