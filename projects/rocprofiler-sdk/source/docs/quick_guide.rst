.. meta::
  :description: Quick reference guide for rocprofv3 commands and rocprofiler-sdk tools
  :keywords: rocprofv3 quick guide, rocprofiler-sdk quick reference, rocprofv3 commands, ROCprofiler-SDK CLI, GPU profiling quick start

.. _quick-guide:

==============================================
ROCprofiler-SDK Quick Reference Guide
==============================================

This quick reference guide provides an overview of the most commonly used ``rocprofv3`` commands and links to detailed documentation sections.

Getting Started
===============

Export the ROCm binary path:

.. code-block:: bash

   source /opt/rocm/share/rocprofiler-sdk/setup-env.sh

Check rocprofv3 version and help:

.. code-block:: bash

   rocprofv3 --version
   rocprofv3 --help

Essential Commands
==================

Querying System Capabilities
-----------------------------

List available counters and capabilities:

.. code-block:: bash

   # List all available features
   rocprofv3 --list-avail

   # Using the dedicated tool for detailed queries
   rocprofv3-avail list
   rocprofv3-avail info

**Documentation:** :ref:`using-rocprofv3-avail`

Basic Tracing
-------------

Application tracing (HIP API + kernel dispatches + memory operations):

.. code-block:: bash

   # Runtime tracing (recommended for most use cases)
   rocprofv3 --runtime-trace -- ./your_app

   # System-level tracing (includes HSA API)
   rocprofv3 --sys-trace -- ./your_app

**Documentation:** :ref:`using-rocprofv3`

Granular Tracing Options
------------------------

.. code-block:: bash

   # HIP API, kernel dispatches, and memory operations tracing
   rocprofv3 --hip-trace --kernel-trace --memory-copy-trace -- ./your_app


**Documentation:** :ref:`using-rocprofv3` (Basic tracing section)

Performance Counter Collection
------------------------------

.. code-block:: bash

   # List available counters
   rocprofv3-avail list --pmc

   # Check if counters can be collected together
   rocprofv3-avail pmc-check SQ_WAVES SQ_INSTS_VALU

   # Collect specific counters
   rocprofv3 --pmc SQ_WAVES,SQ_INSTS_VALU -- ./your_app

**Documentation:** :ref:`using-rocprofv3` (Counter collection section)

Advanced Profiling Features
============================

PC Sampling (Beta)
------------------

.. code-block:: bash

   # Check PC sampling support
   rocprofv3-avail list --pc-sampling

   # Enable PC sampling
   rocprofv3 --pc-sampling-beta-enabled --pc-sampling-interval 1000 -- ./your_app

**Documentation:** :ref:`using-pc-sampling`

Thread Trace
------------

.. code-block:: bash

   # Collect thread trace data
   rocprofv3 --att --output-format csv -- ./your_app

**Documentation:** :ref:`using-thread-trace`

Process Attachment
------------------

.. code-block:: bash

   # Attach to a running process by PID
   rocprofv3 --pid 12345 --runtime-trace -d ./results
   # or

   # Attach for a specific duration (10 seconds)
   rocprofv3 --pid 12345 --runtime-trace --attach-duration-msec 1000

**Documentation:** :ref:`using-rocprofv3-process-attachment`

Output Formats and Post-processing
===================================

rocprofv3 supports multiple output formats for different analysis needs. The default format is ``rocpd``, which stores data in a structured SQLite3 database.

Working with rocpd Database Format
-----------------------------------

.. code-block:: bash

   # Generate rocpd database (default format)
   rocprofv3 --runtime-trace -- ./your_app
   # Creates: hostname/pid_results.db

   # Query the database directly with SQL
   sqlite3 hostname/12345_results.db "SELECT * FROM regions;"

   # Convert rocpd database to other formats
   rocpd convert -i *.db -f csv pftrace otf2 --start 20% --end 80%

Collecting and converting to Other Formats
-------------------------------------------

.. code-block:: bash

   # Multiple output formats in one run
   rocprofv3 --runtime-trace --output-format csv json pftrace otf2 -- ./your_app


**Documentation:** :ref:`using-rocpd-output-format`

Summary and Statistics
----------------------

.. code-block:: bash

   # Overall summary statistics per domain grouped by kernel and memory operations
   rocprofv3 --runtime-trace --summary-per-domain --summary-groups "KERNEL_DISPATCH|MEMORY_COPY" -- ./your_app

**Documentation:** :ref:`using-rocprofv3` (Post-processing tracing section)

Filtering and Selection
=======================

Kernel Filtering
----------------

.. code-block:: bash

   # Include specific kernels by regex
   rocprofv3 --kernel-trace --kernel-iteration-range 10-20 --kernel-include-regex "matmul.*" --kernel-exclude-regex ".*copy.*" -- ./your_app

**Documentation:** :ref:`using-rocprofv3` (Filtering section)

Time-based Collection
---------------------

.. code-block:: bash

   # Collect for specific time periods (start_delay:collection_time:repeat)
   rocprofv3 --runtime-trace --collection-period 500:2000:0 --collection-period-unit msec -- ./your_app

**Documentation:** :ref:`using-rocprofv3` (Filtering section)

Kernel Naming and Display
=========================

.. code-block:: bash

   # Keep mangled kernel names
   rocprofv3 --kernel-trace --mangled-kernels -- ./your_app

   # Truncate kernel names for readability
   rocprofv3 --kernel-trace --truncate-kernels -- ./your_app

   # Use ROCTx regions to rename kernels
   rocprofv3 --kernel-trace --kernel-rename -- ./your_app

**Documentation:** :ref:`using-rocprofv3` (Kernel naming section)

Code Annotation with ROCTx
===========================

.. code-block:: bash

   # Trace ROCTx markers and ranges
   rocprofv3 --marker-trace -- ./your_app

**Documentation:** :ref:`using-rocprofiler-sdk-roctx`

Parallel and Distributed Applications
======================================

MPI Applications
----------------

.. code-block:: bash

   # Profile MPI applications
   mpirun -n 4 rocprofv3 --runtime-trace --output-format csv -- ./your_mpi_app

**Documentation:** :ref:`using-rocprofv3-with-mpi`

OpenMP Applications
-------------------

.. code-block:: bash

   # Profile OpenMP applications
   rocprofv3 --runtime-trace --output-format csv -- ./your_openmp_app

**Documentation:** :ref:`using-rocprofv3-with-openmp`

Output Management
=================

File Organization
-----------------

.. code-block:: bash

   # Specify output directory
   rocprofv3 --runtime-trace --output-directory ./results --output-file my_trace   -- ./your_app

   # Generate configuration file
   rocprofv3 --runtime-trace --output-config -- ./your_app

**Documentation:** :ref:`using-rocprofv3` (I/O options section)

Common Use Cases
================

Basic Performance Analysis
--------------------------

.. code-block:: bash

   # Quick performance overview
   rocprofv3 --runtime-trace --summary -- ./your_app

**Use case:** Get a high-level view of application performance

Detailed Kernel Analysis
-------------------------

.. code-block:: bash

   # Detailed kernel profiling with counters
   rocprofv3 --kernel-trace --pmc SQ_WAVES,SQ_INSTS_VALU,TCP_PERF_SEL_TOTAL_CACHE_ACCESSES -- ./your_app

**Use case:** Analyze specific kernel performance bottlenecks

Memory Transfer Analysis
------------------------

.. code-block:: bash

   # Focus on memory operations
   rocprofv3 --memory-copy-trace --memory-allocation-trace -- ./your_app

**Use case:** Optimize data movement between CPU and GPU

Timeline Visualization
----------------------

.. code-block:: bash

   # Generate timeline for visualization tools
   rocprofv3 --runtime-trace  -- ./your_app

   # Convert to Perfetto format
   rocpd2pftrace -i hostname/pid_results.db -o perfetto_trace

**Use case:** Visualize execution timeline in Perfetto or similar tools

Installation and Setup
======================

**Installation Documentation:** :ref:`installing-rocprofiler-sdk`

**API Reference:** :doc:`Tool library <api-reference/tool_library>`

**Samples and Examples:** :doc:`Samples <how-to/samples>`

Troubleshooting Quick Tips
==========================

1. **Permission Issues:** Ensure proper access to GPU devices and ``/dev/kfd``
2. **Counter Collection Fails:** Use ``rocprofv3-avail pmc-check`` to verify counter compatibility
3. **Large Output Files:** Use ``--minimum-output-data`` to set file size thresholds
4. **Signal Handling:** Use ``--disable-signal-handlers`` if conflicts with application handlers
5. **ROCm Path Issues:** Use ``--rocm-root`` to specify custom ROCm installation paths

For comprehensive documentation on each feature, refer to the detailed sections linked throughout this guide.
