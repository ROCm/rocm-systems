.. meta::
  :description: Quick reference guide for rocprofv3 commands and rocprofiler-sdk tools
  :keywords: rocprofv3 quick guide, rocprofiler-sdk quick reference, rocprofv3 commands, ROCprofiler-SDK CLI, GPU profiling quick start

.. _quick-guide:

=======================================
ROCprofiler-SDK quick reference guide
=======================================

This quick reference guide provides an overview of the most commonly used ``rocprofv3`` commands and documentation links to various useful functionalities. For comprehensive documentation on each feature, click on the respective link.

Getting started
================

Export the ROCm binary path:

.. code-block:: bash

   source /opt/rocm/share/rocprofiler-sdk/setup-env.sh

Check rocprofv3 version and help:

.. code-block:: bash

   rocprofv3 --version
   rocprofv3 --help

Essential commands
===================

Querying system capabilities
-----------------------------

List available counters and capabilities:

.. code-block:: bash

   # List all available features
   rocprofv3 --list-avail

   # Use the dedicated tool for detailed queries
   rocprofv3-avail list
   rocprofv3-avail info

**Documentation:** :ref:`using-rocprofv3-avail`

Basic tracing
--------------

Application tracing (HIP API + kernel dispatches + memory operations):

.. code-block:: bash

   # Runtime tracing (recommended for most use cases)
   rocprofv3 --runtime-trace -- ./your_app

   # System-level tracing (includes HSA API)
   rocprofv3 --sys-trace -- ./your_app

**Documentation:** :ref:`application-tracing`

Granular tracing options
-------------------------

.. code-block:: bash

   # HIP API, kernel dispatches, and memory operations tracing
   rocprofv3 --hip-trace --kernel-trace --memory-copy-trace -- ./your_app

**Documentation:** :ref:`application-tracing`

Performance counter collection
-------------------------------

.. code-block:: bash

   # List available counters
   rocprofv3-avail list --pmc

   # Check if counters can be collected together
   rocprofv3-avail pmc-check SQ_WAVES SQ_INSTS_VALU

   # Collect specific counters
   rocprofv3 --pmc SQ_WAVES,SQ_INSTS_VALU -- ./your_app

**Documentation:** :ref:`kernel-counter-collection`

Advanced profiling features
============================

PC sampling (beta)
-------------------

.. code-block:: bash

   # Check PC sampling support
   rocprofv3-avail list --pc-sampling

   # Enable PC sampling
   rocprofv3 --pc-sampling-beta-enabled --pc-sampling-interval 1000 -- ./your_app

**Documentation:** :ref:`using-pc-sampling`

Thread trace
-------------

.. code-block:: bash

   # Collect thread trace data
   rocprofv3 --att --output-format csv -- ./your_app

**Documentation:** :ref:`using-thread-trace`

Process attachment
------------------

.. code-block:: bash

   # Attach to a running process by PID
   rocprofv3 --pid 12345 --runtime-trace -d ./results
   # or

   # Attach for a specific duration (10 seconds)
   rocprofv3 --pid 12345 --runtime-trace --attach-duration-msec 1000

**Documentation:** :ref:`rocprofv3-process-attachment`

Output formats and post-processing
===================================

``rocprofv3`` supports multiple output formats for different analytical requirements. The default format is ``rocpd``, which stores data in a structured SQLite3 database.

Working with rocpd database format
-----------------------------------

.. code-block:: bash

   # Generate rocpd database (default format)
   rocprofv3 --runtime-trace -- ./your_app
   # Creates: hostname/pid_results.db

   # Query the database directly with SQL
   sqlite3 hostname/12345_results.db "SELECT * FROM regions;"

   # Convert rocpd database to other formats
   rocpd convert -i *.db -f csv pftrace otf2 --start 20% --end 80%

**Documentation:** :ref:`using-rocpd-output-format`

Collection in various formats
------------------------------

.. code-block:: bash

   # Multiple output formats in one run
   rocprofv3 --runtime-trace --output-format csv json pftrace otf2 -- ./your_app

**Documentation:** :ref:`using-rocpd-output-format`

Summary and statistics
-----------------------

.. code-block:: bash

   # Overall summary statistics per domain grouped by kernel and memory operations
   rocprofv3 --runtime-trace --summary-per-domain --summary-groups "KERNEL_DISPATCH|MEMORY_COPY" -- ./your_app

**Documentation:** :ref:`using-rocprofv3` (Post-processing tracing section)

Filtering and selection
========================

Kernel filtering
-----------------

.. code-block:: bash

   # Include specific kernels by regex
   rocprofv3 --kernel-trace --kernel-iteration-range 10-20 --kernel-include-regex "matmul.*" --kernel-exclude-regex ".*copy.*" -- ./your_app

**Documentation:** :ref:`kernel-filtering`

Time-based collection
----------------------

.. code-block:: bash

   # Collect for specific time periods (start_delay:collection_time:repeat)
   rocprofv3 --runtime-trace --collection-period 500:2000:0 --collection-period-unit msec -- ./your_app

**Documentation:** :ref:`collection-period`

Kernel naming and display
==========================

.. code-block:: bash

   # Keep mangled kernel names
   rocprofv3 --kernel-trace --mangled-kernels -- ./your_app

   # Truncate kernel names for readability
   rocprofv3 --kernel-trace --truncate-kernels -- ./your_app

   # Use ROCTx regions to rename kernels
   rocprofv3 --kernel-trace --kernel-rename -- ./your_app

**Documentation:** :ref:`kernel-naming-filtering`

Code annotation with ROCTx
===========================

.. code-block:: bash

   # Trace ROCTx markers and ranges
   rocprofv3 --marker-trace -- ./your_app

**Documentation:** :ref:`using-rocprofiler-sdk-roctx`

Parallel and distributed applications
======================================

MPI applications
-----------------

.. code-block:: bash

   # Profile MPI applications
   mpirun -n 4 rocprofv3 --runtime-trace --output-format csv -- ./your_mpi_app

**Documentation:** :ref:`using-rocprofv3-with-mpi`

OpenMP applications
--------------------

.. code-block:: bash

   # Profile OpenMP applications
   rocprofv3 --runtime-trace --output-format csv -- ./your_openmp_app

**Documentation:** :ref:`using-rocprofv3-with-openmp`

Output management
==================

File organization
------------------

.. code-block:: bash

   # Specify output directory
   rocprofv3 --runtime-trace --output-directory ./results --output-file my_trace   -- ./your_app

   # Generate configuration file
   rocprofv3 --runtime-trace --output-config -- ./your_app

**Documentation:** :ref:`rocprofv3-io-options`

Common use cases
=================

Basic performance analysis
---------------------------

**Use case:** To get a high-level view of application performance:

.. code-block:: bash

   # Quick performance overview
   rocprofv3 --runtime-trace --summary -- ./your_app

Detailed kernel analysis
-------------------------

**Use case:** To analyze specific kernel performance bottlenecks:

.. code-block:: bash

   # Detailed kernel profiling with counters
   rocprofv3 --kernel-trace --pmc SQ_WAVES,SQ_INSTS_VALU,TCP_PERF_SEL_TOTAL_CACHE_ACCESSES -- ./your_app

Memory transfer analysis
-------------------------

**Use case:** To optimize data movement between CPU and GPU:

.. code-block:: bash

   # Focus on memory operations
   rocprofv3 --memory-copy-trace --memory-allocation-trace -- ./your_app

Timeline visualization
----------------------

**Use case:** To visualize execution timeline in Perfetto or similar tools:

.. code-block:: bash

   # Generate timeline for visualization tools
   rocprofv3 --runtime-trace  -- ./your_app

   # Convert to Perfetto format
   rocpd2pftrace -i hostname/pid_results.db -o perfetto_trace

Installation and setup
=======================

**Installation documentation:** :ref:`installing-rocprofiler-sdk`

**API reference:** :doc:`Tool library <api-reference/tool_library>`

**Samples and examples:** :doc:`Samples <how-to/samples>`

Quick troubleshooting tips
===========================

- **Permission issues:** Ensure proper access to GPU devices and ``/dev/kfd``.

- **Counter collection failure:** Use ``rocprofv3-avail pmc-check`` to verify counter compatibility.

- **Large output files:** Use ``--minimum-output-data`` to set file size thresholds.

- **Signal handling:** Use ``--disable-signal-handlers`` in case of conflicts with application handlers.

- **ROCm path issues:** Use ``--rocm-root`` to specify custom ROCm installation paths.
