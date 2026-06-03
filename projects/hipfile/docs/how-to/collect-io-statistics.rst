.. meta::
   :description: How to enable hipfile statistics collection and use the ais-stats tool to read bandwidth, latency, and error histograms from a running process.
   :keywords: hipfile, ais-stats, statistics, IO statistics, bandwidth, latency, histogram, ROCm, HIPFILE_STATS_LEVEL

====================================
Collect IO statistics with ais-stats
====================================

hipfile includes a built-in statistics collection framework and a companion tool, ``ais-stats``, that connects to a running hipfile process to report IO bandwidth, latency, and error histograms. This page describes how to enable statistics collection, run a hipfile workload, and attach ``ais-stats`` to read the collected data.

For background on how the statistics framework works internally, see :doc:`/conceptual/statistics-collection`. For a complete reference of the ``ais-stats`` tool, see :doc:`/reference/ais-stats-tool`.

Prerequisites
*************

- A hipfile workload (for example, ``aiscp`` or your own application using the hipfile C API or Python bindings)
- The ``ais-stats`` tool, built as part of the hipfile project
- Sufficient permissions to open a ``pidfd`` to the target process (typically the same user or root)

Step 1: Enable statistics collection
************************************

hipfile controls statistics collection through the ``HIPFILE_STATS_LEVEL`` environment variable. Set this variable before launching the hipfile workload:

- **0** — Statistics collection is disabled (default).
- **1 or higher** — Basic statistics are collected. Higher values increase the level of detail.

.. code-block:: shell

   export HIPFILE_STATS_LEVEL=1

.. note::

   The ``HIPFILE_STATS_LEVEL`` environment variable must be set in the environment of the hipfile process you want to monitor. Setting it after the process has started has no effect.

Step 2: Run a hipfile workload
******************************

Launch your hipfile application with statistics enabled. For example, using the ``aiscp`` example program:

.. code-block:: shell

   HIPFILE_STATS_LEVEL=1 ./aiscp /path/to/source /path/to/dest &

Note the process ID (PID) of the running workload. You can find it with:

.. code-block:: shell

   echo $!

Or use standard tools such as ``ps`` or ``pgrep``.

Step 3: Attach ais-stats to the process
***************************************

With the hipfile process running (and its PID known), invoke ``ais-stats`` with the target PID:

.. code-block:: shell

   ais-stats <PID>

Replace ``<PID>`` with the actual process ID of the hipfile workload.

The tool performs the following actions:

1. Opens a process file descriptor (``pidfd``) for the target process.
2. Connects to the hipfile statistics server running inside the target process.
3. Waits for the target process to complete (or polls for updated statistics).
4. Generates a formatted report and writes it to standard output.

Interpreting the output
***********************

The ``ais-stats`` report contains per-GPU, per-backend (fastpath and fallback) histogram data for the following metrics:

IO size histogram
   Shows the distribution of IO request sizes in bytes. Each bucket covers a power-of-two range starting at 4 KiB (2\ :sup:`12` bytes). There are up to 16 buckets, with the last bucket capturing all sizes above its lower bound.

IO count histogram
   Shows the number of IO operations that fell into each size bucket.

IO time histogram
   Shows the distribution of IO operation latency in microseconds, using the same bucket structure.

Error count histogram
   Shows the distribution of failed IO operations by the size of the attempted transfer.

Each metric is reported separately for read and write operations, and separately for the fastpath and fallback backends. Only GPUs that performed IO operations appear in the report.

Histogram bucket ranges
^^^^^^^^^^^^^^^^^^^^^^^

The histogram uses logarithmic bucket boundaries:

- **Bucket 0**: 0 to 4 KiB
- **Bucket 1**: 4 KiB to 8 KiB
- **Bucket 2**: 8 KiB to 16 KiB
- Each subsequent bucket doubles the range
- **Bucket 15** (the last bucket): 128 MiB and above

Common variations
*****************

Blocking until the process exits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``ais-stats`` polls and waits for the target process to finish before generating the final report. If you want to capture a snapshot while the process is still running, consult the :doc:`/reference/ais-stats-tool` for available options.

Using higher statistics levels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setting ``HIPFILE_STATS_LEVEL`` to a value of 2 or higher enables more detailed collection:

.. code-block:: shell

   export HIPFILE_STATS_LEVEL=2

The exact additional detail collected at each level depends on the hipfile version. See :doc:`/conceptual/statistics-collection` for a description of each level.

Troubleshooting
***************

ais-stats cannot find the target process
   Verify that the PID is correct and the process is still running. The ``ais-stats`` tool returns an error if the target process has already exited or is not accessible.

No statistics in the report
   Confirm that ``HIPFILE_STATS_LEVEL`` was set to 1 or higher in the environment of the target process *before* it was launched. If the variable was not set or was set to 0, no data is collected.

Permission denied
   The ``ais-stats`` tool uses ``pidfd_open`` to connect to the target process. You must run ``ais-stats`` as the same user that owns the target process, or as root.
