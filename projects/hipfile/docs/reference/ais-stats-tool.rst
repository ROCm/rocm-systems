.. meta::
   :description: Reference documentation for the ais-stats command-line tool, which attaches to a live hipfile process and prints IO statistics.
   :keywords: hipfile, ais-stats, statistics, IO monitoring, ROCm, GPU IO, bandwidth, latency, histogram

========================
ais-stats tool reference
========================

The ``ais-stats`` tool attaches to a live hipfile process by process ID (PID) and prints IO statistics collected during that process's execution. It uses the internal hipfile statistics collection API to connect to the target process, poll for IO activity, and generate a formatted report covering per-GPU and per-backend metrics.

For background on how hipfile collects statistics internally, see :doc:`/conceptual/statistics-collection`.

Purpose
*******

Use ``ais-stats`` to monitor the IO behavior of a running hipfile application without modifying the application itself. The tool connects to the shared-memory statistics region of the target process and produces a human-readable report containing:

- The statistics format version
- The configured statistics level
- Per-backend (fastpath and fallback) and per-IO-direction (read and write) totals
- Per-GPU histograms for IO size, bandwidth, latency, and errors

.. note::

   The target hipfile process must have statistics collection enabled. Set the ``HIPFILE_STATS_LEVEL`` environment variable to ``1`` or higher before launching the target process. A value of ``0`` disables statistics collection.

Usage syntax
************

.. code-block:: bash

   ais-stats <pid>

:param pid: The process ID of the running hipfile process to monitor.

The tool performs the following steps:

1. Creates a statistics context for the specified PID.
2. Connects to the target process's shared-memory statistics region.
3. Polls the target process (blocking until the process completes or statistics are available).
4. Generates a formatted report and writes it to standard output.

Statistics collection API
*************************

The ``ais-stats`` tool is built on the internal statistics collection C API defined in ``hipfile-stats.h``. The functions and types in this API are documented below.

Types
-----

``hipFileStatsContext_t``
   Opaque pointer to a statistics collection context. Created by ``hipFileStatsCreateContext`` and freed by ``hipFileStatsCloseContext``.

``hipFileStatsError_t``
   Error codes returned by statistics collection API functions.

   .. code-block:: c

      typedef enum hipFileStatsError {
          hipFileStatsSuccess,
          hipFileStatsInvalidArgument,
          hipFileStatsTargetProcessNotFound,
          hipFileStatsTargetProcessNotAccessible,
          hipFileStatsReportGenerationFailed,
      } hipFileStatsError_t;

   - ``hipFileStatsSuccess`` — Operation completed successfully.
   - ``hipFileStatsInvalidArgument`` — An invalid argument was passed to the function.
   - ``hipFileStatsTargetProcessNotFound`` — The target process with the given PID was not found.
   - ``hipFileStatsTargetProcessNotAccessible`` — Cannot access the target process.
   - ``hipFileStatsReportGenerationFailed`` — Failed to generate or write the report.

Functions
---------

``hipFileStatsCreateContext``
   Create a new statistics collection context for a target process.

   .. code-block:: c

      hipFileStatsError_t hipFileStatsCreateContext(hipFileStatsContext_t **context, int targetPid)

   :param context: Pointer to store the created context handle.
   :param targetPid: Process ID of the target process to monitor.
   :returns: ``hipFileStatsSuccess`` on success, or an error code otherwise.

   The returned context must be freed with ``hipFileStatsCloseContext``.

``hipFileStatsCloseContext``
   Close and free a statistics collection context.

   .. code-block:: c

      void hipFileStatsCloseContext(hipFileStatsContext_t *context)

   :param context: Statistics context handle to close. May be ``NULL``.

   Releases all resources associated with the context. Safe to call with a ``NULL`` pointer.

``hipFileStatsConnectToTargetProcess``
   Establish a connection to the target process for statistics collection.

   .. code-block:: c

      hipFileStatsError_t hipFileStatsConnectToTargetProcess(hipFileStatsContext_t *context)

   :param context: Statistics context handle.
   :returns: ``hipFileStatsSuccess`` on success, or an error code otherwise.

   You must call this function before calling ``hipFileStatsGenerateReport``.

``hipFileStatsPollTargetProcess``
   Poll the target process for updated statistics.

   .. code-block:: c

      hipFileStatsError_t hipFileStatsPollTargetProcess(const hipFileStatsContext_t *context, bool block)

   :param context: Statistics context handle.
   :param block: If ``true``, block indefinitely until the target process completes.
   :returns: ``hipFileStatsSuccess`` on success, or an error code otherwise.

``hipFileStatsGenerateReport``
   Generate a formatted statistics report and write it to a file descriptor.

   .. code-block:: c

      hipFileStatsError_t hipFileStatsGenerateReport(const hipFileStatsContext_t *context, int fd)

   :param context: Statistics context handle.
   :param fd: File descriptor to write the report to.
   :returns: ``hipFileStatsSuccess`` on success, or an error code otherwise.

Output format
*************

The report generated by ``ais-stats`` contains the following sections:

Version and statistics level
----------------------------

The report header includes:

- **Version** — The statistics data format version (for example, version 1).
- **Stats level** — The statistics detail level configured in the target process, corresponding to the ``HIPFILE_STATS_LEVEL`` environment variable setting.

Per-backend and per-direction totals
------------------------------------

Totals are broken down by:

- **Backend** — Either *fastpath* (the accelerated path using HIP runtime extensions) or *fallback* (the POSIX IO path).
- **IO direction** — Read or write.

Per-GPU histograms
------------------

For each GPU that performed IO during the target process's lifetime, the report includes histograms for:

- **IO size** — Distribution of individual IO operation sizes in bytes. The histogram uses logarithmic buckets starting at 4 KiB (2\ :sup:`12` bytes), with 16 buckets total.
- **Bandwidth** — Derived from IO size and latency measurements.
- **Latency** — Distribution of IO operation durations in microseconds.
- **Errors** — Distribution of error counts by IO size.

Each histogram contains up to 16 logarithmic buckets. The first bucket covers sizes from 0 to 4 KiB. Subsequent buckets double in range, and the final bucket captures all values above the second-to-last bucket boundary.

Enabling statistics collection
******************************

To use ``ais-stats``, you must enable statistics collection in the target hipfile process by setting the ``HIPFILE_STATS_LEVEL`` environment variable before launching the process:

.. code-block:: bash

   export HIPFILE_STATS_LEVEL=1

The supported levels are:

- ``0`` — Statistics collection is disabled (default).
- ``1`` or higher — Basic statistics are collected.

.. warning::

   Running with statistics collection enabled may have a small performance impact on IO operations because of the additional timing and counting overhead.

Related pages
*************

- :doc:`/conceptual/statistics-collection` — Conceptual overview of how hipfile collects and stores IO statistics.
