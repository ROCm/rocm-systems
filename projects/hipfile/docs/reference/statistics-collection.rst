.. meta::
  :description: How hipFile collects per-GPU I/O statistics using histograms, shared memory, and Unix domain sockets, and how to inspect them with ais-stats.
  :keywords: hipFile, statistics, I/O stats, histogram, shared memory, memfd, Unix domain socket, ais-stats, HIPFILE_STATS_LEVEL, ROCm

*************************
I/O statistics collection
*************************

hipFile collects runtime I/O statistics to help you understand I/O behavior across GPUs and backends. This page explains the data model, the shared-memory transport that exposes statistics to external tools, the ``HIPFILE_STATS_LEVEL`` environment variable, and the ``ais-stats`` tool.

Histogram data model
********************

hipFile organizes statistics along three independent dimensions:

GPU slot
   Up to 16 GPU slots are supported. Each slot corresponds to a GPU device index. A slot is marked in use when any backend records I/O activity for that device.

Backend
   Two backends are tracked independently: fastpath and fallback. Every I/O operation is attributed to exactly one backend.

Direction
   Each backend tracks read and write directions separately.

For every combination of GPU, backend, and direction, hipFile maintains four histograms:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Histogram
     - Description
   * - I/O size (bytes)
     - Distribution of transfer sizes across bucket ranges
   * - I/O count
     - Number of I/O operations that fell into each size bucket
   * - I/O time (microseconds)
     - Latency distribution of completed I/O operations
   * - Error count
     - Distribution of failed I/O operations by size

Each histogram contains 16 buckets. Bucket boundaries are determined by a logarithmic scheme: the first bucket covers sizes from 0 up to 2\ :sup:`12` (4 KiB), and each subsequent bucket doubles in range. The final bucket captures all values equal to or larger than the lower bound of bucket 15. The ``toHistogramBucket`` function right-shifts the byte count by 12 bits and then computes ``floor(log2)``, clamping to the maximum bucket index.

The per-GPU statistics structure (``PerGpuStatsV1``) is a standard-layout type with atomic counters so that the producing process and a consuming tool can read the data concurrently without locks.

.. mermaid::

   flowchart LR
       I/O["I/O operation"] --> GPU["GPU slot (0–15)"]
       GPU --> BE["Backend\n(Fastpath | Fallback)"]
       BE --> DIR["Direction\n(Read | Write)"]
       DIR --> H1["I/O size histogram"]
       DIR --> H2["I/O count histogram"]
       DIR --> H3["I/O time histogram"]
       DIR --> H4["Error count histogram"]

Recording an I/O event
---------------------

When an I/O operation completes, a ``StatsIoTracker`` records the elapsed wall-clock time (measured from construction to the ``complete`` call) and the number of bytes transferred. The ``StatsCollection::addIo`` method increments the appropriate histograms. If the operation fails, ``StatsCollection::error`` increments the error-count histogram instead.

Shared-memory transport
***********************

hipFile uses a ``memfd_create`` anonymous file and ``mmap`` to allocate a shared-memory region that holds the entire ``Stats`` structure (version field, stats level, and the array of per-GPU, per-backend statistics). Because the structure uses only atomic counters and is standard-layout, it can be safely read by an external process that maps the same memory.

The transport works as follows:

1. StatsServer: Created inside the hipFile library when the driver initializes. It allocates the ``StatsContainer`` (which calls ``memfd_create`` and ``mmap``), then starts a background thread that listens on a Unix domain socket in the abstract namespace.

2. StatsClient: Lives in an external monitoring process (such as ``ais-stats``). It opens a ``pidfd`` for the target process and connects to the server's Unix domain socket. The server sends the ``memfd`` file descriptor over the socket using ``SCM_RIGHTS`` ancillary data, and the client maps it into its own address space.

3. Once the mapping is established, the client reads statistics directly from the shared memory region without further socket communication.

The ``StatsContainer`` owns the ``memfd`` file descriptor and the ``mmap`` pointer. On destruction it unmaps and closes the file descriptor, ensuring cleanup even if the process exits unexpectedly.

HIPFILE_STATS_LEVEL
*******************

The ``HIPFILE_STATS_LEVEL`` environment variable controls how much data hipFile collects:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Value
     - Behavior
   * - ``0``
     - Statistics recording is disabled. The shared-memory region and statistics server still initialize, but histogram counters are not updated.
   * - ``1``, default
     - Basic statistics are collected. I/O size, count, latency, and error histograms are populated for every I/O operation.
   * - ``2`` or higher
     - Reserved for future detail levels. Current releases collect the same histograms as level ``1``.

The level is read once during library initialization and stored in the ``Stats`` structure so that the client can see which level was active.

For the complete list of hipFile environment variables, see :doc:`/reference/hipFile-environment-variables`.

The ais-stats tool
******************

``ais-stats`` is a command-line utility that attaches to a live hipFile process and prints a formatted statistics report. It uses the ``StatsClient`` class and the C API defined in the internal statistics header.

The typical workflow is:

1. Create a context: ``hipFileStatsCreateContext()`` takes the PID of the target process.
2. Connect: ``hipFileStatsConnectToTargetProcess()`` opens a ``pidfd`` for the target, connects to its Unix domain socket, receives the ``memfd`` file descriptor, and maps the shared-memory region.
3. Poll: ``hipFileStatsPollTargetProcess()`` can optionally block until the target process exits, to make sure all I/O has completed.
4. Generate report: ``hipFileStatsGenerateReport()`` writes a human-readable report to a file descriptor, typically ``stdout``).
5. Close: ``hipFileStatsCloseContext()`` releases all resources.

Because the client reads from shared memory, it imposes negligible overhead on the monitored process.

For usage details and command-line options, see :doc:`/how-to/hipFile-ais-stats-tool`.

Statistics C API
----------------

The internal statistics header exposes five functions that ``ais-stats`` and other custom tooling can use:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Function
     - Purpose
   * - ``hipFileStatsCreateContext``
     - Create a stats context for a target PID
   * - ``hipFileStatsCloseContext``
     - Free a stats context (safe to call with ``NULL``)
   * - ``hipFileStatsConnectToTargetProcess``
     - Connect to the target and map its stats region
   * - ``hipFileStatsPollTargetProcess``
     - Poll or block-wait for the target process
   * - ``hipFileStatsGenerateReport``
     - Write a formatted report to a file descriptor

Each function returns a ``hipFileStatsError_t`` indicating success or a specific error such as ``hipFileStatsTargetProcessNotFound`` or ``hipFileStatsTargetProcessNotAccessible``.
