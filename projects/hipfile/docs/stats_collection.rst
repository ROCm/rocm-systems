*********************
Stats collection tool
*********************

``ais-stats`` collects runtime hipFile I/O statistics from an application.
Use it for quick performance checks and backend-path validation.

Command-line tool
-----------------

``ais-stats`` can be run in two modes:

- ``ais-stats -p <PID> [-i]`` attaches to a running process. Pass ``-i`` to
  report immediately instead of waiting for process exit.
- ``ais-stats <program> [args...]`` launches ``<program>`` with the provided
  arguments and reports stats after waiting for the process to exit.

Quick start examples:

.. code:: shell

   export HIPFILE_STATS_LEVEL=1
   ais-stats -p 12345

   HIPFILE_STATS_LEVEL=1 ais-stats ./my_app --input data.bin

Configuration
-------------

Collection is controlled by the environment variable ``HIPFILE_STATS_LEVEL``.

===== =================================================
Value Description
===== =================================================
0     Histogram recording disabled
1     Basic, default when unset
2     Reserved for future detail levels. Same as level 1 today.
===== =================================================

Stats collected
---------------

At level ``1``, hipFile records the following per GPU, backend, and direction:

- Bytes read and written on the fastpath and fallback backends
- Bandwidth for reads and writes on each backend
- Latency for reads and writes on each backend
- I/O error counts for reads and writes on each backend
- Histograms of size, bandwidth, latency, and errors by I/O size bucket

Output
------

The report shows total counters for each metric, followed by histograms broken
down by I/O size. Report sections are I/O Size, I/O Bandwidth, I/O Latency, and I/O
Errors. I/O count data is collected internally but not printed as a separate
histogram section.

Example output shape:

.. code:: shell

   AIS-STATS Version: 1
   HipFile Stats Level: 1
   Total Fastpath Read Size (B): 67108864
   Average Fastpath Read Bandwidth (GiB/s): 0.776272
   Average Fastpath Read Latency (us): 1258.02
   Total Fastpath Read Errors: 0

Scope and limitations
---------------------

- Reported values reflect hipFile-managed I/O paths.
- Stats are broken out by backend to help identify fastpath and fallback usage.
- Level ``2`` and higher are reserved for future expansion.
- Attaching to a running process requires Linux kernel 5.3 or later for
  ``pidfd_open`` support.

Troubleshooting
---------------

- No report produced: make sure ``HIPFILE_STATS_LEVEL`` is set to ``1`` or
  higher, confirm ``ais-stats`` is available in ``PATH``, and confirm the
  target process performed hipFile I/O.
- ``-p <PID>`` attach fails: make sure the PID exists and that the current user
  has permission to inspect it.
- Unexpected zero values: confirm the workload exercised the expected backend
  and operation type.

Related documentation
---------------------

- See :doc:`/how-to/collect-io-statistics` for the canonical workflow.
- See :doc:`/how-to/hipFile-ais-stats-tool` for the full tool reference.
- See :doc:`fio` for a benchmark workflow that can generate I/O activity.
