Stats Collection Tool
=====================

``ais-stats`` collects runtime hipFile I/O statistics from an application.
Use it for quick performance checks and backend-path validation.

Command-line Tool
-----------------
``ais-stats`` can be run in two modes:
* ``$ ais-stats -p <PID> [-i]`` collects stats from a running process.
  ``-i`` reports immediately rather than waiting for process exit.
* ``$ ais-stats <program> [args...]`` launches ``<program>`` with the
  provided arguments and reports stats after waiting for the process to exit.

Quick start examples: ::

    $ export HIPFILE_STATS_LEVEL=1
    $ ais-stats -p 12345

    $ HIPFILE_STATS_LEVEL=1 ais-stats ./my_app --input data.bin

Configuration
-------------
Collection is controlled by the environment variable ``HIPFILE_STATS_LEVEL``.

===== =================================================
Value Description
===== =================================================
0     Disabled
1     Basic (default value)
2     Detailed (currently same as basic; reserved for future use)
===== =================================================

Stats Collected
---------------
* Basic: Number of file handle registrations.
* Basic: Number of buffer registrations.
* Basic: Bytes read/written on the fastpath/fallback backends.
* Basic: Bandwidth for reads/writes on the fastpath/fallback backends.
* Basic: Latency for reads/writes on the fastpath/fallback backends.
* Basic: I/O error counts for reads/writes on the fastpath/fallback backends.
* Basic: Histograms of above stats broken into buckets based on the size of the I/O.

Output
------
The report shows total counters for each metric, followed by histograms of the metrics broken down by I/O size.

Example output shape: ::

    AIS-STATS Version: 1
    HipFile Stats Level: 1
    File Handle Registrations: 1
    Buffer Registrations: 1

    Total Fastpath Read Size (B): 67108864
    Average Fastpath Read Bandwidth (GiB/s): 0.776272
    Average Fastpath Read Latency (us): 1258.02
    Total Fastpath Read Errors: 0

    Total Fastpath Write Size (B): 67108864
    Average Fastpath Write Bandwidth (GiB/s): 0.358882
    Average Fastpath Write Latency (us): 2721.12
    Total Fastpath Write Errors: 0

    Total Fallback Read Size (B): 0
    Average Fallback Read Bandwidth (GiB/s): 0
    Average Fallback Read Latency (us): 0
    Total Fallback Read Errors: 0

    Total Fallback Write Size (B): 0
    Average Fallback Write Bandwidth (GiB/s): 0
    Average Fallback Write Latency (us): 0
    Total Fallback Write Errors: 0

    GPU 0:
    IO Size Histogram
    IO Size (KiB)               Fastpath Read Size (B)           Fastpath Write Size (B)            Fallback Read Size (B)           Fallback Write Size (B)
    0-4                                              0                                 0                                 0                                 0
    4-8                                              0                                 0                                 0                                 0
    8-16                                             0                                 0                                 0                                 0
    16-32                                            0                                 0                                 0                                 0
    32-64                                            0                                 0                                 0                                 0
    64-128                                           0                                 0                                 0                                 0
    128-256                                          0                                 0                                 0                                 0
    256-512                                          0                                 0                                 0                                 0
    512-1024                                         0                                 0                                 0                                 0
    1024-2048                                 67108864                          67108864                                 0                                 0
    2048-4096                                        0                                 0                                 0                                 0
    4096-8192                                        0                                 0                                 0                                 0
    8192-16384                                       0                                 0                                 0                                 0
    16384-32768                                      0                                 0                                 0                                 0
    32768-65536                                      0                                 0                                 0                                 0
    65536-...                                        0                                 0                                 0                                 0
    IO Bandwidth Histogram
    IO Size (KiB)      Fastpath Read Bandwidth (GiB/s)  Fastpath Write Bandwidth (GiB/s)   Fallback Read Bandwidth (GiB/s)  Fallback Write Bandwidth (GiB/s)
    0-4                                              0                                 0                                 0                                 0
    ...
    65536-...                                        0                                 0                                 0                                 0
    IO Latency Histogram
    IO Size (KiB)           Fastpath Read Latency (us)       Fastpath Write Latency (us)        Fallback Read Latency (us)       Fallback Write Latency (us)
    0-4                                              0                                 0                                 0                                 0
    ...
    65536-...                                        0                                 0                                 0                                 0
    IO Errors Histogram
    IO Size (KiB)            Fastpath Read Error Count        Fastpath Write Error Count         Fallback Read Error Count        Fallback Write Error Count
    0-4                                              0                                 0                                 0                                 0
    ...
    65536-...                                        0                                 0                                 0                                 0

Scope and Limitations
---------------------
* Reported values reflect hipFile-managed I/O paths.
* Stats are broken out by backend to help identify fastpath/fallback usage.
* Detailed mode (``HIPFILE_STATS_LEVEL=2``) is reserved for future expansion.
* Stats collection is only available on systems with a Linux kernel version 5.3 or later.

Troubleshooting
---------------
* No report produced: ensure ``HIPFILE_STATS_LEVEL`` is set to ``1`` or ``2``,
  verify ``ais-stats`` is available in ``PATH``, and confirm the target process
  performed hipFile I/O.
* ``-p <PID>`` attach fails: verify the PID exists and that the current user
  has permission to inspect it.
* Unexpected zero values: confirm the workload exercised the expected backend
  and operation type.

Related documentation
---------------------
* See :doc:`fio` for a benchmark workflow that can generate I/O activity.
* See the project ``README.md`` for build and runtime setup details.
