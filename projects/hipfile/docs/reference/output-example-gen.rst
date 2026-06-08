.. meta::
  :description: Example stdout report from the ais-stats tool showing header lines, aggregate totals, and per-GPU histogram tables.
  :keywords: hipFile, ais-stats, statistics, example output, histogram, ROCm

********************************************************************
ais-stats example report
********************************************************************

The following sample is illustrative stdout from ``ais-stats -p <PID>`` after a hipFile workload on GPU 0. Numeric values are fictional. Column alignment matches the formatted tables the tool prints. Report labels use ``IO``, not ``I/O``, because that is what the tool emits.

.. code-block:: text

   AIS-STATS Version: 1
   HipFile Stats Level: 1
   Total Fastpath Read Size (B): 268435456
   Average Fastpath Read Bandwidth (GiB/s): 8.42
   Average Fastpath Read Latency (us): 52.3
   Total Fastpath Read Errors: 0

   Total Fastpath Write Size (B): 134217728
   Average Fastpath Write Bandwidth (GiB/s): 6.15
   Average Fastpath Write Latency (us): 48.7
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
   IO Size (KiB)    Fastpath Read Size (B)          Fastpath Write Size (B)         Fallback Read Size (B)          Fallback Write Size (B)
   0-4              0                               0                               0                               0
   4-8              0                               0                               0                               0
   8-16             0                               0                               0                               0
   16-32            0                               0                               0                               0
   32-64            0                               0                               0                               0
   64-128           268435456                       134217728                       0                               0
   128-256          0                               0                               0                               0
   256-512          0                               0                               0                               0
   512-1024         0                               0                               0                               0
   1024-2048        0                               0                               0                               0
   2048-4096        0                               0                               0                               0
   4096-8192        0                               0                               0                               0
   8192-16384       0                               0                               0                               0
   16384-32768      0                               0                               0                               0
   32768-65536      0                               0                               0                               0
   65536-...        0                               0                               0                               0
   IO Bandwidth Histogram
   IO Size (KiB)    Fastpath Read Bandwidth (GiB/s) Fastpath Write Bandwidth (GiB/s)Fallback Read Bandwidth (GiB/s) Fallback Write Bandwidth (GiB/s)
   0-4              0                               0                               0                               0
   4-8              0                               0                               0                               0
   8-16             0                               0                               0                               0
   16-32            0                               0                               0                               0
   32-64            0                               0                               0                               0
   64-128           8.42                            6.15                            0                               0
   128-256          0                               0                               0                               0
   256-512          0                               0                               0                               0
   512-1024         0                               0                               0                               0
   1024-2048        0                               0                               0                               0
   2048-4096        0                               0                               0                               0
   4096-8192        0                               0                               0                               0
   8192-16384       0                               0                               0                               0
   16384-32768      0                               0                               0                               0
   32768-65536      0                               0                               0                               0
   65536-...        0                               0                               0                               0
   IO Latency Histogram
   IO Size (KiB)    Fastpath Read Latency (us)      Fastpath Write Latency (us)     Fallback Read Latency (us)      Fallback Write Latency (us)
   0-4              0                               0                               0                               0
   4-8              0                               0                               0                               0
   8-16             0                               0                               0                               0
   16-32            0                               0                               0                               0
   32-64            0                               0                               0                               0
   64-128           52.3                            48.7                            0                               0
   128-256          0                               0                               0                               0
   256-512          0                               0                               0                               0
   512-1024         0                               0                               0                               0
   1024-2048        0                               0                               0                               0
   2048-4096        0                               0                               0                               0
   4096-8192        0                               0                               0                               0
   8192-16384       0                               0                               0                               0
   16384-32768      0                               0                               0                               0
   32768-65536      0                               0                               0                               0
   65536-...        0                               0                               0                               0
   IO Errors Histogram
   IO Size (KiB)    Fastpath Read Error Count       Fastpath Write Error Count      Fallback Read Error Count       Fallback Write Error Count
   0-4              0                               0                               0                               0
   4-8              0                               0                               0                               0
   8-16             0                               0                               0                               0
   16-32            0                               0                               0                               0
   32-64            0                               0                               0                               0
   64-128           0                               0                               0                               0
   128-256          0                               0                               0                               0
   256-512          0                               0                               0                               0
   512-1024         0                               0                               0                               0
   1024-2048        0                               0                               0                               0
   2048-4096        0                               0                               0                               0
   4096-8192        0                               0                               0                               0
   8192-16384       0                               0                               0                               0
   16384-32768      0                               0                               0                               0
   32768-65536      0                               0                               0                               0
   65536-...        0                               0                               0                               0

Reading the report
******************

The header lines identify the stats format version and the ``HIPFILE_STATS_LEVEL`` value in the target process. The eight total blocks summarize bytes moved, average bandwidth, average latency, and error counts for each backend and direction, aggregated across all GPUs.

Each active GPU adds four histogram tables. Rows are I/O size buckets from 0–4 KiB through 64 MiB and above. Columns are always ordered as **Fastpath Read**, **Fastpath Write**, **Fallback Read**, and **Fallback Write**. GPUs with no recorded I/O are omitted entirely.

For field definitions and bucket boundaries, see :doc:`/how-to/ais-stats-tool`.
