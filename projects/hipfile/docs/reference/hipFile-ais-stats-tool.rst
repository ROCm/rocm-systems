.. meta::
   :description: Reference documentation for the ais-stats command-line tool, which attaches to a live hipFile process and prints I/O statistics.
   :keywords: hipFile, ais-stats, statistics, I/O monitoring, ROCm, GPU I/O, bandwidth, latency, histogram

***********************
hipFile ais-stats tool
***********************

The ``ais-stats`` tool generates a report of I/O statistics of a running hipFile application.

The statistics gathered include the read size and write size for fastpath and fallback, the read bandwidth and write bandwidth for fastpath and fallback, the read latency and write latency for fastpath and fallback, the read error count and write error count for fastpath and fallback, and the unaligned read count and unaligned write count for fallback. Fastpath does not support unaligned I/O.

Statistics are gathered and reported for each GPU.

.. note:: 

   Statistics collection is on by default. If statistics collection has been turned off, set ``HIPFILE_STATS_LEVEL=1`` before launching the application.

Launch the application and pass the application's PID to ``ais-stats``:

.. code:: shell

   ais-stats -p PID

The report is generated after the process exits. Use the ``-i`` option to generate a report before the process exits.

You can also launch an application with ``ais-stats``:

.. code:: shell

   ais-stats APPLICATION


.. note::

   Running with statistics collection enabled may have a small performance impact on I/O operations. Set ``HIPFILE_STATS_LEVEL=0`` to disable statistics collection.

The ``ais-stats`` report provides histogram data for I/O size, I/O latency, I/O time, and error count. Data is provided per GPU and per backend. Each metric is reported separately for read and write operations. Only GPUs that performed I/O operations appear in the report.

Histograms use logarithmic bucket boundaries:

- Bucket 0: 0 to 4 KiB
- Bucket 1: 4 KiB to 8 KiB
- Bucket 2: 8 KiB to 16 KiB

Each subsequent bucket doubles the range until bucket 15 which covers 64 MiB and above.

Example Output

.. code:: shell

   $ ais-stats aiscp somefile.txt /tmp/somefile.txt
   AIS-STATS Version: 1
   HipFile Stats Level: 1
   File Handle Registrations: 2
   Buffer Registrations: 0
   Fastpath Rejections: 1

   Total Fastpath Read Size (B): 2302
   Average Fastpath Read Bandwidth (GiB/s): 0.00155243
   Average Fastpath Read Latency (us): 1381
   Total Fastpath Read Errors: 0
   Total Fastpath Read Unaligned: 0

   Total Fastpath Write Size (B): 4096
   Average Fastpath Write Bandwidth (GiB/s): 0.0356514
   Average Fastpath Write Latency (us): 107
   Total Fastpath Write Errors: 0
   Total Fastpath Write Unaligned: 0

   Total Fallback Read Size (B): 0
   Average Fallback Read Bandwidth (GiB/s): 0
   Average Fallback Read Latency (us): 13
   Total Fallback Read Errors: 0
   Total Fallback Read Unaligned: 1

   Total Fallback Write Size (B): 0
   Average Fallback Write Bandwidth (GiB/s): 0
   Average Fallback Write Latency (us): 0
   Total Fallback Write Errors: 0
   Total Fallback Write Unaligned: 0

   GPU 0:
   IO Size Histogram
   IO Size (KiB)               Fastpath Read Size (B)           Fastpath Write Size (B)            Fallback Read Size (B)           Fallback Write Size (B)
   0-4                                           2302                                 0                                 0                                 0
   4-8                                              0                              4096                                 0                                 0
   8-16                                             0                                 0                                 0                                 0
   16-32                                            0                                 0                                 0                                 0
   32-64                                            0                                 0                                 0                                 0
   64-128                                           0                                 0                                 0                                 0
   128-256                                          0                                 0                                 0                                 0
   256-512                                          0                                 0                                 0                                 0
   512-1024                                         0                                 0                                 0                                 0
   1024-2048                                        0                                 0                                 0                                 0
   2048-4096                                        0                                 0                                 0                                 0
   4096-8192                                        0                                 0                                 0                                 0
   8192-16384                                       0                                 0                                 0                                 0
   16384-32768                                      0                                 0                                 0                                 0
   32768-65536                                      0                                 0                                 0                                 0
   65536-...                                        0                                 0                                 0                                 0
   IO Bandwidth Histogram
   IO Size (KiB)      Fastpath Read Bandwidth (GiB/s)  Fastpath Write Bandwidth (GiB/s)   Fallback Read Bandwidth (GiB/s)  Fallback Write Bandwidth (GiB/s)
   0-4                                     0.00155243                                 0                                 0                                 0
   ...                                 0
   65536-...                                        0                                 0                                 0                                 0
   IO Latency Histogram
   IO Size (KiB)           Fastpath Read Latency (us)       Fastpath Write Latency (us)        Fallback Read Latency (us)       Fallback Write Latency (us)
   0-4                                           1381                                 0                                13                                 0
   ...                                 0
   65536-...                                        0                                 0                                 0                                 0
   IO Errors Histogram
   IO Size (KiB)            Fastpath Read Error Count        Fastpath Write Error Count         Fallback Read Error Count        Fallback Write Error Count
   0-4                                              0                                 0                                 0                                 0
   ...                                0
   65536-...                                        0                                 0                                 0                                 0
   Fastpath Read Unaligned Count: 0
   Fastpath Write Unaligned Count: 0
   Fallback Read Unaligned Count: 1
   Fallback Write Unaligned Count: 0