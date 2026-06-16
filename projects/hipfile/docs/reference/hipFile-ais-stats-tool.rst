.. meta::
  :description: Reference documentation for the ais-stats command-line tool, which attaches to a live hipFile process and prints I/O statistics.
  :keywords: hipFile, ais-stats, statistics, I/O monitoring, ROCm, GPU I/O, bandwidth, latency, histogram

**********************************
hipFile ais-stats tool 
**********************************

The ``ais-stats`` tool is used to generate a report of I/O statistics of a running hipFile application.

The statistics gathered include the read size and write size for fastpath and fallback, the read bandwidth and write bandwidth for fastpath and fallback, the read latency and write latency for fastpath and fallback, and the read error count and write error count for fastpath and fallback.

Statistics are gathered and reported for each GPU.

Set ``HIPFILE_STATS_LEVEL=1``, launch the application, and pass the application's PID to ``ais-stats``:

.. code:: shell

   ais-stats -p PID

The report is generated after the process exists. Use the ``-i`` option to generate a report before the process exits.

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

Each subsequent bucket doubles the range until bucket 15 which covers 64 MiB and above
