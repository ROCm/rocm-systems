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
