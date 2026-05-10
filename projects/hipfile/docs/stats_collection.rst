Stats Collection Tool
=====================

Command-line Tool
-----------------
``ais-stats`` can be run two ways:
* ``$ ais-stats -p <PID> [-i]`` will collect stats from a running process. ``-i`` will report immediately rather than wait for the process to exit.
* ``$ ais-stats <program> [args...]`` will launch ``<program>`` with the provided arguments and report the collected stats when it exits.

Configuration
-------------
Collection is controlled by the environment variable ``HIPFILE_STATS_LEVEL``.

===== =================================================
Value Description
===== =================================================
0     Disabled
1     Basic (default value)
2     Detailed (same as basic; reserved for future use)
===== =================================================

Stats Collected
---------------
* Basic: Bytes read/written on the fastpath/fallback backends.
* Basic: Bandwidth for reads/writes on the fastpath/fallback backends.
* Basic: Latency for reads/writes on the fastpath/fallback backends.
* Basic: Histograms of above stats broken into buckets based on the size of the I/O.
