.. meta::
   :description: How to use ROCm Compute Profiler's analyze mode
   :keywords: ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, AMD,
              analysis, analyze mode

************
Analyze mode
************

ROCm Compute Profiler offers several ways to interact with the metrics it generates from
profiling. Your level of familiarity with the profiled application, computing
environment, and experience with ROCm Compute Profiler should inform the analysis method you
choose.

.. note::

   Analyze mode requires Python 3.9 or newer; its dependencies (numpy, pandas,
   dash, textual) drop support for older versions. Profile mode runs on Python
   3.8+. See the Python version support table in :doc:`/install/quickstart`.

.. note::

   Analyze mode builds ``pmc_perf.csv`` by concatenating per-pass
   ``results_*.csv`` files from ``rocpd`` profiling, or uses an existing
   ``pmc_perf.csv`` as-is. Current releases read ``.csv`` and ``.csv.gz``
   counter files alike, so older workloads do need not to be regenerated. Releases
   before 3.9.0 read only uncompressed names; run ``gunzip`` on ``.csv.gz``
   files to analyze a compressed workload on an older release.

.. note::

   Reading intermediate ``results_*.csv`` files produced by ``rocpd`` profiling is
   deprecated and will be removed in a future release. The analyze step will read ``.db``
   files directly.

See the following sections to explore ROCm Compute Profiler's analysis and visualization
options.

* :doc:`cli`
* :doc:`standalone-gui` (experimental feature)
* :doc:`tui` (experimental feature)
* :doc:`optiq` (graphical application)

.. note::

   Analysis examples in this chapter borrow profiling results from the
   ``vcopy.cpp`` workload introduced in :ref:`profile-example` in the
   previous chapter.

   Unless otherwise noted, the performance analysis is done on the
   :ref:`MI200 platform <def-soc>`.

Learn about profiling with ROCm Compute Profiler in :doc:`../profile/mode`. For an overview of
ROCm Compute Profiler's other modes, see :ref:`modes`.
