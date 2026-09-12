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

   Analyze mode needs a Python version and a set of third-party packages that
   profile mode does not. Install them in their own virtual environment. See
   the Python version support table in :doc:`/install/quickstart`.

.. note::

   Analyze mode concatenates the per-pass ``results_*.csv.gz`` files written by
   ``rocpd`` profiling into a unified ``pmc_perf.csv.gz`` for analysis. If the
   workload directory already contains a ``pmc_perf.csv.gz``, that file is used
   as-is.

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
