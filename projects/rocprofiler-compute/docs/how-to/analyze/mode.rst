.. meta::
   :description: How to use ROCm Compute Profiler's analyze mode
   :keywords: ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, AMD,
              Grafana, analysis, analyze mode

************
Analyze mode
************

ROCm Compute Profiler offers several ways to interact with the metrics it generates from
profiling. Your level of familiarity with the profiled application, computing
environment, and experience with ROCm Compute Profiler should inform the analysis method you
choose.

While analyzing with the CLI offers quick and straightforward access to ROCm Compute Profiler
metrics from the terminal, Grafana's dashboard GUI adds an extra layer of
readability and interactivity you might prefer.

See the following sections to explore ROCm Compute Profiler's analysis and visualization
options.

* :doc:`cli`
* :doc:`grafana-gui`
* :doc:`standalone-gui`
* :doc:`text-based user interface (TUI)`

.. note::

   Analysis examples in this chapter borrow profiling results from the
   ``vcopy.cpp`` workload introduced in :ref:`profile-example` in the
   previous chapter.

   Unless otherwise noted, the performance analysis is done on the
   :ref:`MI200 platform <def-soc>`.

Learn about profiling with ROCm Compute Profiler in :doc:`../profile/mode`. For an overview of
ROCm Compute Profiler's other modes, see :ref:`modes`.
