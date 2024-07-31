.. meta::
   :description: How to use Omniperf's analyze mode
   :keywords: Omniperf, ROCm, profiler, tool, Instinct, accelerator, AMD,
              Grafana, analysis, analyze mode

************
Analyze mode
************

Omniperf offers several ways to interact with the metrics it generates from
profiling. Your level of familiarity with the profiled application, computing
environment, and experience with Omniperf should inform the analysis method you
choose.

While analyzing with the CLI offers quick and straightforward access to Omniperf
metrics from the terminal, Grafana's dashboard GUI adds an extra layer of
readability and interactivity you might prefer.

See the following sections to explore Omniperf's analysis and visualization
options.

* :doc:`cli`
* :doc:`grafana-gui`
* :doc:`standalone-gui`

.. note::

   Analysis examples in this chapter borrow profiling results from the
   ``vcopy.cpp`` workload introduced in :ref:`profile-example` in the
   previous chapter.

   Unless otherwise noted, the performance analysis is done on the
   :ref:`MI200 platform <def-soc>`.

Learn about profiling with Omniperf in :doc:`../profile/mode`. For an overview of
Omniperf's other modes, see :ref:`modes`.
