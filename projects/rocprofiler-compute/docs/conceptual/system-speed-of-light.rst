.. meta::
   :description: ROCm Compute Profiler performance model: System Speed-of-Light
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, AMD, system, speed of light

.. _sys-sol:

*********************
System Speed-of-Light
*********************

System Speed-of-Light summarizes some of the key metrics from various sections
of ROCm Compute Profiler’s profiling report.

.. warning::

   The theoretical maximum throughput for some metrics in this section are
   currently computed with the maximum achievable clock frequency, as reported
   by ``rocminfo``, for an accelerator. This may not be realistic for
   all workloads.

   Also, not all metrics -- such as FLOP counters -- are available on all AMD
   Instinct™ MI-series accelerators. For more detail on how operations are
   counted, see the :ref:`metrics-flop-count` section.

.. jinja:: sys-sol
   :file: _templates/metrics_table.j2
