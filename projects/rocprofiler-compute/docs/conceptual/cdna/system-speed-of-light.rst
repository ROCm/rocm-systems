.. meta::
   :description: Learn about the System Speed‑of‑Light metrics for AMD Instinct MI GPUs using CDNA–CDNA4 architectures with ROCm Compute Profiler.
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, AMD, system, speed of light

.. _sys-sol:

*********************
System Speed-of-Light
*********************

The page lists the System Speed-of-Light metrics from various sections
of the ROCm Compute Profiler’s profiling report for AMD Instinct™ MI-series
GPUs based on CDNA-CDNA4 architectures.

For RDNA3.5 (gfx115x) APUs, see :doc:`../rdna/system-speed-of-light` under RDNA3.

.. warning::

   The theoretical maximum throughput for some metrics in this section are
   currently computed with the maximum achievable clock frequency, as reported
   by ``rocminfo``, for an accelerator. This may not be realistic for
   all workloads.

   Also, not all metrics -- such as FLOP counters -- are available on all AMD
   Instinct™ MI-series accelerators. For more detail on how operations are
   counted, see the :ref:`metrics-flop-count` section.

.. tab-set::

   .. tab-item:: CDNA

      .. jinja:: sys-sol-gfx908
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 2

      .. jinja:: sys-sol-gfx90a
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 3

      .. jinja:: sys-sol-gfx942
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 4
      :selected:

      .. jinja:: sys-sol-gfx950
         :file: _templates/metrics_table.j2
