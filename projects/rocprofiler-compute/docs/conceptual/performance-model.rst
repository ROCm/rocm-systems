.. meta::
   :description: ROCm Compute Profiler performance model
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, performance, model, profiler, tool, Instinct,
              accelerator, AMD

*****************
Performance model
*****************

ROCm Compute Profiler makes available an extensive list of metrics to better understand
achieved application performance on AMD Instinct™ MI-series accelerators
including Graphics Core Next™ (GCN) GPUs like the AMD Instinct MI50, CDNA™
accelerators like the MI100, and CDNA2 accelerators such as the MI250X, MI250,
and MI210.

To best use profiling data, it's important to understand the role of various
hardware blocks of AMD Instinct accelerators. This section describes each
hardware block on the accelerator as interacted with by a software developer to
give a deeper understanding of the metrics reported by profiling data. Refer to
:doc:`/tutorial/profiling-by-example` for more practical examples and details on how
to use ROCm Compute Profiler to optimize your code.

.. _mixxx-note:

.. note::

   In this chapter, **MI2XX** refers to any of the CDNA2 architecture-based AMD
   Instinct MI250X, MI250, and MI210 accelerators interchangeably in cases
   where the exact product at hand is not relevant.

   For a comparison of AMD Instinct accelerator specifications, refer to
   :doc:`Hardware specifications <rocm:reference/gpu-arch-specs>`. For product
   details, see the :prod-page:`MI250X <mi200/mi250x>`,
   :prod-page:`MI250 <mi200/mi250>`, and :prod-page:`MI210 <mi200/mi210>`
   product pages.

In this chapter, the AMD Instinct performance model used by ROCm Compute Profiler is divided into a handful of
key hardware blocks, each detailed in the following sections:

* :doc:`compute-unit`

* :doc:`l2-cache`

* :doc:`shader-engine`

* :doc:`command-processor`

* :doc:`system-speed-of-light`