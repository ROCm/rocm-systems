.. meta::
   :description: ROCm Compute Profiler performance model
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, performance, model, profiler, tool, Instinct,
              accelerator, AMD, CDNA, RDNA

*****************
Performance model
*****************

ROCm Compute Profiler makes available an extensive list of metrics to better understand
achieved application performance on AMD Instinct™ MI-series accelerators
including Graphics Core Next™ (GCN) GPUs like the AMD Instinct MI50, CDNA™
accelerators like the MI100, CDNA2 accelerators such as the AMD Instinct MI250X, MI250,
and MI210, CDNA3 accelerators such as the AMD Instinct MI300A, MI300X, MI325X, and CDNA4 accelerators such as MI350X and MI355X.

The same tool also ships analysis panels for select **AMD Radeon™ / RDNA™**
configurations (for example **RDNA3.5** as **gfx1151**). Public architecture summaries and
:doc:`GPU specifications <rocm:reference/gpu-arch-specs>` are the right place for
generational comparisons (including **RDNA4** products); the chapters below focus on how
the profiler presents metrics for each hardware family.

Architecture-specific chapters
==============================

* :doc:`CDNA / Instinct performance model <conceptual/cdna/cdna-performance-model>` — comparison tables for
  CDNA through CDNA4, **tabbed die diagrams** (CDNA, CDNA2, CDNA3, CDNA4), and nested pages for
  compute unit, L2, shader engine, command processor, System Speed-of-Light, and references,
  all under ``conceptual/cdna/``.

* :doc:`RDNA client GPU performance model <conceptual/rdna/rdna-performance-model>` — Radeon / RDNA profiler
  context with **tabs by generation** (**RDNA3.5 (gfx1151)** today; **RDNA4** placeholder).
  Full metric text: :ref:`RDNA3.5 (gfx1151) metrics <gfx1151-metrics-ref>`. **System Speed-of-Light** is split by
  family: :doc:`CDNA / Instinct <conceptual/cdna/system-speed-of-light>` and :doc:`RDNA (gfx1151) <conceptual/rdna/system-speed-of-light>`.
