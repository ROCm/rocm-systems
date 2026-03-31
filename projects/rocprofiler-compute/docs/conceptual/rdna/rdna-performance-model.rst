.. meta::
   :description: ROCm Compute Profiler RDNA client GPU performance model
   :keywords: ROCm Compute Profiler, RDNA, RDNA3, gfx1151, Radeon, ROCm

.. _rdna-performance-model:

=================================
RDNA client GPU performance model
=================================

This chapter covers **AMD Radeon™ / RDNA™** configurations exposed in ROCm Compute Profiler.
It complements the :doc:`../cdna/cdna-performance-model` chapter, which is written around Instinct
accelerators and CDNA naming (CU, shader engine, etc.). Here the focus is on **WGPs**, **TCP /
GL1C / GL2C**, **GCEA**, and related panels when an analysis config targets RDNA-class parts.

Public architecture summaries and :doc:`GPU / accelerator specifications <rocm:reference/gpu-arch-specs>`
remain the best reference for packaging, SIMD width, and generational differences between
RDNA3, RDNA3.5, and RDNA4. The sections below describe what the **profiler measures and names**
in the shipped YAML for each supported generation.

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx1151)
      :selected:

      ROCm Profiler includes analysis panels targeting **RDNA3.5** parts reporting as
      **gfx1151** — for example integrated graphics on AMD Ryzen AI Max+ (Strix Halo) class
      processors.

      .. rubric:: Memory hierarchy in the tool

      For gfx1151, the on-disk **Memory Chart** panel walks the path described in
      ``src/rocprof_compute_soc/analysis_configs/gfx1151/0300_Memory_Chart.yaml``:
      instruction and scalar paths, **TCP** (vector L0), **LDS**, interfaces to
      **GL1C** (L1), **GL2C** (L2), and **GCEA** toward system memory. The CLI
      memory diagram (``mem_chart_gfx11``) uses the same metric key conventions.

      .. rubric:: Workgroups and execution

      RDNA3-class GPUs organize compute around **Workgroup Processors (WGPs)** and
      **Compute Units (CUs)**; wavefronts are typically **wave32**-oriented in this
      configuration. The **WGP**, **SPI**, and **CPC** panels in ``gfx1151`` expose
      dispatch, occupancy, and command-processor side metrics that complement the
      :doc:`Instinct / CDNA conceptual pages <../cdna/compute-unit>` (which use
      naming such as ``CU`` and ``SE`` in places).

      .. rubric:: Where to read metric text

      * :ref:`RDNA3.5 (gfx1151) metrics <gfx1151-metrics-ref>` — single-page index of every documented
        metric (same tables, flat outline).
      * Conceptual sub-pages below group those tables by block (**WGP**, **GL1** / **TCP** + **GL1C**,
        **GL2** / **GL2C** + **GCEA**, **shader engine** / **SPI**, **command processor** / **CPC**,
        **System Speed-of-Light**).
      * :doc:`system-speed-of-light` — SoL tab uses the same metric keys as the analysis panel.
      * Panel YAMLs under ``src/rocprof_compute_soc/analysis_configs/gfx1151/`` —
        formulas, peaks, and counter bindings.

   .. tab-item:: RDNA4

      **RDNA4** (e.g. gfx1200 / gfx1201) does not yet have a dedicated analysis configuration
      in this documentation tree. When ROCm exposes compatible counters and a ``gfx120*``
      panel set ships, this tab will be expanded to match the RDNA3.5 layout (tables, metric
      reference, and Speed-of-Light tab).

In this chapter, profiler concepts for RDNA use **client GPU naming** (WGP, GL1/GL2, …) and embed the
**RDNA3.5 (gfx1151)** metric tables under each block:

* :doc:`wgp` — roofline, WGP utilization, waves, instruction mix, WGP I$/scalar caches.

* :doc:`gl1-cache` — **TCP** (vector L0), **GL1C** (L1), and Memory Chart rows through GL1C.

* :doc:`gl2-cache` — **GL2C**, **GCEA** / DRAM / arbiter, and Memory Chart rows from GL1C–GL2 through system memory.

* :doc:`shader-engine` — GRBM GPU/SE utilization and **SPI** dispatch statistics.

* :doc:`command-processor` — **CPC** / **MEC** metrics (same role as CDNA CP, different tab layout in ``gfx1151``).

* :doc:`system-speed-of-light` — SoL table for **gfx1151**.

* :doc:`references` — public references and link to Instinct citations.

Related
=======

* :doc:`../performance-model` — top-level overview (CDNA and RDNA).
* :doc:`../cdna/cdna-performance-model` — Instinct / CDNA tables and conceptual sections.
