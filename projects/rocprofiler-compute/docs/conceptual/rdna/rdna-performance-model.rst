.. meta::
   :description: Understand the AMD RDNA architectures and hardware blocks with ROCm Compute Profiler to analyze and optimize performance on AMD Ryzen APUs.
   :keywords: ROCm Compute Profiler, RDNA, RDNA3, gfx1151, Radeon, ROCm, Ryzen

.. _rdna-performance-model:

********
RDNA3
********

ROCm Compute Profiler makes available an extensive list of metrics to better understand achieved application performance on RDNA3.5 architecture based AMD Ryzen™ APUs like AMD Ryzen AI Max Series - Strix Halo (gfx1151).

To best use profiling data, it’s important to understand the role of various hardware blocks of AMD RDNA3 architecture. Refer to the following top-level block diagram to understand the hardware blocks of RDNA3 architecture.

.. figure:: ../../data/conceptual/RDNA3_Block_Diagram.png
   :alt: AMD RDNA3 generation series block diagram — host CPU, system and device memory, memory controller, L2/L1 caches, global data share, command processors, ultra-threaded dispatch, and processor array of WGPs with CUs, LDS, instruction and constant caches
   :align: center

For more details on AMD RDNA3 architecture, see page 5 of `RDNA3 shader instruction set architecture <https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0#page=5>`__.

.. Note::
   
   * For top-level metrics details on CDNA and RDNA architecture, see :doc:`../performance-model`.

   * For details on metrics available for CDNA-CDNA4 based Instinct GPUs, see :doc:`../cdna/cdna-performance-model`.
   
   * For details on packaging, SIMD width, and generational differences between RDNA3, RDNA3.5, and later APUs, refer to :doc:`GPU hardware specifications <rocm:reference/gpu-arch-specs>` and the public architecture summaries.

ROCm Compute Profiler includes analysis panels targeting RDNA3.5 parts reporting as
gfx1151 — for example, integrated graphics on AMD Ryzen AI Max Series - Strix Halo
processors.

Memory hierarchy in the tool
==============================

For gfx1151, the Memory Chart panel walks the path from instruction and scalar
paths, TCP (GL0), LDS, interfaces to GL1C, GL2C, and GCEA toward
system memory.

Workgroups and execution
==============================

RDNA3 architecture based APUs organize compute around Workgroup Processors (WGPs) and
Compute Units (CUs). Wavefronts are typically wave32-oriented in this
configuration. The Workgroup processor (WGP), Shader Processor Input (SPI), and Command Processor Compute (CPC) panels in gfx1151 expose the dispatch, occupancy, and command-processor side metrics that complement the Instinct/CDNA
:doc:`Compute unit page <../cdna/compute-unit>`, which uses terminology such as CU and Shader Engines (SE).

Hardware block chapters
========================

The RDNA3.5 architecture based metrics tables are categorized by the following blocks. Profiler concepts for RDNA use APU naming (such as, WGP, GL1/GL2) and embed the RDNA3.5 (gfx1151) metric tables under each block:

* :doc:`system-speed-of-light` — SoL table for gfx1151. It uses the same
  metric keys as the analysis panel.

* :doc:`wgp` — Roofline, WGP utilization, waves, instruction mix, WGP instruction and data caches.

* :doc:`tcp-cache` — TCP (GL0): Panel tables and Memory Chart rows through TCP-GL1.

* :doc:`gl1-cache` — GL1C: Panel tables and Memory Chart GL1C Cache.

* :doc:`gl2-cache` — GL2C, GCEA / DRAM / arbiter, and related panel metrics.

* :doc:`shader-engine` — GRBM GPU/SE utilization and SPI dispatch statistics.

* :doc:`command-processor` — CPC / Micro Engine (ME) metrics (same role as CDNA CP, different tab layout in gfx1151).

* :doc:`references` — Public references and link to Instinct citations.

.. Note::

   ROCm Compute Profiler currently has limited support for WMMA on Strix Halo.
