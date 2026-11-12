.. meta::
   :description: Understand AMD RDNA architectures and hardware blocks with ROCm Compute Profiler on AMD Ryzen APUs and Radeon GPUs.
   :keywords: ROCm Compute Profiler, RDNA, RDNA3.5, RDNA4, gfx115x, gfx1201, Radeon, ROCm, Ryzen

.. _rdna-performance-model:
.. _rdna4-performance-model:

***********************************
AMD RDNA architecture (RDNA3-RDNA4)
***********************************

ROCm Compute Profiler makes available an extensive list of metrics to better understand the
achieved application performance on AMD RDNA™ GPUs including:

* RDNA3.5 architecture based APUs: AMD Ryzen AI Max / Ryzen AI Max+ (gfx115x)
* RDNA4 architecture based GPUs: AMD Radeon RX 9070 and RX 9070 XT, Navi 48 (gfx1201)

Navi 44 (gfx1200) is not supported in this release. For gfx1201 profiling
prerequisites, see :ref:`profiling power-gating requirements <profile-rdna-power-gating>`.

.. note::

  * For AMD Instinct™ GPUs (CDNA-CDNA4), see :doc:`../cdna/cdna-performance-model`.

  * For top-level metrics details on CDNA and RDNA architecture, see :doc:`../performance-model`.

The table provides key details for the RDNA architectures covered by these chapters.

**Architecture details**

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Architecture
     - RDNA3.5
     - RDNA4
   * - Products
     - AMD Ryzen AI Max / Ryzen AI Max+ (gfx115x)
     - AMD Radeon RX 9070 and RX 9070 XT, Navi 48 (gfx1201)
   * - Chip packaging
     - APU with shared system memory
     - Discrete GPU with dedicated GDDR6
   * - Device memory in the tool
     - System memory through GCEA
     - GDDR6. The 64 MB Infinity Cache (MALL) has a roofline and no MALL counters
   * - Compute organization
     - Each WGP pairs two CUs. Wave32 is the primary mode, and wave64 is supported
     - Navi 48 has 4 shader engines, 64 CUs, and 32 WGPs. Wave32 is the primary mode, and wave64 is supported
   * - Matrix operations
     - WMMA. This profiler has limited WMMA support on gfx115x
     - WMMA, including FP8, BF8, FP16, and BF16. RDNA4 has no MFMA

To best use profiling data, it's important to understand the role of various
hardware blocks. Refer to the following top-level block diagram for each architecture.

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. image:: ../../data/conceptual/RDNA3_Block_Diagram.png
         :alt: AMD RDNA3 generation series block diagram - host CPU, system and device memory, memory controller, L2/L1 caches, global data share, command processors, ultra-threaded dispatch, and processor array of WGPs with CUs, LDS, instruction and constant caches
         :align: center

      For more details on AMD RDNA3 architecture, see page 5 of the `RDNA3 shader instruction set architecture <https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0#page=5>`__.

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. image:: ../../data/conceptual/RDNA4_Block_Diagram.png
         :alt: AMD RDNA4 generation series block diagram - host CPU, system and device memory, memory controller with DMAs, L2 read/write cache, L1 read-only cache, command processors, ultra-threaded dispatch processor, and WGPs with local data share, vector and scalar ALUs, instruction cache, and constant cache
         :align: center

      .. image:: ../../data/conceptual/RDNA4_Shared_Memory_Hierarchy.png
         :alt: RDNA4 shared memory hierarchy - L2 read/write cache per memory channel, crossbar, GL1 write-combining cache, and two workgroup processors, each with constant cache, two L0 texture read/write caches, instruction cache, two CUs with SIMDs, and a 128 KB LDS per WGP
         :align: center

      For more details on AMD RDNA4 architecture, see section 1.2 of the `RDNA4 instruction set architecture <https://gpuopen.com/download/rdna4-instruction-set-architecture.pdf#page=16>`__.
      GL1 is the write-combining cache between the WGPs and GL2.

Hardware blocks
================

The chapters below cover RDNA3.5 (gfx115x) and RDNA4 (gfx1201).
Each metric table has a tab for each generation.

* :doc:`system-speed-of-light`

* :doc:`shader-engine`

  * :doc:`spi`

  * :doc:`wgp`

  * :doc:`gl0-cache`

  * :doc:`gl1-cache`

* :doc:`gl2-cache`

* :doc:`gcea`

* :doc:`command-processor`

* :doc:`grbm`

* :doc:`references`

.. note::

   The System Speed-of-Light, Roofline, Memory Chart, and Efficiency Arbiter
   chapters include metric tables for gfx1201. SPI, WGP, GL0, GL2, and GRBM
   also have metric descriptions in the analysis report.
   ROCm Compute Profiler currently has limited support for WMMA on gfx115x.
