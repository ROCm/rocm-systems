.. meta::
   :description: Understand AMD RDNA architectures and hardware blocks with ROCm Compute Profiler on AMD Ryzen APUs and Radeon GPUs.
   :keywords: ROCm Compute Profiler, RDNA, RDNA3.5, RDNA4, gfx115x, gfx1201, Radeon, ROCm, Ryzen

.. _rdna-performance-model:

***********************************
AMD RDNA architecture (RDNA3-RDNA4)
***********************************

ROCm Compute Profiler makes available an extensive list of metrics to better understand the
achieved application performance on AMD RDNA architectures including:

* RDNA3.5 architecture based APUs: AMD Ryzen AI Max / Ryzen AI Max+ (gfx115x)
* RDNA4 architecture based GPUs: AMD Radeon RX 9070 series / Radeon AI PRO R9600 series / Radeon AI PRO R9700 series (Navi 48 / gfx1201)

Navi 44 (gfx1200) is not supported. For gfx1201 profiling
prerequisites, see :ref:`profiling power-gating requirements <profile-rdna-power-gating>`.

.. warning::

   gfx1201 support is experimental. Add ``--experimental`` to each
   ``rocprof-compute`` command on gfx1201, for example
   ``rocprof-compute --experimental profile -n <name> -- <app>`` and
   ``rocprof-compute --experimental analyze -p <workload>``.

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
     - AMD Radeon RX 9070 series / Radeon AI PRO R9600 series / Radeon AI PRO R9700 series (Navi 48 / gfx1201)
   * - Chip packaging
     - APU with shared system memory
     - Discrete GPU with dedicated VRAM
   * - Device memory in the tool
     - System memory through GCEA
     - VRAM
   * - Compute organization
     - Each WGP pairs two CUs. Wave32 is the primary mode, and wave64 is supported
     - Navi 48 has 4 shader engines, and each WGP pairs two CUs. The CU count depends on the board. Wave32 is the primary mode, and wave64 is supported
   * - Matrix operations
     - This profiler has limited WMMA support on gfx115x
     - WMMA support, including FP8, BF8, FP16, and BF16.

To best use profiling data, it's important to understand the role of various
hardware blocks. Refer to the following top-level block diagram for each architecture.

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. image:: ../../data/conceptual/RDNA3_Block_Diagram.png
         :alt: AMD RDNA3 generation series block diagram - host CPU, system and device memory, memory controller, L2/L1 caches, global data share, command processors, ultra-threaded dispatch, and processor array of WGPs with CUs, LDS, instruction and constant caches
         :align: center

      For more details on AMD RDNA3 architecture, see page 5 of the `RDNA3 shader instruction set architecture <https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0#page=5>`__.

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. image:: ../../data/conceptual/RDNA4_Block_Diagram.png
         :alt: AMD RDNA4 generation series block diagram - host CPU, system and device memory, memory controller with DMAs, L2 read/write cache, GL1 write-combining buffer, command processors, ultra-threaded dispatch processor, and WGPs with local data share, vector and scalar ALUs, instruction cache, and constant cache
         :align: center

      .. image:: ../../data/conceptual/RDNA4_Shared_Memory_Hierarchy.png
         :alt: RDNA4 shared memory hierarchy - L2 read/write cache per memory channel, crossbar, GL1 write-combining buffer, and two workgroup processors, each with constant cache, two L0 texture read/write caches, instruction cache, two CUs with SIMDs, and a 128 KB LDS per WGP
         :align: center

      For more details on AMD RDNA4 architecture, see section 1.2 of the `RDNA4 instruction set architecture <https://gpuopen.com/download/rdna4-instruction-set-architecture.pdf#page=16>`__.
      GL1 is the write-combining buffer between the WGPs and GL2.

Hardware blocks
================

The chapters below cover RDNA3.5 (gfx115x) and RDNA4 (gfx1201).
Each metric table has a tab for each generation.

* :doc:`system-speed-of-light`: The top-level metrics and how they count FLOPs and bandwidth.

* :doc:`shader-engine`: The shader engine and its blocks.

  * :doc:`spi`: The workgroup manager and wave dispatch.

  * :doc:`wgp`: The roofline, waves, instruction mix, and local caches of the workgroup processor.

  * :doc:`gl0-cache`: The vector L0 cache.

  * :doc:`gl1-cache`: The GL1 stage between GL0 and GL2.

* :doc:`gl2-cache`: The L2 cache.

* :doc:`gcea`: The Graphics Core Efficiency Arbiter (GCEA), where traffic leaves the cache hierarchy.

* :doc:`command-processor`: The command processor.

* :doc:`grbm`: The graphics register bus manager.

* :doc:`references`: Further reading.

.. note::

   * The :doc:`system-speed-of-light`, :doc:`wgp`, :doc:`gl0-cache`, :doc:`gl1-cache`, :doc:`gl2-cache`, :doc:`spi`, and :doc:`grbm` chapters have gfx1201 tables.
   * The roofline is in the :doc:`wgp` chapter. The Memory Chart tables are in the :doc:`wgp`, :doc:`gl0-cache`, :doc:`gl2-cache`, and :doc:`gcea` chapters.
   * The :doc:`command-processor` chapter says that gfx1201 has no CPC panel. Some panels are missing because the profiler has no public counters for them, not because the hardware lacks that block.
   * ROCm Compute Profiler currently has limited support for WMMA on gfx115x.
