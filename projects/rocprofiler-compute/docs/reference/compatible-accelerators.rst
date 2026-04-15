.. meta::
   :description: ROCm Compute Profiler support: compatible accelerators and GPUs
   :keywords: Omniperf, compatible, cdna, gcn, gfx, rdna, radeon, ryzen, hardware, architecture

***********************
Compatible GPUs/APUs
***********************

The following table lists SoCs (System on Chip) tested for compatibility with
ROCm Compute Profiler. See :doc:`rocm:reference/gpu-arch-specs` for full AMD GPUs and APUs specifications.

.. _def-soc:

.. note::

   In ROCm Compute Profiler documentation, the term System on Chip (SoC) refers to a
   particular family of AMD GPUs/APUs.

.. tab-set::

  .. tab-item:: AMD Instinct GPUs

    .. list-table::
      :header-rows: 1

      * - Platform
        - Status

      * - AMD Instinct™ MI350
        - Supported ✅

      * - AMD Instinct™ MI300
        - Supported ✅

      * - AMD Instinct MI200
        - Supported ✅

      * - AMD Instinct MI100
        - Supported ✅

      * - AMD Instinct MI50, MI60 (Vega 20)
        - No support ❌


  
  .. tab-item:: AMD Ryzen APUs

    .. list-table::
      :header-rows: 1

      * - Platform
        - Status
      
      * - AMD RDNA3.5 (gfx1151), e.g. AMD Ryzen™ AI Max+ / Strix Halo integrated graphics
        - Supported ✅ (see :doc:`/conceptual/rdna/rdna-performance-model`)
