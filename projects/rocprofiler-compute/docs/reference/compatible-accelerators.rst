.. meta::
   :description: ROCm Compute Profiler support: compatible accelerators and GPUs
   :keywords: Omniperf, compatible, cdna, gcn, gfx, rdna, radeon, ryzen, hardware, architecture

***********************
Compatible GPUs/APUs
***********************

The following table lists SoCs (System on Chip) tested for compatibility with
ROCm Compute Profiler. See :doc:`rocm:reference/gpu-arch-specs` for full AMD Instinct™ GPUs and AMD Ryzen™ APUs specifications.

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

      * - AMD Instinct MI350 Series
        - Supported ✅

      * - AMD Instinct MI300 Series
        - Supported ✅

      * - AMD Instinct MI200 Series
        - Supported ✅

      * - AMD Instinct MI100 Series
        - Supported ✅

      * - AMD Instinct MI50, MI60 (Vega 20)
        - No support ❌



  .. tab-item:: AMD Ryzen APUs

    .. list-table::
      :header-rows: 1

      * - Platform
        - Status

      * - AMD Ryzen AI Max / Ryzen™ AI Max+ 300 and 400 Series integrated graphics (Strix/Halo, Gorgon/Halo)
        - Supported ✅ (see :doc:`/conceptual/rdna/rdna-performance-model`)
