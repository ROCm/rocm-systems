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

      * - Strix Halo, Strix Point, Gorgon Halo, Gorgon Point and Krackan Point
        - Supported ✅ (see :doc:`/conceptual/rdna/rdna-performance-model`)

  .. tab-item:: AMD Radeon GPUs

    .. list-table::
      :header-rows: 1

      * - Platform
        - Status

      * - AMD Radeon RX 9070 series / Radeon AI PRO R9600 series / Radeon AI PRO R9700 series (Navi 48, gfx1201)
        - Supported ✅ (experimental, see :doc:`/conceptual/rdna/rdna-performance-model`)

      * - Navi 44 (gfx1200)
        - Supported ✅ (experimental)
