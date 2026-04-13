.. meta::
   :description: ROCm Compute Profiler — RDNA3.5 shader engine / SPI / GRBM metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, shader engine, SPI

.. _rdna-shader-engine:

===============
Shader engine
===============

Shader engines (SEs) still partition the GPU on RDNA hardware; gfx1151 reports
chip- and SE-level utilization through GRBM-derived counters and SPI (Shader
Processor Input) dispatch statistics. This complements the WGP chapter, which
focuses on per-WGP execution metrics.

For Instinct-centric SE, sL1D, and L1I metric tabs, see
:doc:`../cdna/shader-engine`.

Graphics Register Bus Manager (GRBM)
=====================================

GPU Utilization
---------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gpu-utilization-gfx1151
         :file: _templates/metrics_table.j2

Shader Engine Utilization
-------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-shader-engine-utilization-gfx1151
         :file: _templates/metrics_table.j2

Shader Processor Input (SPI)
============================

SPI Utilization
---------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-spi-utilization-gfx1151
         :file: _templates/metrics_table.j2

Wave Dispatch Statistics
------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-wave-dispatch-statistics-gfx1151
         :file: _templates/metrics_table.j2
