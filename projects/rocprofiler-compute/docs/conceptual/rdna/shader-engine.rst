.. meta::
   :description: ROCm Compute Profiler — RDNA3.5 shader engine / SPI / GRBM metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, shader engine, SPI

.. _rdna-shader-engine:

===============
Shader engine
===============

Shader engines (SEs) still partition the GPU on RDNA hardware; gfx1151 reports
Shader Processor Input (SPI) utilization through GRBM-derived counters and dispatch statistics.
This complements the WGP chapter, which focuses on per-WGP execution metrics.

.. Note::
   For Instinct-centric SE, sL1D, and L1I metric tabs, see
   :doc:`../cdna/shader-engine`.

Graphics Register Bus Manager (GRBM)
=====================================

GPU utilization
---------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gpu-utilization-gfx1151
         :file: _templates/metrics_table.j2

Shader engine utilization
-------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-shader-engine-utilization-gfx1151
         :file: _templates/metrics_table.j2

Shader Processor Input (SPI)
============================

SPI utilization
---------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-spi-utilization-gfx1151
         :file: _templates/metrics_table.j2

Wave dispatch statistics
------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-wave-dispatch-statistics-gfx1151
         :file: _templates/metrics_table.j2
