.. meta::
   :description: ROCm Compute Profiler - RDNA3.5 GRBM GPU and shader-engine utilization
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, GRBM

.. _rdna-grbm:

==========================================
Graphics Register Bus Manager (GRBM)
==========================================

The **Graphics Register Bus Manager (GRBM)** exposes coarse-grained utilization for the whole GPU and for individual shader engines (SEs).
These counters summarize how busy the graphics/compute complex is; pair them with SPI, WGP, and cache chapters when you need root-cause detail.

The sections below list RDNA3.5 (gfx115x) GRBM-derived utilization tables.

.. Note::
   For Instinct-centric shader-engine metrics, see :doc:`../cdna/shader-engine`.

GPU utilization
---------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-gpu-utilization-gfx115x
         :file: _templates/metrics_table.j2

Shader engine utilization
---------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-shader-engine-utilization-gfx115x
         :file: _templates/metrics_table.j2
