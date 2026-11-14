.. meta::
   :description: ROCm Compute Profiler - RDNA3.5 GRBM GPU and shader-engine utilization
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, GRBM

.. _rdna-grbm:
.. _rdna4-grbm:

==========================================
Graphics Register Bus Manager (GRBM)
==========================================

The Graphics Register Bus Manager (GRBM) exposes coarse-grained utilization for the whole GPU and for individual Shader engines (SEs).
These counters summarize how busy the graphics/compute complex is; pair them with Workgroup Manager (SPI), Workgroup processor (WGP), and cache chapters when you need root-cause detail.

The sections below list GRBM utilization tables. Select the RDNA3.5 or RDNA4 tab on each table. On gfx1201, GRBM Metrics reports GPU Busy.

.. Note::
   For AMD Instinct-centric Shader engine metrics, see :doc:`../cdna/shader-engine`.

GPU utilization
---------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-gpu-utilization-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      The gfx1201 GRBM panel has one table, GRBM Metrics. It reports GPU Busy. Those metrics have descriptions in the analysis report. This page does not publish a separate gfx1201 table.

Shader engine utilization
---------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-shader-engine-utilization-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      The gfx1201 GRBM panel has one table, GRBM Metrics. It reports GPU Busy. Those metrics have descriptions in the analysis report. This page does not publish a separate gfx1201 table.
