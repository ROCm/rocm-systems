.. meta::
   :description: ROCm Compute Profiler - RDNA4 GRBM GPU utilization
   :keywords: ROCm Compute Profiler, RDNA, RDNA4, gfx1201, GRBM

.. _rdna4-grbm:

==========================================
Graphics Register Bus Manager (GRBM)
==========================================

The Graphics Register Bus Manager (GRBM) exposes coarse-grained utilization for the whole GPU.
These counters summarize how busy the graphics and compute complex is; pair them with Workgroup Manager (SPI), Workgroup processor (WGP), and cache chapters when you need root-cause detail.

The GRBM (Graphics Register Bus Manager) panel (panel 1700) has one table, GRBM
Metrics. The gfx1201 GRBM metrics have no descriptions yet, so this page has no
metric tables. Run ``rocprof-compute analyze`` with ``--block 17`` to show the panel.

.. Note::
   For AMD Instinct-centric Shader engine metrics, see :doc:`../cdna/shader-engine`.
   For RDNA3.5 APUs (gfx115x), see :doc:`../rdna/grbm`.
