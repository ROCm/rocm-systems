.. meta::
   :description: ROCm Compute Profiler - RDNA4 GRBM GPU utilization
   :keywords: ROCm Compute Profiler, RDNA, RDNA4, gfx1201, GRBM

.. _rdna4-grbm:

==========================================
Graphics Register Bus Manager (GRBM)
==========================================

The Graphics Register Bus Manager (GRBM) exposes coarse-grained utilization for the whole GPU.
These counters summarize how busy the graphics and compute complex is; pair them with Workgroup Manager (SPI), Workgroup processor (WGP), and cache chapters when you need root-cause detail.

The Graphics Register Bus Manager (GRBM) panel (panel 1700) has one table,
GRBM Metrics. It reports GPU Busy, CP Busy, GL2C Busy, and TA Busy. The
texture addresser stays TA. These metrics have descriptions in the analysis
report. Run ``rocprof-compute analyze`` with ``--block 17`` to show the panel.

.. Note::
   For AMD Instinct-centric Shader engine metrics, see :doc:`../cdna/shader-engine`.
   For RDNA3.5 APUs (gfx115x), see :doc:`../rdna/grbm`.
