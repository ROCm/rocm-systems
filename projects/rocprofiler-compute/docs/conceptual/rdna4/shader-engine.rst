.. meta::
   :description: ROCm Compute Profiler - RDNA4 shader engine overview and navigation
   :keywords: ROCm Compute Profiler, RDNA, RDNA4, gfx1201, shader engine

.. _rdna4-shader-engine:

===============
Shader engine
===============

On RDNA4 architecture-based GPUs, shader engines (SEs) partition the programmable graphics and compute array into repeating slices.
Navi 48 (gfx1201) has four SEs with 64 Compute Units (CUs) in total.
Within each SE, the Workgroup Manager (SPI) accepts dispatched kernels and schedules waves onto Workgroup Processors (WGPs).
Each WGP pairs two CUs that share a 128 KB Local Data Share (LDS) and execute the scheduled waves.
GL0, the vector L0 cache, and the GL1 write-combining cache feed those CUs.

Follow the nested sections under Shader engine in the navigation for gfx1201 metrics:

* :doc:`spi`: Wavefront dispatch between the command processor and WGP execution.

* :doc:`wgp`: Roofline, occupancy, instruction mix, and the WGP instruction and scalar data caches.

* :doc:`gl0-cache`: The vector L0 cache (TCP) and LDS, through the GL0-GL1 interface.

* :doc:`gl1-cache`: The GL1 write-combining cache and the GL1-GL2 interface.

GPU-wide utilization summarized through GRBM is documented separately; see :doc:`grbm`.

.. Note::
   For AMD Instinct-centric Shader engine metric tabs, see :doc:`../cdna/shader-engine`.
   For RDNA3.5 APUs (gfx115x), see :doc:`../rdna/shader-engine`.
