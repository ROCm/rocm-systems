.. meta::
   :description: ROCm Compute Profiler - RDNA3.5 shader engine overview and navigation
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, shader engine

.. _rdna-shader-engine:
.. _rdna4-shader-engine:

===============
Shader engine
===============

On RDNA3.5 and RDNA4, shader engines (SEs) partition the programmable graphics and compute array into repeating slices.
Within each SE, the Workgroup Manager (SPI) accepts dispatched kernels and schedules waves onto Workgroup Processors (WGPs).
Each WGP pairs two Compute Units (CUs) that share execution resources and execute the scheduled waves.
Navi 48 (gfx1201) has four SEs and 64 CUs. Each of its WGPs shares a 128 KB Local Data Share (LDS).
GL0 is the vector L0 cache. On gfx1201, GL1 is the write-combining cache.

Follow the nested sections under Shader engine in the navigation. Each metric table has an RDNA3.5 tab and an RDNA4 tab:

* :doc:`spi`: Utilization and wave dispatch statistics that sit between the command processor and WGP execution.

* :doc:`wgp`: Occupancy, waves, instruction mix, and WGP-local instruction/data caches at CU pair granularity.

* :doc:`gl0-cache`: Panels from GL0 utilization through the TCP-GL1 boundary.

* :doc:`gl1-cache`: Utilization, requests, cache performance, and the GL1-GL2 interface.

GPU-wide and per-SE utilization summarized through GRBM is documented separately; see :doc:`grbm`.

.. Note::
   For AMD Instinct-centric Shader engine metric tabs, see :doc:`../cdna/shader-engine`.
