.. meta::
   :description: ROCm Compute Profiler - RDNA3.5 shader engine overview and navigation
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, shader engine

.. _rdna-shader-engine:

===============
Shader engine
===============

On RDNA3-class GPUs, **shader engines (SEs)** partition the programmable graphics and compute array into repeating slices.
Within each SE, the **Workgroup Manager (SPI)** accepts dispatched kernels and schedules waves onto **Workgroup Processors (WGPs)**.
Each WGP maps to **two Compute Units (CUs)** that share execution resources and execute the scheduled waves.
**GL0** and **GL1** implement the per-SE vector cache hierarchy feeding those CUs.

Follow the nested chapters under **Shader engine** in the navigation for gfx115x metric tables:

* :doc:`spi` - SPI / Workgroup Manager: utilization and wave dispatch statistics that sit between the command processor and WGP execution.

* :doc:`wgp` - Workgroup Processor: occupancy, waves, instruction mix, and WGP-local instruction/data caches at CU pair granularity.

* :doc:`gl0-cache` - GL0 (TCP vector cache): panels from GL0 utilization through the TCP-GL1 boundary.

* :doc:`gl1-cache` - GL1 Cache: utilization, requests, cache performance, and the GL1-GL2 interface.

GPU-wide and per-SE utilization summarized through GRBM is documented separately; see :doc:`grbm`.

.. Note::
   For Instinct-centric shader-engine metric tabs, see :doc:`../cdna/shader-engine`.
