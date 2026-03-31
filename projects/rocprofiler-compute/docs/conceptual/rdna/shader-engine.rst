.. meta::
   :description: ROCm Compute Profiler performance model — RDNA shader-engine context
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, shader engine

***************
Shader engine
***************

High-level grouping of CUs / WGPs still exists on RDNA parts, but the shipped **gfx1151**
analysis panels emphasize **WGP**, **SPI**, and memory hierarchy blocks rather than a
separate shader-engine chapter. Use the **WGP** and **SPI** YAMLs under
``analysis_configs/gfx1151/`` for the metrics the tool exposes today.

For **shader engine (SE)** diagrams, scalar L1 / L1I metric tables, and workgroup-manager
discussion written for **AMD Instinct™** CDNA GPUs, see:

* :doc:`../cdna/shader-engine`

Return to :doc:`rdna-performance-model` or the top-level :doc:`../performance-model`.
