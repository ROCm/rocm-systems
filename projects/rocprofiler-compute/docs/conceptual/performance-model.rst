.. meta::
   :description: Performance model overview in ROCm Compute Profiler, covering CDNA and RDNA architectures, GPU metrics, panel layouts, and profiling concepts.
   :keywords: ROCm Compute Profiler, performance model, CDNA, RDNA, Instinct, Radeon

*****************
Performance model
*****************

ROCm Compute Profiler exposes detailed metrics for CDNA™ architecture-based AMD Instinct™
MI-series GPUs, for select RDNA™ architecture-based AMD Ryzen™ APUs, and for
RDNA4 architecture-based AMD Radeon™ GPUs with the supported analysis configurations.

Use the following pages for architecture-specific naming, panel layout, and conceptual
overview:

* **Instinct (CDNA)** - :doc:`AMD CDNA architecture (CDNA-CDNA4) <cdna/cdna-performance-model>`:
  Architecture and data-type tables, top-level CDNA block diagrams, MI-series terminology,
  and chapters for:

  * :doc:`cdna/system-speed-of-light`

  * :doc:`cdna/compute-unit`

  * :doc:`cdna/l2-cache`

  * :doc:`cdna/shader-engine`

  * :doc:`cdna/command-processor`

  * :doc:`cdna/references`

* **Ryzen APU and Radeon GPU (RDNA3-RDNA4)** - :doc:`AMD RDNA architecture (RDNA3-RDNA4) <rdna/rdna-performance-model>`:
  Architecture comparison, block-diagram tabs, and one chapter per block. Each metric table has an RDNA3.5 tab and an RDNA4 tab.

  * :doc:`rdna/system-speed-of-light`

  * :doc:`rdna/shader-engine` (overview)

    * :doc:`rdna/spi`

    * :doc:`rdna/wgp`

    * :doc:`rdna/gl0-cache`

    * :doc:`rdna/gl1-cache`

  * :doc:`rdna/gl2-cache`

  * :doc:`rdna/gcea`

  * :doc:`rdna/command-processor`

  * :doc:`rdna/grbm`

  * :doc:`rdna/references`

For practical profiling walkthroughs, see :doc:`/tutorial/profiling-by-example`.
