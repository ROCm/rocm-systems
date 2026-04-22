.. meta::
   :description: Performance model overview in ROCm Compute Profiler, covering CDNA and RDNA architectures, GPU metrics, panel layouts, and profiling concepts.
   :keywords: ROCm Compute Profiler, performance model, CDNA, RDNA, Instinct, Radeon

*****************
Performance model
*****************

ROCm Compute Profiler exposes detailed metrics for AMD Instinct™ / CDNA™ architecture-based
MI-series GPUs and for select AMD Ryzen™ / RDNA™ architecture-based APUs with
supported analysis configurations.

Use the following pages for architecture-specific naming, panel layout, and conceptual
overview:

* **Instinct (CDNA)** — :doc:`AMD CDNA architecture (CDNA-CDNA4) <cdna/cdna-performance-model>`:
  Architecture and data-type tables, top-level CDNA block diagrams, MI-series terminology,
  and chapters for:
  * :doc:`cdna/system-speed-of-light`
  
  * :doc:`cdna/compute-unit`

  * :doc:`cdna/l2-cache`

  * :doc:`cdna/shader-engine`

  * :doc:`cdna/command-processor`

  * :doc:`cdna/references`

* **Ryzen APU (RDNA)** — :doc:`RDNA3 <rdna/rdna-performance-model>`: Architecture, top-level RDNA3 block diagram, GCEA, gfx1151 panel layout, and chapters for:
  * :doc:`rdna/system-speed-of-light`
  
  * :doc:`rdna/wgp`

  * :doc:`rdna/tcp-cache`

  * :doc:`rdna/gl1-cache`
  
  * :doc:`rdna/gl2-cache`

  * :doc:`rdna/shader-engine`

  * :doc:`rdna/command-processor`

  * :doc:`rdna/references`

For practical profiling walkthroughs, see :doc:`/tutorial/profiling-by-example`.
