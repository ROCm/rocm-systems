.. meta::
   :description: ROCm Compute Profiler performance model — RDNA L2 / GL2C context
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, GL2C, L2 cache

*****************
L2 cache (GL2C)
*****************

For **gfx1151**, the profiler’s **Memory Chart** and related panels refer to last-level
cache clients as **GL2C** (and the memory path toward **GCEA**). Counter and metric
names follow the shipped YAML under
``src/rocprof_compute_soc/analysis_configs/gfx1151/``.

The **L2 cache (TCC)** chapter documents the Instinct-centric coherence point, channel
hashing, and metric families (Speed-of-Light, accesses, fabric, etc.) for MI-series
accelerators:

* :doc:`../cdna/l2-cache`

Return to :doc:`rdna-performance-model` or the top-level :doc:`../performance-model`.
