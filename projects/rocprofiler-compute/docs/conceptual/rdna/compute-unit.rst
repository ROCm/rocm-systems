.. meta::
   :description: ROCm Compute Profiler performance model — RDNA compute unit / WGP context
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, WGP, compute unit

*******************
Compute unit (WGP)
*******************

RDNA-class parts in ROCm Compute Profiler (for example **gfx1151**) expose **Workgroup
Processors (WGPs)** and their **Compute Units (CUs)** in panels such as
``0700_WGP.yaml``. Kernels are usually scheduled as **wave32** wavefronts in these
configs, and occupancy / dispatch metrics are named for that view of the machine.

The long-form **compute unit** chapter — CDNA-oriented diagrams, VALU / VMEM / MFMA
pipeline discussion, and MI-series metric tables — lives under the Instinct track:

* :doc:`../cdna/compute-unit`

Return to :doc:`rdna-performance-model` or the top-level :doc:`../performance-model`.
