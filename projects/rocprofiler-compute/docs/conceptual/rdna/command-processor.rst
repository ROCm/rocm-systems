.. meta::
   :description: ROCm Compute Profiler performance model — RDNA command processor context
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, command processor, CPC

*************************
Command processor (CP)
*************************

The command processor still mediates kernel submission and scheduling on RDNA parts.
For **gfx1151**, **CPC**-related metrics appear in the analysis YAMLs (for example
panels grouped under command-processor naming in the shipped config tree).

The full **command processor (CP)** write-up — fetcher (CPF), packet processor (CPC),
and Instinct-focused metric sections — is maintained here:

* :doc:`../cdna/command-processor`

Return to :doc:`rdna-performance-model` or the top-level :doc:`../performance-model`.
