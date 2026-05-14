.. meta::
   :description: ROCm Compute Profiler performance model: System Speed-of-Light (RDNA)
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, system speed of light

.. _sys-sol-rdna:

=====================
System Speed-of-Light
=====================

This page documents System Speed-of-Light metrics for RDNA3.5 (gfx1151) with the
shipped gfx1151 analysis configuration.
**System Speed-of-Light** is a high-level summary: it highlights the most important
metrics for how your workload is performing on the target GPU, so you can spot
bottlenecks before diving into block-specific panels.

VALU FLOPs rows use aggregate VALU instruction counters (they include FP16, FP32,
and FP64). WMMA follows a separate ISA path and is not broken out in this panel.

Wavefronts and FLOPs accounting
--------------------------------

RDNA 3.5 supports both Wave32 (typical primary mode) and Wave64 wavefronts;
wavefront size is fixed per kernel at compile time. The profiler's ``$wave_size``
depends on the hardware architecture (i.e., ``rocminfo``), while the executing kernel
may use a different size—treat peaks as an approximation when those differ. VALU FLOPs
in this panel scale roughly as ``wave_size * SQ_INSTS_VALU_sum / time``.

Peak theoretical VALU rates (per CU, before multiplying by CU count and clock)
---------------------------------------------------------------------------------

These ceilings explain how POP (% of peak) is anchored for FLOPs-oriented rows:

* FP32 FMA, single-issue: 128 FLOPs/CU/cycle — two SIMD32 lanes per CU
  (``$simd_per_cu``), Wave32 (``$wave_size`` 32), times two for FMA.
* FP32 VOPD dual-issue: 256 FLOPs/CU/cycle (paired dual-VALU ops such as
  ``V_DUAL_ADD_F32`` / ``V_DUAL_MUL_F32``).
* FP16 packed FMA, single-issue: 256 FLOPs/CU/cycle (packed half-rate vs FP32).
* FP16 packed VOPD dual-issue: 512 FLOPs/CU/cycle (dual-issue packed half ops).
* FP64 FMA: 4 FLOPs/CU/cycle — 1/32 of the FP32 single-issue FMA rate on RDNA.
* Mixed FP32 single-issue plus FP16 packed dual-issue (illustrative ceiling):
  about 384 FLOPs/CU/cycle combined (128 + 256).

Scaling and clocks
------------------

``$cu_per_gpu`` is the total CU count from system info, not WGP count. On RDNA 3.5,
each WGP pairs two CUs, so CU count is approximately twice the WGP count; peak
FLOPs scale with CUs. ``$max_sclk`` is the shader/engine clock in MHz from profiler
system specs.

Bandwidth and cache POP rows
----------------------------

TCP, GL1, GL2, and SQC throughput POP values use heuristic ceilings (bytes per
cycle * instance count * clock). They are not tied to a single public RDNA 3.5 table—
treat Pct of Peak as indicative, not exact.

Memory hierarchy (for context): GL0 (TCP) → GL1 → GL2 → system memory (GCEA).

.. tip::

   If Pct of Peak for VALU FLOPs goes above 100%, the assumed peak likely underestimates
   achievable throughput—for example when the kernel uses VOPD dual-issue or heavy packed
   FP16 instead of the single-issue FP32 FMA baseline. Inspect the ``Instructions - Dual VALU (VOPD)``
   metric on the :doc:`wgp` panel (counter ``SQ_INSTS_DUAL_VALU_WAVE32``) to gauge dual-issue
   activity and interpret POP accordingly.

.. Note::
   For AMD Instinct accelerators (CDNA-CDNA4), see
   :doc:`../cdna/system-speed-of-light`.

   Other gfx1151 metric tables follow the RDNA3 hierarchy in :doc:`rdna-performance-model`
   (shader engine: :doc:`spi`, :doc:`wgp`, :doc:`gl0-cache`, :doc:`gl1-cache`; then
   :doc:`gl2-cache`, :doc:`gcea`, :doc:`command-processor`, :doc:`grbm`).

.. warning::

   Theoretical peaks use the maximum clock frequency reported for the GPU (for
   example via ``rocminfo``). That may not match sustained clocks under your
   workload.

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: sys-sol-gfx1151
         :file: _templates/metrics_table.j2
