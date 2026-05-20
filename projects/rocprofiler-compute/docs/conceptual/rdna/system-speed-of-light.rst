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

The VALU FLOPs rows use aggregate VALU instruction counters, which include FP16,
FP32, and FP64 contributions together. WMMA instructions follow a separate ISA
path and are not broken out in this panel.

Wavefronts and FLOPs accounting
-------------------------------

RDNA 3.5 supports both Wave32 (the typical primary mode) and Wave64 wavefronts.
The wavefront size is fixed per kernel at compile time. The profiler reads
``$wave_size`` from the hardware specs reported by ``rocminfo``, but a given
kernel may have been compiled for the other size. When that happens, treat the
peak rates here as approximations. The VALU FLOPs row in this panel scales as
``wave_size * SQ_INSTS_VALU_sum / time``.

.. _rdna-dual-issue-valu:

Dual-issue VALU (VOPD)
----------------------

RDNA 3 and RDNA 3.5 add a dual-issue path to the VALU. A pair of independent
VALU operations can be encoded into a single instruction and issued together in
the same cycle. The RDNA ISA refers to this encoding as **VOPD**. Hardware
accepts the pairing only when register, opcode, and operand constraints are
satisfied, and the compiler emits VOPD when those constraints hold.

When dual-issue runs successfully, peak VALU throughput per CU per cycle
doubles relative to single-issue: FP32 from 128 to 256 FLOPs/CU/cycle, and
packed FP16 from 256 to 512 FLOPs/CU/cycle.

The aggregate ``VALU FLOPs`` row in this panel uses the FP32 single-issue FMA
ceiling (128 FLOPs/CU/cycle) as the peak. A workload that issues VOPD heavily,
or that runs packed FP16, can therefore report a percentage of peak above
100%. Use VALU FLOPs against the ceilings below together with ISA-level VOPD
pairing to reason about how much throughput is coming from dual-issue paths.

Peak theoretical VALU rates
---------------------------

The values below are per CU, before multiplying by the CU count and the shader
clock. They anchor the percentage of peak reported for the FLOPs related rows.

* FP32 FMA, single-issue: 128 FLOPs/CU/cycle. Two SIMD32 lanes per CU
  (``$simd_per_cu``), Wave32 (``$wave_size`` 32), multiplied by two for FMA.
* FP32 VOPD dual-issue: 256 FLOPs/CU/cycle. Paired dual-VALU instructions such
  as ``V_DUAL_ADD_F32`` and ``V_DUAL_MUL_F32``.
* FP16 packed FMA, single-issue: 256 FLOPs/CU/cycle. Packed FP16 runs at twice
  the FP32 rate.
* FP16 packed VOPD dual-issue: 512 FLOPs/CU/cycle. Dual-issue packed FP16
  pairings.
* FP64 FMA: 4 FLOPs/CU/cycle. RDNA 3.5 runs FP64 FMA at 1/32 of the FP32
  single-issue FMA rate.
* Mixed FP32 single-issue with FP16 packed dual-issue (illustrative ceiling):
  about 384 FLOPs/CU/cycle combined (128 + 256).

Scaling and clocks
------------------

``$cu_per_gpu`` is the total CU count from system info, not WGP count. On RDNA 3.5,
each WGP pairs two CUs, so CU count is roughly twice the WGP count - since
``$cu_per_gpu`` reflects active CUs discovered at runtime (rather than just doubling
a fixed WGP count). Peak FLOPs scale with CUs. ``$max_sclk`` is the shader/engine
clock in MHz from profiler system specs.

Bandwidth and cache rows
------------------------

The throughput rows for TCP, GL1, GL2, and SQC use heuristic ceilings (bytes
per cycle, multiplied by instance count and clock). They are not anchored to a
single public RDNA 3.5 table, so the percentage of peak reported for these
rows is indicative rather than exact.

For context, the memory hierarchy is GL0 (TCP Cache), then GL1, then GL2, then
system memory through GCEA and Data Fabric.

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
