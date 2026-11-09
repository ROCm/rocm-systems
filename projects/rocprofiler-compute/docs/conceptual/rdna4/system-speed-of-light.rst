.. meta::
    :description: ROCm Compute Profiler performance model: System Speed-of-Light (RDNA4)
    :keywords: ROCm Compute Profiler, RDNA, RDNA4, gfx1201, system speed of light

.. _rdna4-sys-sol:

=====================
System Speed-of-Light
=====================

This section details the System Speed-of-Light metrics for RDNA4 (gfx1201) with the
shipped gfx1201 analysis configuration. System Speed-of-Light is a high-level summary of workload performance. It highlights the most important
metrics for how your workload is performing on the target GPU, so you can spot
bottlenecks before diving into block-specific panels.

The ``VALU FLOPs (FP32)`` row uses the aggregate VALU instruction counter, with
Wave Matrix Multiply-Accumulate (WMMA) instructions subtracted. WMMA instructions
have their own row, ``WMMA FLOPs (F16->F32 16x16x16)``.

Wavefronts and FLOPs accounting
-------------------------------

RDNA4 supports both wave32 (wavefront size 32: primary execution mode) and wave64 (wavefront size 64).
The wavefront size is fixed per kernel at compile time. The profiler reads
``$wave_size`` from the hardware specs reported by ``rocminfo``, but a given
kernel can be compiled for the other size. When that happens, treat the
peak rates here as approximations. The ``VALU FLOPs (FP32)`` row scales as
``wave_size * (SQ_INSTS_VALU_sum - SQ_WMMA_VALU_INSTS_sum) / time``.

The ``WMMA FLOPs`` row counts 8192 FLOPs per WMMA instruction, the work of one
16x16x16 multiply-accumulate. The counter increments once per vec32, so a wave64
WMMA counts twice. The counter does not separate FP16, BF16, FP8, and BF8 WMMA.

.. _rdna4-dual-issue-valu:

Dual-issue VALU (VOPD)
----------------------

RDNA4 keeps the dual-issue VALU path of RDNA3. A pair of independent VALU
operations can be encoded into a single instruction and issued together in the
same cycle. The RDNA4 ISA refers to this encoding as Vector Operation - Dual (VOPD).
Hardware accepts the pairing only when register, opcode, and operand constraints
are satisfied, and the compiler emits VOPD when those constraints hold.

The ``VALU FLOPs (FP32)`` row uses the FP32 single-issue Fused Multiply-Add (FMA)
ceiling (128 FLOPs/CU/cycle) as the peak. A workload that issues VOPD heavily, or
that runs packed FP16, can therefore report a percentage of peak above 100%.

Peak theoretical rates
----------------------

The values below are per CU, before multiplying by the CU count and the shader
clock. These values anchor the percentage of peak reported for the FLOPs rows.

* FP32 FMA single-issue: 128 FLOPs/CU/cycle. Two SIMD32 units per CU, wave32,
  multiplied by two for FMA.
* WMMA FP16 with FP32 accumulate: 1024 FLOPs/CU/cycle, the published dense FP16
  matrix rate of Navi 48.
* FP64 FMA: 4 FLOPs/CU/cycle, 1/32 of the FP32 single-issue FMA rate.

Scaling and clocks
------------------

``$cu_per_gpu`` is the total CU count from system info, not the WGP count. On Navi 48,
``$cu_per_gpu`` is 64 and the 64 CUs form 32 WGPs. HIP reports the 32 WGPs as
multiprocessors. Peak FLOPs scale with CUs. ``$max_sclk`` is the shader clock in MHz
from profiler system specs.

Bandwidth and cache rows
------------------------

The memory hierarchy runs GL0 (TCP) -> GL1 write-combining cache -> GL2 -> MALL (Infinity Cache) -> DRAM. The
throughput rows use one peak transfer per cycle per instance, scaled by instance
count and ``$max_sclk``. These ceilings are not taken from a public RDNA4 table, so
the percentage of peak is indicative rather than exact.

* GL1 read: One 128-byte return per cycle per GL1 instance --
  ``$num_gl1c * 128 B/cycle * $max_sclk``. ``$num_gl1c`` is the shader engine count
  times the shader arrays per engine.
* GL2: One 128-byte request per cycle per GL2 channel --
  ``$total_l2_chan * 128 B/cycle * $max_sclk``. The tool models the 8 MB GL2 as 32
  channels.
* Scalar data cache: One 64-byte request per cycle per SQC instance --
  ``$sqc_per_gpu * 64 B/cycle * $max_sclk``.

Some rows are not available on gfx1201 and show N/A:

* ``GL0 Cache BW (TCP Cache)`` and ``Instruction Cache BW``: the counters give no access size.
* ``GL1 Cache Hit Rate``: the GL1 write-combining cache has no hit or miss counter.

The ``GL2-Fabric Read BW`` and ``GL2-Fabric Write BW`` rows have no peak. The 64 MB
MALL (Infinity Cache) serves part of the traffic that GL2 sends to the fabric, so the
rows are not compared with the 640 GB/s DRAM peak. For the measured DRAM and
MALL roofs, see :ref:`rdna4-roofline`.

.. Note::

  * For AMD Instinct GPUs (CDNA-CDNA4), see :doc:`../cdna/system-speed-of-light`.
  * For RDNA3.5 APUs (gfx115x), see :doc:`../rdna/system-speed-of-light`.
  * Other gfx1201 chapters follow the RDNA4 hierarchy under :doc:`rdna4-performance-model`.

.. warning::

   Theoretical peaks use the maximum clock frequency reported for the GPU (for
   example via ``rocminfo``). That may not match sustained clocks under your
   workload.

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-sys-sol-gfx1201
         :file: _templates/metrics_table.j2
