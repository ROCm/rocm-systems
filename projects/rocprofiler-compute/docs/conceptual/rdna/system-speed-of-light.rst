.. meta::
    :description: ROCm Compute Profiler performance model: System Speed-of-Light (RDNA3-RDNA4)
    :keywords: ROCm Compute Profiler, RDNA, RDNA3.5, RDNA4, gfx115x, gfx1201, system speed of light

.. _sys-sol-rdna:
.. _rdna4-sys-sol:

*********************
System Speed-of-Light
*********************

The page lists the System Speed-of-Light metrics for RDNA3.5 (gfx115x) and
RDNA4 (gfx1201). System Speed-of-Light is a high-level summary of workload
performance. It highlights the most important metrics for how your workload
is performing on the target GPU, so you can spot bottlenecks before diving
into block-specific panels.

For AMD Instinct GPUs (CDNA-CDNA4), see :doc:`../cdna/system-speed-of-light`.

.. warning::

   Theoretical peaks use the maximum clock frequency reported for the GPU (for
   example via ``rocminfo``). That may not match sustained clocks under your
   workload.

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: sys-sol-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-sys-sol-gfx1201
         :file: _templates/metrics_table.j2

RDNA3.5 rows
============

The ``VALU FLOPs`` rows use aggregate VALU instruction counters, which include FP16,
FP32, and FP64 contributions together. Wave Matrix Multiply-Accumulate (WMMA) instructions follow a separate Instruction Set Architecture (ISA)
path and are not broken out in this panel.

Wavefronts and FLOPs accounting
-------------------------------

RDNA3.5 supports both Wave32 (wavefront size 32: primary execution mode) and Wave64 (wavefront size 64).
The wavefront size is fixed per kernel at compile time. The profiler reads
``$wave_size`` from the hardware specs reported by ``rocminfo``, however a given
kernel may also been compiled for the other size. When that happens, treat the
peak rates here as approximations. The ``VALU FLOPs`` row in this panel scales as
``wave_size * SQ_INSTS_VALU_sum / time``.

.. _rdna-dual-issue-valu:

Dual-issue VALU (VOPD)
----------------------

RDNA3.5 add a dual-issue path to the VALU. A pair of independent
VALU operations can be encoded into a single instruction and issued together in
the same cycle. The RDNA ISA refers to this encoding as Vector Operation - Dual (VOPD). Hardware
accepts the pairing only when register, opcode, and operand constraints are
satisfied, and the compiler emits VOPD when those constraints hold.

When dual-issue runs successfully, peak VALU throughput per compute unit (CU) per cycle
doubles relative to single-issue: ``FP32`` from 128 to 256 FLOPs/CU/cycle, and
packed ``FP16`` from 256 to 512 FLOPs/CU/cycle.

The aggregate ``VALU FLOPs`` row in this panel uses the ``FP32`` single-issue Fused Multiply-Add (FMA)
ceiling (128 FLOPs/CU/cycle) as the peak. A workload that issues VOPD heavily,
or that runs packed FP16, can therefore report a percentage of peak above
100%. To gauge how much of the throughput is coming from VOPD, inspect the
``Instructions - Dual VALU (VOPD)`` row on the :doc:`wgp` panel, which counts
``SQ_INSTS_DUAL_VALU_WAVE32``.

Peak theoretical VALU rates
---------------------------

The values below are per CU, before multiplying by the CU count and the shader
clock. These values anchor the percentage of peak reported for the FLOPs related rows.

* FP32 FMA single-issue: 128 FLOPs/CU/cycle. Two SIMD32 lanes per CU
  (``$simd_per_cu``), Wave32 (``$wave_size`` 32), multiplied by two for FMA.
* FP32 VOPD dual-issue: 256 FLOPs/CU/cycle. Paired dual-VALU instructions such
  as ``V_DUAL_ADD_F32`` and ``V_DUAL_MUL_F32``.
* FP16 packed FMA single-issue: 256 FLOPs/CU/cycle. Packed FP16 runs at twice
  the FP32 rate.
* FP16 packed VOPD dual-issue: 512 FLOPs/CU/cycle. Dual-issue packed FP16
  pairings.
* FP64 FMA: 4 FLOPs/CU/cycle. RDNA3.5 runs FP64 FMA at 1/32 of the FP32
  single-issue FMA rate.
* Mixed FP32 single-issue with FP16 packed dual-issue (illustrative ceiling):
  about 384 FLOPs/CU/cycle combined (128 + 256).

Scaling and clocks
------------------

``$cu_per_gpu`` is the total CU count from system info, not WGP count. On RDNA3.5,
each WGP pairs two CUs, so CU count is roughly twice the WGP count - since
``$cu_per_gpu`` reflects active CUs discovered at runtime (rather than just doubling
a fixed WGP count). Peak FLOPs scale with CUs. ``$max_sclk`` is the shader/engine
clock in MHz from profiler system specs.

Bandwidth and cache rows
------------------------

The throughput rows for GL0 (TCP Cache), GL1, GL2, and SQC use heuristic ceilings that are
not anchored to a single public RDNA3.5 table, so the percentage of peak
reported for these rows is indicative rather than exact. The memory hierarchy
runs GL0 (TCP Cache) -> GL1 -> GL2 -> system memory via GCEA. Each level's
ceiling is one peak transfer per cycle, scaled by instance count and
``$max_sclk``:

* GL0 (TCP Cache): One 128-byte cacheline per cycle per CU --
  ``$cu_per_gpu * 128 B/cycle * $max_sclk``
* GL1: One 128-byte request per cycle per GL1C instance -- ``($cu_per_gpu / 8) * 128 B/cycle * $max_sclk``
* GL2: One 128-byte request per cycle per L2 bank --
  ``$total_l2_chan * 128 B/cycle * $max_sclk``
* SQC (scalar data cache and instruction cache): One 64-byte Texture Cache (TC) request per
  cycle per SQC instance -- ``$sqc_per_gpu * 64 B/cycle * $max_sclk``

RDNA4 rows
==========

``VALU Issue Rate`` and ``FP64 VALU Issue Rate`` count instruction issues per
second. They include integer and floating-point issues and do not report FLOPs
or a percentage of a FLOP peak. Wave Matrix Multiply-Accumulate (WMMA)
instructions have their own row, ``WMMA FLOPs (F16->F32 16x16x16)``.

Wavefronts and issue accounting
-------------------------------

RDNA4 supports both wave32 (wavefront size 32: primary execution mode) and wave64 (wavefront size 64).
The wavefront size is fixed per kernel at compile time. The profiler reads
``$wave_size`` from the hardware specs reported by ``rocminfo``, but a given
kernel can be compiled for the other size.

``Wave Dependency Wait`` is the percentage of wave cycles spent waiting for
anything (``SQ_WAIT_ANY / SQG_WAVE_CYCLES``). ``Wave Issue Wait`` is the
percentage spent waiting for any instruction to issue
(``SQ_WAIT_INST_ANY / SQG_WAVE_CYCLES``).

The ``WMMA FLOPs`` row counts 8192 FLOPs per WMMA instruction, the work of one
16x16x16 multiply-accumulate. The counter increments once per vec32, so a wave64
WMMA counts twice. The counter does not separate FP16, BF16, FP8, and BF8 WMMA.
The row compares with 1024 FLOPs/CU/cycle. Treat that conversion and the
percentage of peak as approximations.

.. _rdna4-dual-issue-valu:

Dual-issue VALU (VOPD)
----------------------

RDNA4 keeps the dual-issue VALU path of RDNA3. A pair of independent VALU
operations can be encoded into a single instruction and issued together in the
same cycle. The RDNA4 ISA refers to this encoding as Vector Operation - Dual (VOPD).
Hardware accepts the pairing only when register, opcode, and operand constraints
are satisfied, and the compiler emits VOPD when those constraints hold.

The issue-rate rows count instructions, so a dual-issued pair is one issue.
They are not compared with the FP32 FMA ceiling.

Scaling and clocks
------------------

``$cu_per_gpu`` is the total CU count from system info, not the WGP count. On Navi 48,
``$cu_per_gpu`` is 64 and the 64 CUs form 32 WGPs. HIP reports the 32 WGPs as
multiprocessors. Peak FLOPs scale with CUs. ``$max_sclk`` is the shader clock in MHz
from profiler system specs.

Bandwidth and cache rows
------------------------

The memory hierarchy runs GL0 (TCP) -> GL1 write-combining cache -> GL2 -> MALL (Infinity Cache) -> DRAM.
Rows that have a peak use one transfer per cycle per instance, scaled by instance
count and ``$max_sclk``.

* GL1 read: One 128-byte return per cycle per GL1 instance --
  ``$num_gl1c * 128 B/cycle * $max_sclk``. ``$num_gl1c`` is the shader engine count
  times the shader arrays per engine.
* GL2 Cache BW: One 128-byte request per cycle per GL2 channel --
  ``$total_l2_chan * 128 B/cycle * $max_sclk``. The percentage of peak uses this
  L2 interface width.
* Scalar data cache: One 64-byte request per cycle per SQC instance --
  ``$sqc_per_gpu * 64 B/cycle * $max_sclk``.

Some rows are not available on gfx1201 and show N/A:

* ``GL0 Cache BW (TCP Cache)`` and ``Instruction Cache BW``: the counters give no access size.
* ``GL1 Cache Hit Rate``: the GL1 write-combining cache has no hit or miss counter.

The ``GL2-Fabric Read BW`` and ``GL2-Fabric Write BW`` rows have no peak. The 64 MB
MALL (Infinity Cache) serves part of the traffic that GL2 sends to the fabric, so the
rows are not compared with the 640 GB/s DRAM peak. These EA rates use the configured
32, 64, 128, and 256-byte request categories. Separate compressed request and data
counters are not included. For the measured DRAM and MALL roofs, see :ref:`rdna4-roofline`.

``GL2 Cache BW`` measures uncompressed read requests at the GL2 output FIFOs. It does
not measure all L2 traffic. The percentage of peak uses the L2 interface width, not
the GDDR6 peak. See :doc:`gl2-cache`.

An ``N/A`` value means the metric is unavailable, not zero. GL1 is a write-combining
stage on gfx1201 and has no hit-rate metric. Some bandwidth rows also lack counters
that describe request sizes. Counter availability depends on the installed ROCm
profiler backend. If a requested WMMA counter is missing, profiling diagnostics
identify it. A diagnostic can suggest the corresponding instruction or cycle counter
only when that peer counter was requested and appeared in the same profiling pass.
The two raw counters have different units and cannot replace each other.

Do not subtract WMMA issues from the Speed-of-Light VALU rows to infer FP32 FLOPs,
and do not combine those counts into an independent instruction total. The roofline
rows subtract WMMA by design, as an estimate described in :ref:`rdna4-roofline`.
``Wavefront Occupancy`` is a provisional estimate. Measured wave counts can exceed
the application launch geometry. Do not infer exact resident-wave counts, shader
cycles, or occupancy limits from these values. WGP ``SQ_INST_CYCLES_VALU`` is the
raw SDK aggregate of SIMD execution cycles. Overlapping V_OP32_1 and V_OP32_T
instructions count separately. Do not interpret it as elapsed GPU time, a FLOP
count, or a utilization percentage.
