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
