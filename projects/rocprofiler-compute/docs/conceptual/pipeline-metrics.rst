.. meta::
   :description: ROCm Compute Profiler performance model: Pipeline metrics
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, pipeline, wavefront, metrics, launch, runtime
              VALU, MFMA, instruction mix, FLOPs, arithmetic, operations

****************
Pipeline metrics
****************

In this section, we describe the metrics available in ROCm Compute Profiler to analyze the
pipelines discussed in the :doc:`pipeline-descriptions`.

.. _wavefront:

Wavefront
=========

.. _wavefront-launch-stats:

Wavefront launch stats
----------------------

The wavefront launch stats panel gives general information about the
kernel launch:

.. list-table::
   :header-rows: 1
   :widths: 20 65 15

   * - Metric

     - Description

     - Unit

   * - Grid Size

     - The total number of work-items (or, threads) launched as a part of
       the kernel dispatch.  In HIP, this is equivalent to the total grid size
       multiplied by the total workgroup (or, block) size.

     - :ref:`Work-items <desc-work-item>`

   * - Workgroup Size

     - The total number of work-items (or, threads) in each workgroup
       (or, block) launched as part of the kernel dispatch.  In HIP, this is
       equivalent to the total block size.

     - :ref:`Work-items <desc-work-item>`

   * - Total Wavefronts

     - The total number of wavefronts launched as part of the kernel dispatch.
       On AMD Instinct™ CDNA™ accelerators and GCN™ GPUs, the wavefront size is
       always 64 work-items.  Thus, the total number of wavefronts should be
       equivalent to the ceiling of grid size divided by 64.

     - :ref:`Wavefronts <desc-wavefront>`

   * - Saved Wavefronts

     - The total number of wavefronts saved at a context-save. See
       `cwsr_enable <https://docs.kernel.org/gpu/amdgpu/module-parameters.html?highlight=cwsr>`_.

     - :ref:`Wavefronts <desc-wavefront>`

   * - Restored Wavefronts

     - The total number of wavefronts restored from a context-save. See
       `cwsr_enable <https://docs.kernel.org/gpu/amdgpu/module-parameters.html?highlight=cwsr>`_.

     - :ref:`Wavefronts <desc-wavefront>`

   * - VGPRs

     - The number of architected vector general-purpose registers allocated for
       the kernel, see :ref:`VALU <desc-valu>`.  Note: this may not exactly
       match the number of VGPRs requested by the compiler due to allocation
       granularity.

     - :ref:`VGPRs <desc-valu>`

   * - AGPRs

     - The number of accumulation vector general-purpose registers allocated for
       the kernel, see :ref:`AGPRs <desc-agprs>`.  Note: this may not exactly
       match the number of AGPRs requested by the compiler due to allocation
       granularity.

     - :ref:`AGPRs <desc-agprs>`

   * - SGPRs

     - The number of scalar general-purpose registers allocated for the kernel,
       see :ref:`SALU <desc-salu>`.  Note: this may not exactly match the number
       of SGPRs requested by the compiler due to allocation granularity.

     - :ref:`SGPRs <desc-salu>`

   * - LDS Allocation

     - The number of bytes of :doc:`LDS <local-data-share>` memory (or, shared
       memory) allocated for this kernel.  Note: This may also be larger than
       what was requested at compile time due to both allocation granularity and
       dynamic per-dispatch LDS allocations.

     - Bytes per :ref:`workgroup <desc-workgroup>`

   * - Scratch Allocation

     - The number of bytes of :ref:`scratch memory <memory-spaces>` requested
       per work-item for this kernel. Scratch memory is used for stack memory
       on the accelerator, as well as for register spills and restores.

     - Bytes per :ref:`work-item <desc-work-item>`

.. _wavefront-runtime-stats:

Wavefront runtime stats
-----------------------

The wavefront runtime statistics gives a high-level overview of the
execution of wavefronts in a kernel:

.. list-table::
   :header-rows: 1
   :widths: 18 65 17

   * - Metric

     - Description

     - Unit

   * - :ref:`Kernel time <kernel-time>`

     - The total duration of the executed kernel. Note: this should not be
       directly compared to the wavefront cycles / timings below.

     - Nanoseconds

   * - :ref:`Kernel cycles <kernel-cycles>`

     - The total duration of the executed kernel in cycles. Note: this should
       not be directly compared to the wavefront cycles / timings below.

     - Cycles

   * - Instructions per wavefront

     - The average number of instructions (of all types) executed per wavefront.
       This is averaged over all wavefronts in a kernel dispatch.

     - Instructions / wavefront

   * - Wave cycles

     - The number of cycles a wavefront in the kernel dispatch spent resident on
       a compute unit per :ref:`normalization unit <normalization-units>`. This
       is averaged over all wavefronts in a kernel dispatch.  Note: this should
       not be directly compared to the kernel cycles above.

     - Cycles per :ref:`normalization unit <normalization-units>`

   * - Dependency wait cycles

     - The number of cycles a wavefront in the kernel dispatch stalled waiting
       on memory of any kind (e.g., instruction fetch, vector or scalar memory,
       etc.) per :ref:`normalization unit <normalization-units>`. This counter
       is incremented at every cycle by *all* wavefronts on a CU stalled at a
       memory operation.  As such, it is most useful to get a sense of how waves
       were spending their time, rather than identification of a precise limiter
       because another wave could be actively executing while a wave is stalled.
       The sum of this metric, Issue Wait Cycles and Active Cycles should be
       equal to the total Wave Cycles metric.

     - Cycles per :ref:`normalization unit <normalization-units>`

   * - Issue Wait Cycles

     - The number of cycles a wavefront in the kernel dispatch was unable to
       issue an instruction for any reason (e.g., execution pipe back-pressure,
       arbitration loss, etc.) per
       :ref:`normalization unit <normalization-units>`.  This counter is
       incremented at every cycle by *all* wavefronts on a CU unable to issue an
       instruction.  As such, it is most useful to get a sense of how waves were
       spending their time, rather than identification of a precise limiter
       because another wave could be actively executing while a wave is issue
       stalled.  The sum of this metric, Dependency Wait Cycles and Active
       Cycles should be equal to the total Wave Cycles metric.

     - Cycles per :ref:`normalization unit <normalization-units>`

   * - Active Cycles

     - The average number of cycles a wavefront in the kernel dispatch was
       actively executing instructions per
       :ref:`normalization unit <normalization-units>`. This measurement is made
       on a per-wavefront basis, and may include cycles that another wavefront
       spent actively executing (on another execution unit, for example) or was
       stalled.  As such, it is most useful to get a sense of how waves were
       spending their time, rather than identification of a precise limiter. The
       sum of this metric, Issue Wait Cycles and Active Wait Cycles should be
       equal to the total Wave Cycles metric.

     - Cycles per :ref:`normalization unit <normalization-units>`

   * - Wavefront Occupancy

     - The time-averaged number of wavefronts resident on the accelerator over
       the lifetime of the kernel. Note: this metric may be inaccurate for
       short-running kernels (less than 1ms).

     - :ref:`Wavefronts <desc-wavefront>`

.. note::

   As mentioned earlier, the measurement of kernel cycles and time typically
   cannot be directly compared to, for example, wave cycles. This is due to two factors:
   first, the kernel cycles/timings are measured using a counter that is
   impacted by scheduling overhead, this is particularly noticeable for
   "short-running" kernels (less than 1ms) where scheduling overhead forms a
   significant portion of the overall kernel runtime. Secondly, the wave cycles
   metric is incremented per-wavefront scheduled to a SIMD every cycle whereas
   the kernel cycles counter is incremented only once per-cycle when *any*
   wavefront is scheduled.

.. _instruction-mix:

Instruction mix
===============

The instruction mix panel shows a breakdown of the various types of instructions
executed by the user’s kernel, and which pipelines on the
:doc:`CU <compute-unit>` they were executed on. In addition, ROCm Compute Profiler reports
further information about the breakdown of operation types for the
:ref:`VALU <desc-valu>`, vector-memory, and :ref:`MFMA <desc-mfma>`
instructions.

.. note::

   All metrics in this section count *instructions issued*, and *not* the total
   number of operations executed. The values reported by these metrics will not
   change regardless of the execution mask of the wavefront. Note that even if
   the execution mask is identically zero (meaning that *no lanes are active*)
   the instruction will still be counted, as CDNA accelerators still consider
   these instructions *issued*. See
   :mi200-isa-pdf:`EXECute Mask, section 3.3 of the CDNA2 ISA guide<19>` for
   examples and further details.

Overall instruction mix
-----------------------

This panel shows the total number of each type of instruction issued to
the :doc:`various compute pipelines </conceptual/pipeline-descriptions>` on the
:doc:`CU </conceptual/compute-unit>`. These are:

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - :ref:`VALU <desc-valu>` instructions

     - The total number of vector arithmetic logic unit (VALU) operations
       issued. These are the workhorses of the
       :doc:`compute unit <compute-unit>`, and are used to execute a wide range of
       instruction types including floating point operations, non-uniform
       address calculations, transcendental operations, integer operations,
       shifts, conditional evaluation, etc.

     - Instructions

   * - VMEM instructions

     - The total number of vector memory operations issued. These include most
       loads, stores and atomic operations and all accesses to
       :ref:`generic, global, private and texture <memory-spaces>` memory.

     - Instructions

   * - :doc:`LDS <local-data-share>` instructions

     - The total number of LDS (also known as shared memory) operations issued.
       These include loads, stores, atomics, and HIP's ``__shfl`` operations.

     - Instructions

   * - :ref:`MFMA <desc-mfma>` instructions

     - The total number of matrix fused multiply-add instructions issued.

     - Instructions

   * - :ref:`SALU <desc-salu>` instructions

     - The total number of scalar arithmetic logic unit (SALU) operations
       issued. Typically these are used for address calculations, literal
       constants, and other operations that are *provably* uniform across a
       wavefront. Although scalar memory (SMEM) operations are issued by the
       SALU, they are counted separately in this section.

     - Instructions

   * - SMEM instructions

     - The total number of scalar memory (SMEM) operations issued. These are
       typically used for loading kernel arguments, base-pointers and loads
       from HIP's ``__constant__`` memory.

     - Instructions

   * - :ref:`Branch <desc-branch>` instructions

     - The total number of branch operations issued. These typically consist of
       jump or branch operations and are used to implement control flow.

     - Instructions

.. note::

   Note, as mentioned in the :ref:`desc-branch` section: branch
   operations are not used for execution mask updates, but only for "whole
   wavefront" control flow changes.

.. _valu-arith-instruction-mix:

VALU arithmetic instruction mix
-------------------------------

.. warning::

   Not all metrics in this section (for instance, the floating-point instruction
   breakdowns) are available on CDNA accelerators older than the
   :ref:`MI2XX <mixxx-note>` series.

This panel details the various types of vector instructions that were
issued to the :ref:`VALU <desc-valu>`. The metrics in this section do *not*
include :ref:`MFMA <desc-mfma>` instructions using the same precision; for
instance, the “F16-ADD” metric does not include any 16-bit floating point
additions executed as part of an MFMA instruction using the same precision.

.. list-table::
   :header-rows: 1
   :widths: 15 65 20

   * - Metric

     - Description

     - Unit

   * - INT32

     - The total number of instructions operating on 32-bit integer operands
       issued to the VALU per :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - INT64

     - The total number of instructions operating on 64-bit integer operands
       issued to the VALU per :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - F16-ADD

     - The total number of addition instructions operating on 16-bit
       floating-point operands issued to the VALU per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - F16-MUL

     - The total number of multiplication instructions operating on 16-bit
       floating-point operands issued to the VALU per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - F16-FMA

     - The total number of fused multiply-add instructions operating on 16-bit
       floating-point operands issued to the VALU per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - F16-TRANS

     - The total number of transcendental instructions (e.g., `sqrt`) operating
       on 16-bit floating-point operands issued to the VALU per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - F32-ADD

     - The total number of addition instructions operating on 32-bit
       floating-point operands issued to the VALU per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - F32-MUL

     - The total number of multiplication instructions operating on 32-bit
       floating-point operands issued to the VALU per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - F32-FMA

     - The total number of fused multiply-add instructions operating on 32-bit
       floating-point operands issued to the VALU per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - F32-TRANS

     - The total number of transcendental instructions (such as ``sqrt``)
       operating on 32-bit floating-point operands issued to the VALU per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - F64-ADD

     - The total number of addition instructions operating on 64-bit
       floating-point operands issued to the VALU per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - F64-MUL

     - The total number of multiplication instructions operating on 64-bit
       floating-point operands issued to the VALU per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - F64-FMA

     - The total number of fused multiply-add instructions operating on 64-bit
       floating-point operands issued to the VALU per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - F64-TRANS

     - The total number of transcendental instructions (such as `sqrt`)
       operating on 64-bit floating-point operands issued to the VALU per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - Conversion

     - The total number of type conversion instructions (such as converting data
       to or from F32↔F64) issued to the VALU per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

For an example of these counters in action, refer to
:ref:`valu-arith-instruction-mix-ex`.

.. _vmem-instruction-mix:

VMEM instruction mix
--------------------

This section breaks down the types of vector memory (VMEM) instructions
that were issued. Refer to the
:ref:`Instruction Counts metrics section <ta-instruction-counts>` under address
processor front end of the vL1D cache for descriptions of these VMEM
instructions.

.. _mfma-instruction-mix:

MFMA instruction mix
--------------------

.. warning::

   The metrics in this section are only available on CDNA2
   (:ref:`MI2XX <mixxx-note>`) accelerators and newer.

This section details the types of Matrix Fused Multiply-Add
(:ref:`MFMA <desc-mfma>`) instructions that were issued. Note that
MFMA instructions are classified by the type of input data they operate on, and
*not* the data type the result is accumulated to.

.. list-table::
   :header-rows: 1
   :widths: 25 60 17

   * - Metric

     - Description

     - Unit

   * - MFMA-I8 Instructions

     - The total number of 8-bit integer :ref:`MFMA <desc-mfma>` instructions
       issued per :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - MFMA-F8 Instructions

     - The total number of 8-bit floating point :ref:`MFMA <desc-mfma>`
       instructions issued per :ref:`normalization unit <normalization-units>`. This is supported in AMD Instinct MI300 series and later only.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - MFMA-F16 Instructions

     - The total number of 16-bit floating point :ref:`MFMA <desc-mfma>`
       instructions issued per :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - MFMA-BF16 Instructions

     - The total number of 16-bit brain floating point :ref:`MFMA <desc-mfma>`
       instructions issued per :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - MFMA-F32 Instructions

     - The total number of 32-bit floating-point :ref:`MFMA <desc-mfma>`
       instructions issued per :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - MFMA-F64 Instructions

     - The total number of 64-bit floating-point :ref:`MFMA <desc-mfma>`
       instructions issued per :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

Compute pipeline
================

.. _metrics-flop-count:

FLOP counting conventions
-------------------------

ROCm Compute Profiler’s conventions for VALU FLOP counting are as follows:

* Addition or multiplication: 1 operation

* Transcendentals: 1 operation

* Fused multiply-add (FMA): 2 operations

Integer operations (IOPs) do not use this convention. They are counted
as a single operation regardless of the instruction type.

.. note::

   Packed operations which operate on multiple operands in the same instruction
   are counted identically to the underlying instruction type. For example, the
   ``v_pk_add_f32`` instruction on :ref:`MI2XX <mixxx-note>`, which performs an
   add operation on two pairs of aligned 32-bit floating-point operands is
   counted only as a single addition -- that is, 1 operation.

As discussed in the :ref:`instruction-mix` section, the FLOP/IOP
metrics in this section do not take into account the execution mask of
the operation, and will report the same value even if the execution mask
is identically zero.

For example, a FMA instruction operating on 32-bit floating-point
operands (such as ``v_fma_f32`` on a :ref:`MI2XX <mixxx-note>` accelerator)
would be counted as 128 total FLOPs: 2 operations (due to the
instruction type) multiplied by 64 operations (because the wavefront is
composed of 64 work-items).

.. _compute-speed-of-light:

Compute Speed-of-Light
----------------------

.. warning::

   The theoretical maximum throughput for some metrics in this section are
   currently computed with the maximum achievable clock frequency, as reported
   by ``rocminfo``, for an accelerator. This may not be realistic for all
   workloads.

This section reports the number of floating-point and integer operations
executed on the :ref:`VALU <desc-valu>` and :ref:`MFMA <desc-mfma>` units in
various precisions. We note that unlike the
:ref:`VALU instruction mix <valu-arith-instruction-mix>` and
:ref:`MFMA instruction mix <mfma-instruction-mix>` sections, the metrics here
are reported as FLOPs and IOPs, that is, the total number of operations
executed.

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - VALU FLOPs

     - The total floating-point operations executed per second on the
       :ref:`VALU <desc-valu>`. This is also presented as a percent of the peak
       theoretical FLOPs achievable on the specific accelerator. Note: this does
       not include any floating-point operations from :ref:`MFMA <desc-mfma>`
       instructions.

     - GFLOPs

   * - VALU IOPs

     - The total integer operations executed per second on the
       :ref:`VALU <desc-valu>`. This is also presented as a percent of the peak
       theoretical IOPs achievable on the specific accelerator. Note: this does
       not include any integer operations from :ref:`MFMA <desc-mfma>`
       instructions.

     - GIOPs

   * - MFMA FLOPs (BF16)

     - The total number of 16-bit brain floating point :ref:`MFMA <desc-mfma>`
       operations executed per second. Note: this does not include any 16-bit
       brain floating point operations from :ref:`VALU <desc-valu>`
       instructions. This is also presented as a percent of the peak theoretical
       BF16 MFMA operations achievable on the specific accelerator.

     - GFLOPs

   * - MFMA FLOPs (F16)

     - The total number of 16-bit floating point :ref:`MFMA <desc-mfma>`
       operations executed per second. Note: this does not include any 16-bit
       floating point operations from :ref:`VALU <desc-valu>` instructions. This
       is also presented as a percent of the peak theoretical F16 MFMA
       operations achievable on the specific accelerator.

     - GFLOPs

   * - MFMA FLOPs (F32)

     - The total number of 32-bit floating point :ref:`MFMA <desc-mfma>`
       operations executed per second. Note: this does not include any 32-bit
       floating point operations from :ref:`VALU <desc-valu>` instructions. This
       is also presented as a percent of the peak theoretical F32 MFMA
       operations achievable on the specific accelerator.

     - GFLOPs

   * - MFMA FLOPs (F64)

     - The total number of 64-bit floating point :ref:`MFMA <desc-mfma>`
       operations executed per second. Note: this does not include any 64-bit
       floating point operations from :ref:`VALU <desc-valu>` instructions. This
       is also presented as a percent of the peak theoretical F64 MFMA
       operations achievable on the specific accelerator.

     - GFLOPs

   * - MFMA IOPs (INT8)

     - The total number of 8-bit integer :ref:`MFMA <desc-mfma>` operations
       executed per second. Note: this does not include any 8-bit integer
       operations from :ref:`VALU <desc-valu>` instructions. This is also
       presented as a percent of the peak theoretical INT8 MFMA operations
       achievable on the specific accelerator.

     - GIOPs

.. _pipeline-stats:

Pipeline statistics
-------------------

This section reports a number of key performance characteristics of
various execution units on the :doc:`CU <compute-unit>`. Refer to
:ref:`ipc-example` for a detailed dive into these metrics, and the
:ref:`scheduler <desc-scheduler>` the for a high-level overview of execution
units and instruction issue.

.. list-table::
   :header-rows: 1
   :widths: 20 65 15

   * - Metric

     - Description

     - Unit

   * - IPC

     - The ratio of the total number of instructions executed on the
       :doc:`CU <compute-unit>` over the
       :ref:`total active CU cycles <total-active-cu-cycles>`.

     - Instructions per-cycle

   * - IPC (Issued)

     - The ratio of the total number of
       (non-:ref:`internal <ipc-internal-instructions>`) instructions issued over
       the number of cycles where the :ref:`scheduler <desc-scheduler>` was
       actively working on issuing instructions. Refer to the
       :ref:`Issued IPC <issued-ipc>` example for further detail.

     - Instructions per-cycle

   * - SALU utilization

     - Indicates what percent of the kernel's duration the
       :ref:`SALU <desc-salu>` was busy executing instructions. Computed as the
       ratio of the total number of cycles spent by the
       :ref:`scheduler <desc-scheduler>` issuing SALU / :ref:`SMEM <desc-smem>`
       instructions over the :ref:`total CU cycles <total-cu-cycles>`.

     - Percent

   * - VALU utilization

     - Indicates what percent of the kernel's duration the
       :ref:`VALU <desc-valu>` was busy executing instructions. Does not include
       :ref:`VMEM <desc-vmem>` operations. Computed as the ratio of the total
       number of cycles spent by the :ref:`scheduler <desc-scheduler>` issuing
       VALU instructions over the :ref:`total CU cycles <total-cu-cycles>`.

     - Percent

   * - VMEM utilization

     - Indicates what percent of the kernel's duration the
       :ref:`VMEM <desc-vmem>` unit was busy executing instructions, including
       both global/generic and spill/scratch operations (see the
       :ref:`VMEM instruction count metrics <ta-instruction-counts>` for more
       detail).  Does not include :ref:`VALU <desc-valu>` operations. Computed
       as the ratio of the total number of cycles spent by the
       :ref:`scheduler <desc-scheduler>` issuing VMEM instructions over the
       :ref:`total CU cycles <total-cu-cycles>`.

     - Percent

   * - Branch utilization

     - Indicates what percent of the kernel's duration the
       :ref:`branch <desc-branch>` unit was busy executing instructions.
       Computed as the ratio of the total number of cycles spent by the
       :ref:`scheduler <desc-scheduler>` issuing branch instructions over the
       :ref:`total CU cycles <total-cu-cycles>`.

     - Percent

   * - VALU active threads

     - Indicates the average level of :ref:`divergence <desc-divergence>` within
       a wavefront over the lifetime of the kernel. The number of work-items
       that were active in a wavefront during execution of each
       :ref:`VALU <desc-valu>` instruction, time-averaged over all VALU
       instructions run on all wavefronts in the kernel.

     - Work-items

   * - MFMA utilization

     - Indicates what percent of the kernel's duration the
       :ref:`MFMA <desc-mfma>` unit was busy executing instructions. Computed as
       the ratio of the total number of cycles spent by the
       :ref:`MFMA <desc-salu>` was busy over the
       :ref:`total CU cycles <total-cu-cycles>`.

     - Percent

   * - MFMA instruction cycles

     - The average duration of :ref:`MFMA <desc-mfma>` instructions in this
       kernel in cycles. Computed as the ratio of the total number of cycles the
       MFMA unit was busy over the total number of MFMA instructions. Compare
       to, for example, the
       `AMD Matrix Instruction Calculator <https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator>`_.

     - Cycles per instruction

   * - VMEM latency

     - The average number of round-trip cycles (that is, from issue to data
       return / acknowledgment) required for a VMEM instruction to complete.

     - Cycles

   * - SMEM latency

     - The average number of round-trip cycles (that is, from issue to data
       return / acknowledgment) required for a SMEM instruction to complete.

     - Cycles

.. note::

   The branch utilization reported in this section also includes time spent in
   other instruction types (namely: ``s_endpgm``) that are *typically* a very
   small percentage of the overall kernel execution. This complication is
   omitted for simplicity, but may result in small amounts of branch utilization
   (typically less than 1%) for otherwise branch-less kernels.

.. _arithmetic-operations:

Arithmetic operations
---------------------

This section reports the total number of floating-point and integer
operations executed in various precisions. Unlike the
:ref:`compute-speed-of-light` panel, this section reports both
:ref:`VALU <desc-valu>` and :ref:`MFMA <desc-mfma>` operations of the same precision
(e.g., F32) in the same metric. Additionally, this panel lets the user
control how the data is normalized (i.e., control the
:ref:`normalization unit <normalization-units>`), while the speed-of-light panel does
not. For more detail on how operations are counted see the
:ref:`FLOP counting convention <metrics-flop-count>` section.

.. warning::

   As discussed in :ref:`instruction-mix`, the metrics in this section do not
   take into account the execution mask of the operation, and will report the
   same value even if EXEC is identically zero.

.. list-table::
   :header-rows: 1
   :widths: 18 65 17

   * - Metric

     - Description

     - Unit

   * - FLOPs (Total)

     - The total number of floating-point operations executed on either the
       :ref:`VALU <desc-valu>` or :ref:`MFMA <desc-mfma>` units, per
       :ref:`normalization unit <normalization-units>`.

     - FLOP per :ref:`normalization unit <normalization-units>`

   * - IOPs (Total)

     - The total number of integer operations executed on either the
       :ref:`VALU <desc-valu>` or :ref:`MFMA <desc-mfma>` units, per
       :ref:`normalization unit <normalization-units>`.

     - IOP per :ref:`normalization unit <normalization-units>`

   * - F16 OPs

     - The total number of 16-bit floating-point operations executed on either the
       :ref:`VALU <desc-valu>` or :ref:`MFMA <desc-mfma>` units, per
       :ref:`normalization unit <normalization-units>`.

     - FLOP per :ref:`normalization unit <normalization-units>`

   * - BF16 OPs

     - The total number of 16-bit brain floating-point operations executed on either the
       :ref:`VALU <desc-valu>` or :ref:`MFMA <desc-mfma>` units, per
       :ref:`normalization unit <normalization-units>`. Note: on current CDNA
       accelerators, the VALU has no native BF16 instructions.

     - FLOP per :ref:`normalization unit <normalization-units>`

   * - F32 OPs

     - The total number of 32-bit floating-point operations executed on either
       the :ref:`VALU <desc-valu>` or :ref:`MFMA <desc-mfma>` units, per
       :ref:`normalization unit <normalization-units>`.

     - FLOP per :ref:`normalization unit <normalization-units>`

   * - F64 OPs

     - The total number of 64-bit floating-point operations executed on either
       the :ref:`VALU <desc-valu>` or :ref:`MFMA <desc-mfma>` units, per
       :ref:`normalization unit <normalization-units>`.

     - FLOP per :ref:`normalization unit <normalization-units>`

   * - INT8 OPs

     - The total number of 8-bit integer operations executed on either the
       :ref:`VALU <desc-valu>` or :ref:`MFMA <desc-mfma>` units, per
       :ref:`normalization unit <normalization-units>`. Note: on current CDNA
       accelerators, the VALU has no native INT8 instructions.

     - IOPs per :ref:`normalization unit <normalization-units>`
