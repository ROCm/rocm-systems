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

.. jinja:: wavefront-launch-stats
   :file: _templates/metrics_table.j2

.. _wavefront-runtime-stats:

Wavefront runtime stats
-----------------------

The wavefront runtime statistics gives a high-level overview of the
execution of wavefronts in a kernel:

.. jinja:: wavefront-runtime-stats
   :file: _templates/metrics_table.j2

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

.. jinja:: instruction-mix
   :file: _templates/metrics_table.j2

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

.. jinja:: valu-arith-instruction-mix
   :file: _templates/metrics_table.j2

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

.. jinja:: mfma-instruction-mix
   :file: _templates/metrics_table.j2

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

.. jinja:: compute-speed-of-light
   :file: _templates/metrics_table.j2

.. _pipeline-stats:

Pipeline statistics
-------------------

This section reports a number of key performance characteristics of
various execution units on the :doc:`CU <compute-unit>`. Refer to
:ref:`ipc-example` for a detailed dive into these metrics, and the
:ref:`scheduler <desc-scheduler>` the for a high-level overview of execution
units and instruction issue.

.. jinja:: pipeline-stats
   :file: _templates/metrics_table.j2

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

.. jinja:: arithmetic-operations
   :file: _templates/metrics_table.j2