.. meta::
   :description: ROCm Compute Profiler performance model: Local data share (LDS)
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, local, data, share, LDS

**********************
Local data share (LDS)
**********************

.. _lds-sol:

LDS Speed-of-Light
==================

.. warning::

   The theoretical maximum throughput for some metrics in this section are
   currently computed with the maximum achievable clock frequency, as reported
   by ``rocminfo``, for an accelerator. This may not be realistic for all
   workloads.

The :ref:`LDS <desc-lds>` speed-of-light chart shows a number of key metrics for
the LDS as a comparison with the peak achievable values of those metrics.

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - Utilization

     - Indicates what percent of the kernel's duration the :ref:`LDS <desc-lds>`
       was actively executing instructions (including, but not limited to, load,
       store, atomic and HIP's ``__shfl`` operations).  Calculated as the ratio
       of the total number of cycles LDS was active over the
       :ref:`total CU cycles <total-cu-cycles>`.

     - Percent

   * - Access Rate

     - Indicates the percentage of SIMDs in the :ref:`VALU <desc-valu>` [#lds-workload]_
       actively issuing LDS instructions, averaged over the lifetime of the
       kernel. Calculated as the ratio of the total number of cycles spent by
       the :ref:`scheduler <desc-scheduler>` issuing :ref:`LDS <desc-lds>`
       instructions over the
       :ref:`total CU cycles <total-cu-cycles>`.

     - Percent

   * - Theoretical Bandwidth (% of Peak)

     - Indicates the maximum amount of bytes that *could* have been loaded from,
       stored to, or atomically updated in the LDS in this kernel, as a percent
       of the peak LDS bandwidth achievable. See the
       :ref:`LDS bandwidth example <lds-bandwidth>` for more detail.

     - Percent

   * - Bank Conflict Rate

     - Indicates the percentage of active LDS cycles that were spent servicing
       bank conflicts. Calculated as the ratio of LDS cycles spent servicing
       bank conflicts over the number of LDS cycles that would have been
       required to move the same amount of data in an uncontended access. [#lds-bank-conflict]_

     - Percent

.. rubric:: Footnotes

.. [#lds-workload] Here we assume the typical case where the workload evenly distributes
   LDS operations over all SIMDs in a CU (that is, waves on different SIMDs are
   executing similar code). For highly unbalanced workloads, where e.g., one
   SIMD pair in the CU does not issue LDS instructions at all, this metric is
   better interpreted as the percentage of SIMDs issuing LDS instructions on
   :ref:`SIMD pairs <desc-lds>` that are actively using the LDS, averaged over
   the lifetime of the kernel.

.. [#lds-bank-conflict] The maximum value of the bank conflict rate is less than 100%
   (specifically: 96.875%), as the first cycle in the
   :ref:`LDS scheduler <desc-lds>` is never considered contended.

.. _lds-stats:

Statistics
==========

The LDS statistics panel gives a more detailed view of the hardware:

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - LDS Instructions

     - The total number of LDS instructions (including, but not limited to,
       read/write/atomics and HIP's ``__shfl`` instructions) executed per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - Theoretical Bandwidth

     - Indicates the maximum amount of bytes that could have been loaded from,
       stored to, or atomically updated in the LDS per
       :ref:`normalization unit <normalization-units>`. Does *not* take into
       account the execution mask of the wavefront when the instruction was
       executed. See the
       :ref:`LDS bandwidth example <lds-bandwidth>` for more detail.

     - Bytes per :ref:`normalization unit <normalization-units>`

   * - LDS Latency

     - The average number of round-trip cycles (i.e., from issue to data-return
       / acknowledgment) required for an LDS instruction to complete.

     - Cycles

   * - Bank Conflicts/Access

     - The ratio of the number of cycles spent in the
       :ref:`LDS scheduler <desc-lds>` due to bank conflicts (as determined by
       the conflict resolution hardware) to the base number of cycles that would
       be spent in the LDS scheduler in a completely uncontended case. This is
       the unnormalized form of the Bank Conflict Rate.

     - Conflicts/Access

   * - Index Accesses

     - The total number of cycles spent in the :ref:`LDS scheduler <desc-lds>`
       over all operations per :ref:`normalization unit <normalization-units>`.

     - Cycles per :ref:`normalization unit <normalization-units>`

   * - Atomic Return Cycles

     - The total number of cycles spent on LDS atomics with return per
       :ref:`normalization unit <normalization-units>`.

     - Cycles per :ref:`normalization unit <normalization-units>`

   * - Bank Conflicts

     - The total number of cycles spent in the :ref:`LDS scheduler <desc-lds>`
       due to bank conflicts (as determined by the conflict resolution hardware)
       per :ref:`normalization unit <normalization-units>`.

     - Cycles per :ref:`normalization unit <normalization-units>`

   * - Address Conflicts

     - The total number of cycles spent in the :ref:`LDS scheduler <desc-lds>`
       due to address conflicts (as determined by the conflict resolution
       hardware) per :ref:`normalization unit <normalization-units>`.

     - Cycles per :ref:`normalization unit <normalization-units>`

   * - Unaligned Stall

     - The total number of cycles spent in the :ref:`LDS scheduler <desc-lds>`
       due to stalls from non-dword aligned addresses per
       :ref:`normalization unit <normalization-units>`.

     - Cycles per :ref:`normalization unit <normalization-units>`

   * - Memory Violations

     - The total number of out-of-bounds accesses made to the LDS, per
       :ref:`normalization unit <normalization-units>`. This is unused and
       expected to be zero in most configurations for modern CDNA™ accelerators.

     - Accesses per :ref:`normalization unit <normalization-units>`
