.. meta::
   :description: ROCm Compute Profiler performance model: System Speed-of-Light
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, AMD, system, speed of light

*********************
System Speed-of-Light
*********************

System Speed-of-Light summarizes some of the key metrics from various sections
of ROCm Compute Profiler’s profiling report.

.. warning::

   The theoretical maximum throughput for some metrics in this section are
   currently computed with the maximum achievable clock frequency, as reported
   by ``rocminfo``, for an accelerator. This may not be realistic for
   all workloads.

   Also, not all metrics -- such as FLOP counters -- are available on all AMD
   Instinct™ MI-series accelerators. For more detail on how operations are
   counted, see the :ref:`metrics-flop-count` section.

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - :ref:`VALU <desc-valu>` FLOPs

     - The total floating-point operations executed per second on the
       :ref:`VALU <desc-valu>`.  This is also presented as a percent of the peak
       theoretical FLOPs achievable on the specific accelerator. Note: this does
       not include any floating-point operations from :ref:`MFMA <desc-mfma>`
       instructions.

     - GFLOPs

   * - :ref:`VALU <desc-valu>` IOPs

     - The total integer operations executed per second on the
       :ref:`VALU <desc-valu>`. This is also presented as a percent of the peak
       theoretical IOPs achievable on the specific accelerator. Note: this does
       not include any integer operations from :ref:`MFMA <desc-mfma>`
       instructions.

     - GIOPs

   * - :ref:`MFMA <desc-mfma>` FLOPs (F8)

     - The total number of 8-bit floating point :ref:`MFMA <desc-mfma>`
       operations executed per second. This does not include any 16-bit
       brain floating point operations from :ref:`VALU <desc-valu>`
       instructions. This is also presented as a percent of the peak theoretical
       F8 MFMA operations achievable on the specific accelerator. It is supported on AMD Instinct MI300 series and later only.

     - GFLOPs

   * - :ref:`MFMA <desc-mfma>` FLOPs (BF16)

     - The total number of 16-bit brain floating point :ref:`MFMA <desc-mfma>`
       operations executed per second. Note: this does not include any 16-bit
       brain floating point operations from :ref:`VALU <desc-valu>`
       instructions. This is also presented as a percent of the peak theoretical
       BF16 MFMA operations achievable on the specific accelerator.

     - GFLOPs

   * - :ref:`MFMA <desc-mfma>` FLOPs (F16)

     - The total number of 16-bit floating point :ref:`MFMA <desc-mfma>`
       operations executed per second. Note: this does not include any 16-bit
       floating point operations from :ref:`VALU <desc-valu>` instructions. This
       is also presented as a percent of the peak theoretical F16 MFMA
       operations achievable on the specific accelerator.

     - GFLOPs

   * - :ref:`MFMA <desc-mfma>` FLOPs (F32)

     - The total number of 32-bit floating point :ref:`MFMA <desc-mfma>`
       operations executed per second. Note: this does not include any 32-bit
       floating point operations from :ref:`VALU <desc-valu>` instructions. This
       is also presented as a percent of the peak theoretical F32 MFMA
       operations achievable on the specific accelerator.

     - GFLOPs

   * - :ref:`MFMA <desc-mfma>` FLOPs (F64)

     - The total number of 64-bit floating point :ref:`MFMA <desc-mfma>`
       operations executed per second. Note: this does not include any 64-bit
       floating point operations from :ref:`VALU <desc-valu>` instructions. This
       is also presented as a percent of the peak theoretical F64 MFMA
       operations achievable on the specific accelerator.

     - GFLOPs

   * - :ref:`MFMA <desc-mfma>` IOPs (INT8)

     - The total number of 8-bit integer :ref:`MFMA <desc-mfma>` operations
       executed per second. Note: this does not include any 8-bit integer
       operations from :ref:`VALU <desc-valu>` instructions. This is also
       presented as a percent of the peak theoretical INT8 MFMA operations
       achievable on the specific accelerator.

     - GIOPs

   * - :ref:`SALU <desc-salu>` utilization

     - Indicates what percent of the kernel's duration the
       :ref:`SALU <desc-salu>` was busy executing instructions. Computed as the
       ratio of the total number of cycles spent by the
       :ref:`scheduler <desc-scheduler>` issuing :ref:`SALU <desc-salu>` or
       :ref:`SMEM <desc-salu>` instructions over the
       :ref:`total CU cycles <total-cu-cycles>`.

     - Percent

   * - :ref:`VALU <desc-valu>` utilization

     - Indicates what percent of the kernel's duration the
       :ref:`VALU <desc-valu>` was busy executing instructions. Does not include
       :ref:`VMEM <desc-vmem>` operations.  Computed as the ratio of the total
       number of cycles spent by the :ref:`scheduler <desc-scheduler>` issuing
       :ref:`VALU <desc-valu>` instructions over the
       :ref:`total CU cycles <total-cu-cycles>`.

     - Percent

   * - :ref:`MFMA <desc-mfma>` utilization

     - Indicates what percent of the kernel's duration the
       :ref:`MFMA <desc-mfma>` unit was busy executing instructions. Computed as
       the ratio of the total number of cycles the MFMA was busy over the
       :ref:`total CU cycles <total-cu-cycles>`.

     - Percent

   * - :ref:`VMEM <desc-valu>` utilization

     - Indicates what percent of the kernel's duration the
       :ref:`VMEM <desc-valu>` unit was busy executing instructions, including
       both global/generic and spill/scratch operations (see the
       :ref:`VMEM instruction count metrics <ta-instruction-counts>`) for more
       detail). Does not include :ref:`VALU <desc-valu>` operations. Computed as
       the ratio of the total number of cycles spent by the
       :ref:`scheduler <desc-scheduler>` issuing VMEM instructions over the
       :ref:`total CU cycles <total-cu-cycles>`.

     - Percent

   * - :ref:`Branch <desc-branch>` utilization

     - Indicates what percent of the kernel's duration the
       :ref:`branch <desc-branch>` unit was busy executing instructions.
       Computed as the ratio of the total number of cycles spent by the
       :ref:`scheduler <desc-scheduler>` issuing :ref:`branch <desc-branch>`
       instructions over the :ref:`total CU cycles <total-cu-cycles>`

     - Percent

   * - :ref:`VALU <desc-valu>` active threads

     - Indicates the average level of :ref:`divergence <desc-divergence>` within
       a wavefront over the lifetime of the kernel. The number of work-items
       that were active in a wavefront during execution of each
       :ref:`VALU <desc-valu>` instruction, time-averaged over all VALU
       instructions run on all wavefronts in the kernel.

     - Work-items

   * - IPC

     - The ratio of the total number of instructions executed on the
       :doc:`CU <compute-unit>` over the
       :ref:`total active CU cycles <total-active-cu-cycles>`. This is also
       presented as a percent of the peak theoretical bandwidth achievable on
       the specific accelerator.

     - Instructions per-cycle

   * - Wavefront occupancy

     - The time-averaged number of wavefronts resident on the accelerator over
       the lifetime of the kernel. Note: this metric may be inaccurate for
       short-running kernels (less than 1ms). This is also presented as a
       percent of the peak theoretical occupancy achievable on the specific
       accelerator.

     - Wavefronts

   * - :doc:`LDS <local-data-share>` theoretical bandwidth

     - Indicates the maximum amount of bytes that could have been loaded from,
       stored to, or atomically updated in the LDS per unit time (see
       :ref:`LDS Bandwidth <lds-bandwidth>` example for more detail). This is
       also presented as a percent of the peak theoretical F64 MFMA operations
       achievable on the specific accelerator.

     - GB/s

   * - :doc:`LDS <local-data-share>` bank conflicts/access

     - The ratio of the number of cycles spent in the
       :doc:`LDS scheduler <local-data-share>` due to bank conflicts (as
       determined by the conflict resolution hardware) to the base number of
       cycles that would be spent in the LDS scheduler in a completely
       uncontended case. This is also presented in normalized form (i.e., the
       Bank Conflict Rate).

     - Conflicts/Access

   * - :doc:`vL1D <vector-l1-cache>` cache hit rate

     - The ratio of the number of vL1D cache line requests that hit in vL1D
       cache over the total number of cache line requests to the
       :ref:`vL1D cache RAM <desc-tc>`.

     - Percent

   * - :doc:`vL1D <vector-l1-cache>` cache bandwidth

     - The number of bytes looked up in the vL1D cache as a result of
       :ref:`VMEM <desc-vmem>` instructions per unit time. The number of bytes
       is calculated as the number of cache lines requested multiplied by the
       cache line size. This value does not consider partial requests, so e.g.,
       if only a single value is requested in a cache line, the data movement
       will still be counted as a full cache line. This is also presented as a
       percent of the peak theoretical bandwidth achievable on the specific
       accelerator.

     - GB/s

   * - :doc:`L2 <l2-cache>` cache hit rate

     - The ratio of the number of L2 cache line requests that hit in the L2
       cache over the total number of incoming cache line requests to the L2
       cache.

     - Percent

   * - :doc:`L2 <l2-cache>` cache bandwidth

     - The number of bytes looked up in the L2 cache per unit time.  The number
       of bytes is calculated as the number of cache lines requested multiplied
       by the cache line size. This value does not consider partial requests, so
       e.g., if only a single value is requested in a cache line, the data
       movement will still be counted as a full cache line. This is also
       presented as a percent of the peak theoretical bandwidth achievable on
       the specific accelerator.

     - GB/s

   * - :doc:`L2 <l2-cache>`-fabric read BW

     - The number of bytes read by the L2 over the
       :ref:`Infinity Fabric™ interface <l2-fabric>` per unit time. This is also
       presented as a percent of the peak theoretical bandwidth achievable on
       the specific accelerator.

     - GB/s

   * - :doc:`L2 <l2-cache>`-fabric write and atomic BW

     - The number of bytes sent by the L2 over the
       :ref:`Infinity Fabric interface <l2-fabric>` by write and atomic
       operations per unit time. This is also presented as a percent of the peak
       theoretical bandwidth achievable on the specific accelerator.

     - GB/s

   * - :doc:`L2 <l2-cache>`-fabric read latency

     - The time-averaged number of cycles read requests spent in Infinity Fabric
       before data was returned to the L2.

     - Cycles

   * - :doc:`L2 <l2-cache>`-fabric write latency

     - The time-averaged number of cycles write requests spent in Infinity
       Fabric before a completion acknowledgement was returned to the L2.

     - Cycles

   * - :ref:`sL1D <desc-sl1d>` cache hit rate

     - The percent of sL1D requests that hit on a previously loaded line the
       cache. Calculated as the ratio of the number of sL1D requests that hit
       over the number of all sL1D requests.

     - Percent

   * - :ref:`sL1D <desc-sl1d>` bandwidth

     - The number of bytes looked up in the sL1D cache per unit time. This is
       also presented as a percent of the peak theoretical bandwidth achievable
       on the specific accelerator.

     - GB/s

   * - :ref:`L1I <desc-l1i>` bandwidth

     - The number of bytes looked up in the L1I cache per unit time. This is
       also presented as a percent of the peak theoretical bandwidth achievable
       on the specific accelerator.

     - GB/s

   * - :ref:`L1I <desc-l1i>` cache hit rate

     - The percent of L1I requests that hit on a previously loaded line the
       cache. Calculated as the ratio of the number of L1I requests that hit
       over the number of all L1I requests.

     - Percent

   * - :ref:`L1I <desc-l1i>` fetch latency

     - The average number of cycles spent to fetch instructions to a
       :doc:`CU <compute-unit>`.

     - Cycles
