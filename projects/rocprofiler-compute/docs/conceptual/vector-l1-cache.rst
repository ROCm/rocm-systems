.. meta::
   :description: ROCm Compute Profiler performance model: Vector L1 cache (vL1D)
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, AMD, vector, l1, cache, vl1d

**********************
Vector L1 cache (vL1D)
**********************

The vector L1 data (vL1D) cache is local to each
:doc:`compute unit <compute-unit>` on the accelerator, and handles vector memory
operations issued by a wavefront. The vL1D cache consists of several components:

* An address processing unit, also known as the
  :ref:`texture addresser <desc-ta>` which receives commands (instructions) and
  write/atomic data from the :doc:`compute unit <compute-unit>`, and coalesces
  them into fewer requests for the cache to process.

* An address translation unit, also known as the
  :ref:`L1 Unified Translation Cache (UTCL1) <desc-utcl1>`, that translates
  requests from virtual to physical addresses for lookup in the cache. The
  translation unit has an L1 translation lookaside buffer (L1TLB) to reduce the
  cost of repeated translations.

* A Tag RAM that looks up whether a requested cache line is already
  present in the :ref:`cache <desc-tc>`.

* The result of the Tag RAM lookup is placed in the L1 cache controller
  for routing to the correct location; for instance, the
  :ref:`L2 Memory Interface <vl1d-l2-transaction-detail>` for misses or the
  :ref:`cache RAM <desc-tc>` for hits.

* The cache RAM, also known as the :ref:`texture cache (TC) <desc-tc>`, stores
  requested data for potential reuse. Data returned from the
  :doc:`L2 cache <l2-cache>` is placed into the cache RAM before going down the
  :ref:`data-return path <desc-td>`.

* A backend data processing unit, also known as the
  :ref:`texture data (TD) <desc-td>` that routes data back to the requesting
  :doc:`compute unit <compute-unit>`.

Together, this complex is known as the vL1D, or Texture Cache per Pipe
(TCP). A simplified diagram of the vL1D is presented below:

.. figure:: ../data/performance-model/l1perf_model.png
   :align: center
   :alt: Performance model of the vL1D Cache on AMD Instinct
   :width: 800

   Performance model of the vL1D Cache on AMD Instinct MI-series accelerators.

.. _vl1d-sol:

vL1D Speed-of-Light
===================

.. warning::

   The theoretical maximum throughput for some metrics in this section are
   currently computed with the maximum achievable clock frequency, as reported
   by ``rocminfo``, for an accelerator. This may not be realistic for all
   workloads.

The vL1D’s speed-of-light chart shows several key metrics for the vL1D
as a comparison with the peak achievable values of those metrics.

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - Hit Rate

     - The ratio of the number of vL1D cache line requests that hit [#vl1d-hit]_
       in vL1D cache over the total number of cache line requests to the
       :ref:`vL1D Cache RAM <desc-tc>`.

     - Percent

   * - Bandwidth

     - The number of bytes looked up in the vL1D cache as a result of
       :ref:`VMEM <desc-vmem>` instructions, as a percent of the peak
       theoretical bandwidth achievable on the specific accelerator. The number
       of bytes is calculated as the number of cache lines requested multiplied
       by the cache line size. This value does not consider partial requests, so
       for instance, if only a single value is requested in a cache line, the
       data movement will still be counted as a full cache line.

     - Percent

   * - Utilization

     - Indicates how busy the :ref:`vL1D Cache RAM <desc-tc>` was during the
       kernel execution. The number of cycles where the vL1D Cache RAM is
       actively processing any request divided by the number of cycles where the
       vL1D is active [#vl1d-activity]_.

     - Percent

   * - Coalescing

     - Indicates how well memory instructions were coalesced by the
       :ref:`address processing unit <desc-ta>`, ranging from uncoalesced (25%)
       to fully coalesced (100%). Calculated as the average number of
       :ref:`thread-requests <thread-requests>` generated per instruction
       divided by the ideal number of thread-requests per instruction.

     - Percent

.. _desc-ta:

Address processing unit or Texture Addresser (TA)
=================================================

The :doc:`vL1D <vector-l1-cache>`’s address processing unit receives vector
memory instructions (commands) along with write/atomic data from a
:doc:`compute unit <compute-unit>` and is responsible for coalescing these into
requests for lookup in the :ref:`vL1D RAM <desc-tc>`. The address processor
passes information about the commands (coalescing state, destination SIMD,
etc.) to the :ref:`data processing unit <desc-td>` for use after the requested
data has been retrieved.

ROCm Compute Profiler reports several metrics to indicate performance bottlenecks in
the address processing unit, which are broken down into a few
categories:

-  :ref:`ta-busy-stall`

-  :ref:`ta-instruction-counts`

-  :ref:`ta-spill-stack`

.. _ta-busy-stall:

Busy / stall metrics
--------------------

When executing vector memory instructions, the compute unit must send an
address (and in the case of writes/atomics, data) to the address
processing unit. When the front-end cannot accept any more addresses, it
must backpressure the wave-issue logic for the VMEM pipe and prevent the
issue of further vector memory instructions.

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - Busy

     - Percent of the :ref:`total CU cycles <total-cu-cycles>` the address
       processor was busy

     - Percent

   * - Address Stall

     - Percent of the :ref:`total CU cycles <total-cu-cycles>` the address
       processor was stalled from sending address requests further into the vL1D
       pipeline

     - Percent

   * - Data Stall

     - Percent of the :ref:`total CU cycles <total-cu-cycles>` the address
       processor was stalled from sending write/atomic data further into the
       vL1D pipeline

     - Percent

   * - Data-Processor → Address Stall

     - Percent of :ref:`total CU cycles <total-cu-cycles>` the address processor
       was stalled waiting to send command data to the
       :ref:`data processor <desc-td>`

     - Percent

.. _ta-instruction-counts:

Instruction counts
------------------

The address processor also counts instruction types to give the user
information on what sorts of memory instructions were executed by the
kernel. These are broken down into a few major categories:

.. list-table::
   :header-rows: 1

   * - Memory type

     - Usage

     - Description

   * - Global

     - Global memory

     - Global memory can be seen by all threads from a process. This includes
       the local accelerator's DRAM, remote accelerator's DRAM, and the host's
       DRAM.

   * - Generic

     - Dynamic address spaces

     - Generic memory, or "flat" memory, is used when the compiler cannot
       statically prove that a pointer is to memory in one or the other address
       spaces. The pointer could dynamically point into global, local, constant,
       or private memory.

   * - Private Memory

     - Register spills / Stack memory

     - Private memory, or "scratch" memory, is only visible to a particular
       :ref:`work-item <desc-work-item>` in a particular
       :ref:`workgroup <desc-workgroup>`. On AMD Instinct™ MI-series
       accelerators, private memory is used to implement both register spills
       and stack memory accesses.

The address processor counts these instruction types as follows:

.. list-table::
   :header-rows: 1

   * - Type

     - Description

     - Unit

   * - Global/Generic

     - The total number of global & generic memory instructions executed on all
       :doc:`compute units <compute-unit>` on the accelerator, per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - Global/Generic Read

     - The total number of global & generic memory read instructions executed on
       all :doc:`compute units <compute-unit>` on the accelerator, per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - Global/Generic Write

     - The total number of global & generic memory write instructions executed
       on all :doc:`compute units <compute-unit>` on the accelerator, per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - Global/Generic Atomic

     - The total number of global & generic memory atomic (with and without
       return) instructions executed on all :doc:`compute units <compute-unit>`
       on the accelerator, per :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - Spill/Stack

     - The total number of spill/stack memory instructions executed on all
       :doc:`compute units <compute-unit>` on the accelerator, per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - Spill/Stack Read

     - The total number of spill/stack memory read instructions executed on all
       :doc:`compute units <compute-unit>` on the accelerator, per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - Spill/Stack Write

     - The total number of spill/stack memory write instructions executed on all
       :doc:`compute units <compute-unit>` on the accelerator, per
       :ref:`normalization unit <normalization-units>`.

     - Instruction per :ref:`normalization unit <normalization-units>`

   * - Spill/Stack Atomic

     - The total number of spill/stack memory atomic (with and without return)
       instructions executed on all :doc:`compute units <compute-unit>` on the
       accelerator, per :ref:`normalization unit <normalization-units>`.
       Typically unused as these memory operations are typically used to
       implement thread-local storage.

     - Instructions per :ref:`normalization unit <normalization-units>`

.. note::

   The above is a simplified model specifically for the HIP programming language
   that does not consider inline assembly usage, constant memory usage or
   texture memory.

   These categories correspond to:

   * Global/Generic: global and flat memory operations, that are used for global
     and generic memory access.

   * Spill/Stack: buffer instructions which are used on the MI50, MI100, and
     :ref:`MI2XX <mixxx-note>` accelerators for register spills / stack memory.

   These concepts are described in more detail in the :ref:`memory-spaces`,
   while generic memory access is explored in the
   :ref:`generic memory benchmark <flat-memory-ex>` section.

.. _ta-spill-stack:

Spill / stack metrics
---------------------

Finally, the address processing unit contains a separate coalescing
stage for spill/stack memory, and thus reports:

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - Spill/Stack Total Cycles

     - The number of cycles the address processing unit spent working on
       spill/stack instructions, per
       :ref:`normalization unit <normalization-units>`.

     - Cycles per :ref:`normalization unit <normalization-units>`

   * - Spill/Stack Coalesced Read Cycles

     - The number of cycles the address processing unit spent working on
       coalesced spill/stack read instructions, per
       :ref:`normalization unit <normalization-units>`.

     - Cycles per :ref:`normalization unit <normalization-units>`

   * - Spill/Stack Coalesced Write Cycles

     - The number of cycles the address processing unit spent working on
       coalesced spill/stack write instructions, per
       :ref:`normalization unit <normalization-units>`.

     - Cycles per :ref:`normalization unit <normalization-units>`

.. _desc-utcl1:

L1 Unified Translation Cache (UTCL1)
====================================

After a vector memory instruction has been processed/coalesced by the
address processing unit of the vL1D, it must be translated from a
virtual to physical address. This process is handled by the L1 Unified
Translation Cache (UTCL1). This cache contains a L1 Translation
Lookaside Buffer (TLB) which stores recently translated addresses to
reduce the cost of subsequent re-translations.

ROCm Compute Profiler reports the following L1 TLB metrics:

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - Requests

     - The number of translation requests made to the UTCL1 per
       :ref:`normalization unit <normalization-units>`.

     - Requests per :ref:`normalization unit <normalization-units>`

   * - Hits

     - The number of translation requests that hit in the UTCL1, and could be
       reused, per :ref:`normalization unit <normalization-units>`.

     - Requests per :ref:`normalization unit <normalization-units>`

   * - Hit Ratio

     - The ratio of the number of translation requests that hit in the UTCL1
       divided by the total number of translation requests made to the UTCL1.

     - Percent

   * - Translation Misses

     - The total number of translation requests that missed in the UTCL1 due to
       translation not being present in the cache, per
       :ref:`normalization unit <normalization-units>`.

     - Requests per :ref:`normalization unit <normalization-units>`

   * - Permission Misses

     - The total number of translation requests that missed in the UTCL1 due to
       a permission error, per :ref:`normalization unit <normalization-units>`.
       This is unused and expected to be zero in most configurations for modern
       CDNA™ accelerators.

     - Requests per :ref:`normalization unit <normalization-units>`

.. note::

   On current CDNA accelerators, such as the :ref:`MI2XX <mixxx-note>`, the
   UTCL1 does *not* count hit-on-miss requests.

.. _desc-tc:

Vector L1 Cache RAM or Texture Cache (TC)
=========================================

After coalescing in the :ref:`address processing unit <desc-ta>` of the v1LD,
and address translation in the :ref:`L1 TLB <desc-utcl1>` the request proceeds
to the Cache RAM stage of the pipeline. Incoming requests are looked up
in the cache RAMs using parts of the physical address as a tag. Hits
will be returned through the :ref:`data-return path <desc-td>`, while misses
will routed out to the :doc:`L2 Cache <l2-cache>` for servicing.

The metrics tracked by the vL1D RAM include:

-  :ref:`Stall metrics <vl1d-cache-stall-metrics>`

-  :ref:`Cache access metrics <vl1d-cache-access-metrics>`

-  :ref:`vL1D-L2 transaction detail metrics <vl1d-l2-transaction-detail>`

.. _vl1d-cache-stall-metrics:

vL1D cache stall metrics
------------------------

The vL1D also reports where it is stalled in the pipeline, which may
indicate performance limiters of the cache. A stall in the pipeline may
result in backpressuring earlier parts of the pipeline, e.g., a stall on
L2 requests may backpressure the wave-issue logic of the :ref:`VMEM <desc-vmem>`
pipe and prevent it from issuing more vector memory instructions until
the vL1D’s outstanding requests are completed.

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - Stalled on L2 Data

     - The ratio of the number of cycles where the vL1D is stalled waiting for
       requested data to return from the :doc:`L2 cache <l2-cache>` divided by
       the number of cycles where the vL1D is active [#vl1d-activity]_.

     - Percent

   * - Stalled on L2 Requests

     - The ratio of the number of cycles where the vL1D is stalled waiting to
       issue a request for data to the :doc:`L2 cache <l2-cache>` divided by the
       number of cycles where the vL1D is active [#vl1d-activity]_.

     - Percent

   * - Tag RAM Stall (Read/Write/Atomic)

     - The ratio of the number of cycles where the vL1D is stalled due to
       Read/Write/Atomic requests with conflicting tags being looked up
       concurrently, divided by the number of cycles where the
       vL1D is active [#vl1d-activity]_.

     - Percent

.. _vl1d-cache-access-metrics:

vL1D cache access metrics
-------------------------

The vL1D cache access metrics broadly indicate the type of requests
incoming from the :ref:`cache front-end <desc-ta>`, the number of requests that
were serviced by the vL1D, and the number & type of outgoing requests to
the :doc:`L2 cache <l2-cache>`. In addition, this section includes the
approximate latencies of accesses to the cache itself, along with
latencies of read/write memory operations to the :doc:`L2 cache <l2-cache>`.

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - Total Requests

     - The total number of incoming requests from the
       :ref:`address processing unit <desc-ta>` after coalescing.

     - Requests

   * - Total read/write/atomic requests

     - The total number of incoming read/write/atomic requests from the
       :ref:`address processing unit <desc-ta>` after coalescing per
       :ref:`normalization unit <normalization-units>`

     - Requests per :ref:`normalization unit <normalization-units>`

   * - Cache Bandwidth

     - The number of bytes looked up in the vL1D cache as a result of
       :ref:`VMEM <desc-vmem>` instructions per
       :ref:`normalization unit <normalization-units>`.  The number of bytes is
       calculated as the number of cache lines requested multiplied by the cache
       line size.  This value does not consider partial requests, so for
       instance, if only a single value is requested in a cache line, the data
       movement will still be counted as a full cache line.

     - Bytes per :ref:`normalization unit <normalization-units>`

   * - Cache Hit Rate [#vl1d-hit]_

     - The ratio of the number of vL1D cache line requests that hit in vL1D
       cache over the total number of cache line requests to the
       :ref:`vL1D Cache RAM <desc-tc>`.

     - Percent

   * - Cache Accesses

     - The total number of cache line lookups in the vL1D.

     - Cache lines

   * - Cache Hits [#vl1d-hit]_

     - The number of cache accesses minus the number of outgoing requests to the
       :doc:`L2 cache <l2-cache>`, that is, the number of cache line requests
       serviced by the :ref:`vL1D Cache RAM <desc-tc>` per
       :ref:`normalization unit <normalization-units>`.

     - Cache lines per :ref:`normalization unit <normalization-units>`

   * - Invalidations

     - The number of times the vL1D was issued a write-back invalidate command
       during the kernel's execution per
       :ref:`normalization unit <normalization-units>`.  This may be triggered
       by, for instance, the ``buffer_wbinvl1`` instruction.

     - Invalidations per :ref:`normalization unit <normalization-units>`

   * - L1-L2 Bandwidth

     - The number of bytes transferred across the vL1D-L2 interface as a result
       of :ref:`VMEM <desc-vmem>` instructions, per
       :ref:`normalization unit <normalization-units>`. The number of bytes is
       calculated as the number of cache lines requested multiplied by the cache
       line size. This value does not consider partial requests, so for
       instance, if only a single value is requested in a cache line, the data
       movement will still be counted as a full cache line.

     - Bytes per :ref:`normalization unit <normalization-units>`

   * - L1-L2 Reads

     - The number of read requests for a vL1D cache line that were not satisfied
       by the vL1D and must be retrieved from the to the
       :doc:`L2 Cache <l2-cache>` per
       :ref:`normalization unit <normalization-units>`.

     - Requests per :ref:`normalization unit <normalization-units>`

   * - L1-L2 Writes

     - The number of write requests to a vL1D cache line that were sent through
       the vL1D to the :doc:`L2 cache <l2-cache>`, per
       :ref:`normalization unit <normalization-units>`.

     - Requests per :ref:`normalization unit <normalization-units>`

   * - L1-L2 Atomics

     - The number of atomic requests that are sent through the vL1D to the
       :doc:`L2 cache <l2-cache>`, per
       :ref:`normalization unit <normalization-units>`. This includes requests
       for atomics with, and without return.

     - Requests per :ref:`normalization unit <normalization-units>`

   * - L1 Access Latency

     - Calculated as the average number of cycles that a vL1D cache line request
       spent in the vL1D cache pipeline.

     - Cycles

   * - L1-L2 Read Access Latency

     - Calculated as the average number of cycles that the vL1D cache took to
       issue and receive read requests from the :doc:`L2 Cache <l2-cache>`. This
       number also includes requests for atomics with return values.

     - Cycles

   * - L1-L2 Write Access Latency

     - Calculated as the average number of cycles that the vL1D cache took to
       issue and receive acknowledgement of a write request to the
       :doc:`L2 Cache <l2-cache>`. This number also includes requests for
       atomics without return values.

     - Cycles

.. note::

   All cache accesses in vL1D are for a single cache line's worth of data.
   The size of a cache line may vary, however on current AMD Instinct MI CDNA
   accelerators and GCN™ GPUs the L1 cache line size is 64B.

.. rubric :: Footnotes

.. [#vl1d-hit] The vL1D cache on AMD Instinct MI-series CDNA accelerators
   uses a "hit-on-miss" approach to reporting cache hits. That is, if while
   satisfying a miss, another request comes in that would hit on the same
   pending cache line, the subsequent request will be counted as a "hit".
   Therefore, it is also important to consider the access latency metric in the
   :ref:`Cache access metrics <vl1d-cache-stall-metrics>` section when
   evaluating the vL1D hit rate.

.. [#vl1d-activity] ROCm Compute Profiler considers the vL1D to be active when any part of
   the vL1D (excluding the :ref:`address processor <desc-ta>` and
   :ref:`data return <desc-td>` units) are active, for example, when performing
   a translation, waiting for data, accessing the Tag or Cache RAMs, etc.

.. _vl1d-l2-transaction-detail:

vL1D - L2 Transaction Detail
----------------------------

This section provides a more granular look at the types of requests made
to the :doc:`L2 cache <l2-cache>`. These are broken down by the operation type
(read / write / atomic, with, or without return), and the
:ref:`memory type <memory-type>`.

.. _desc-td:

Vector L1 data-return path or Texture Data (TD)
===============================================

The data-return path of the vL1D cache, also known as the Texture Data
(TD) unit, is responsible for routing data returned from the
:ref:`vL1D cache RAM <desc-tc>` back to a wavefront on a SIMD. As described in
the :ref:`vL1D cache front-end <desc-ta>` section, the data-return path is passed
information about the space requirements and routing for data requests
from the :ref:`VALU <desc-valu>`. When data is returned from the
:ref:`vL1D cache RAM <desc-tc>`, it is matched to this previously stored request
data, and returned to the appropriate SIMD.

ROCm Compute Profiler reports the following vL1D data-return path metrics:

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - Data-return Busy

     - Percent of the :ref:`total CU cycles <total-cu-cycles>` the data-return
       unit was busy processing or waiting on data to return to the
       :doc:`CU <compute-unit>`.

     - Percent

   * - Cache RAM → Data-return Stall

     - Percent of the :ref:`total CU cycles <total-cu-cycles>` the data-return
       unit was stalled on data to be returned from the
       :ref:`vL1D Cache RAM <desc-tc>`.

     - Percent

   * - Workgroup manager → Data-return Stall

     - Percent of the :ref:`total CU cycles <total-cu-cycles>` the data-return
       unit was stalled by the :ref:`workgroup manager <desc-spi>` due to
       initialization of registers as a part of launching new workgroups.

     - Percent

   * - Coalescable Instructions

     - The number of instructions submitted to the
       :ref:`data-return unit <desc-td>` by the
       :ref:`address processor <desc-ta>` that were found to be coalescable, per
       :ref:`normalization unit <normalization-units>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - Read Instructions

     - The number of read instructions submitted to the
       :ref:`data-return unit <desc-td>` by the
       :ref:`address processor <desc-ta>` summed over all
       :doc:`compute units <compute-unit>` on the accelerator, per
       :ref:`normalization unit <normalization-units>`. This is expected to be
       the sum of global/generic and spill/stack reads in the
       :ref:`address processor <desc-ta>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - Write Instructions

     - The number of store instructions submitted to the
       :ref:`data-return unit <desc-td>` by the
       :ref:`address processor <desc-ta>` summed over all
       :doc:`compute units <compute-unit>` on the accelerator, per
       :ref:`normalization unit <normalization-units>`. This is expected to be
       the sum of global/generic and spill/stack stores counted by the
       :ref:`vL1D cache-front-end <ta-instruction-counts>`.

     - Instructions per :ref:`normalization unit <normalization-units>`

   * - Atomic Instructions

     - The number of atomic instructions submitted to the
       :ref:`data-return unit <desc-td>` by the
       :ref:`address processor <desc-ta>` summed over all
       :doc:`compute units <compute-unit>` on the accelerator, per
       :ref:`normalization unit <normalization-units>`. This is expected to be
       the sum of global/generic and spill/stack atomics in the
       :ref:`address processor <desc-ta>`.

     - Instructions per :ref:`normalization unit <normalization-units>`
