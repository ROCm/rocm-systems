.. meta::
   :description: ROCm Compute Profiler performance model: L2 cache (TCC)
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, L2, cache, infinity fabric, metrics

**************
L2 cache (TCC)
**************

The L2 cache is the coherence point for current AMD Instinct™ MI-series GCN™
GPUs and CDNA™ accelerators, and is shared by all :doc:`CUs <compute-unit>`
on the device. Besides serving requests from the
:doc:`vector L1 data caches <vector-l1-cache>`, the L2 cache also is responsible
for servicing requests from the :ref:`L1 instruction caches <desc-l1i>`, the
:ref:`scalar L1 data caches <desc-sL1D>` and the
:doc:`command processor <command-processor>`. The L2 cache is composed of a
number of distinct channels (32 on MI100 and :ref:`MI2XX <mixxx-note>` series CDNA
accelerators at 256B address interleaving) which can largely operate
independently. Mapping of incoming requests to a specific L2 channel is
determined by a hashing mechanism that attempts to evenly distribute requests
across the L2 channels. Requests that miss in the L2 cache are passed out to
:ref:`Infinity Fabric™ <l2-fabric>` to be routed to the appropriate memory
location.

The L2 cache metrics reported by ROCm Compute Profiler are broken down into four
categories:

*  :ref:`L2 Speed-of-Light <l2-sol>`

*  :ref:`L2 cache accesses <l2-cache-accesses>`

*  :ref:`L2-Fabric transactions <l2-fabric>`

*  :ref:`L2-Fabric stalls <l2-fabric-stalls>`

.. _l2-sol:

L2 Speed-of-Light
=================

.. warning::

   The theoretical maximum throughput for some metrics in this section
   are currently computed with the maximum achievable clock frequency, as
   reported by ``rocminfo``, for an accelerator. This may not be realistic for
   all workloads.

The L2 cache’s speed-of-light table contains a few key metrics about the
performance of the L2 cache, aggregated over all the L2 channels, as a
comparison with the peak achievable values of those metrics:

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - Utilization

     - The ratio of the
       :ref:`number of cycles an L2 channel was active, summed over all L2 channels on the accelerator <total-active-l2-cycles>`
       over the :ref:`total L2 cycles <total-l2-cycles>`.

     - Percent

   * - Bandwidth

     - The number of bytes looked up in the L2 cache, as a percent of the peak
       theoretical bandwidth achievable on the specific accelerator. The number
       of bytes is calculated as the number of cache lines requested multiplied
       by the cache line size. This value does not consider partial requests, so
       e.g., if only a single value is requested in a cache line, the data
       movement will still be counted as a full cache line.

     - Percent

   * - Hit Rate

     - The ratio of the number of L2 cache line requests that hit in the L2
       cache over the total number of incoming cache line requests to the L2
       cache.

     - Percent

   * - L2-Fabric Read BW

     - The number of bytes read by the L2 over the
       :ref:`Infinity Fabric interface <l2-fabric>` per unit time.

     - GB/s

   * - L2-Fabric Write and Atomic BW

     - The number of bytes sent by the L2 over the
       :ref:`Infinity Fabric interface <l2-fabric>` by write and atomic
       operations per unit time.

     - GB/s

.. note::

   The L2 cache on AMD Instinct MI CDNA accelerators uses a "hit-on-miss"
   approach to reporting cache hits. That is, if while satisfying a miss,
   another request comes in that would hit on the same pending cache line, the
   subsequent request will be counted as a 'hit'. Therefore, it is also
   important to consider the latency metric in the :ref:`L2-Fabric <l2-fabric>`
   section when evaluating the L2 hit rate.

.. _l2-cache-accesses:

L2 cache accesses
=================

This section details the incoming requests to the L2 cache from the
:doc:`vL1D <vector-l1-cache>` and other clients -- for instance, the
:ref:`sL1D <desc-sL1D>` and :ref:`L1I <desc-l1i>` caches.

.. list-table::
   :header-rows: 1
   :widths: 13 70 17

   * - Metric

     - Description

     - Unit

   * - Bandwidth

     - The number of bytes looked up in the L2 cache, per
       :ref:`normalization unit <normalization-units>`.  The number of bytes is
       calculated as the number of cache lines requested multiplied by the cache
       line size. This value does not consider partial requests, so for example,
       if only a single value is requested in a cache line, the data movement
       will still be counted as a full cache line.

     - Bytes per :ref:`normalization unit <normalization-units>`.

   * - Requests

     - The total number of incoming requests to the L2 from all clients for all
       request types, per :ref:`normalization unit <normalization-units>`.

     - Requests per :ref:`normalization unit <normalization-units>`.

   * - Read Requests

     - The total number of read requests to the L2 from all clients.

     - Requests per :ref:`normalization unit <normalization-units>`

   * - Write Requests

     - The total number of write requests to the L2 from all clients.

     - Requests per :ref:`normalization unit <normalization-units>`

   * - Atomic Requests

     - The total number of atomic requests (with and without return) to the L2
       from all clients.

     - Requests per :ref:`normalization unit <normalization-units>`

   * - Streaming Requests

     - The total number of incoming requests to the L2 that are marked as
       *streaming*. The exact meaning of this may differ depending on the
       targeted accelerator, however on an :ref:`MI2XX <mixxx-note>` this
       corresponds to
       `non-temporal load or stores <https://clang.llvm.org/docs/LanguageExtensions.html#non-temporal-load-store-builtins>`_.
       The L2 cache attempts to evict *streaming* requests before normal
       requests when the L2 is at capacity.

     - Requests per :ref:`normalization unit <normalization-units>`

   * - Probe Requests

     - The number of coherence probe requests made to the L2 cache from outside
       the accelerator. On an :ref:`MI2XX <mixxx-note>`, probe requests may be
       generated by, for example, writes to
       :ref:`fine-grained device <memory-type>` memory or by writes to
       :ref:`coarse-grained <memory-type>` device memory.

     - Requests per :ref:`normalization unit <normalization-units>`

   * - Hit Rate

     - The ratio of the number of L2 cache line requests that hit in the L2
       cache over the total number of incoming cache line requests to the L2
       cache.

     - Percent

   * - Hits

     - The total number of requests to the L2 from all clients that hit in the
       cache. As noted in the :ref:`Speed-of-Light <l2-sol>` section, this
       includes hit-on-miss requests.

     - Requests per :ref:`normalization unit <normalization-units>`

   * - Misses

     - The total number of requests to the L2 from all clients that miss in the
       cache. As noted in the :ref:`Speed-of-Light <l2-sol>` section, these do
       not include hit-on-miss requests.

     - Requests per :ref:`normalization unit <normalization-units>`

   * - Writebacks

     - The total number of L2 cache lines written back to memory for any reason.
       Write-backs may occur due to user code (such as HIP kernel calls to
       ``__threadfence_system`` or atomic built-ins) by the
       :doc:`command processor <command-processor>`'s memory acquire/release
       fences, or for other internal hardware reasons.

     - Cache lines per :ref:`normalization unit <normalization-units>`

   * - Writebacks (Internal)

     - The total number of L2 cache lines written back to memory for internal
       hardware reasons, per :ref:`normalization unit <normalization-units>`.

     - Cache lines per :ref:`normalization unit <normalization-units>`.

   * - Writebacks (vL1D Req)

     - The total number of L2 cache lines written back to memory due to requests
       initiated by the :doc:`vL1D cache <vector-l1-cache>`, per
       :ref:`normalization unit <normalization-units>`.

     - Cache lines per :ref:`normalization unit <normalization-units>`.

   * - Evictions (Normal)

     - The total number of L2 cache lines evicted from the cache due to capacity
       limits, per :ref:`normalization unit <normalization-units>`.

     - Cache lines per :ref:`normalization unit <normalization-units>`.

   * - Evictions (vL1D Req)

     - The total number of L2 cache lines evicted from the cache due to
       invalidation requests initiated by the
       :doc:`vL1D cache <vector-l1-cache>`, per
       :ref:`normalization unit <normalization-units>`.

     - Cache lines per :ref:`normalization unit <normalization-units>`.

   * - Non-hardware-Coherent Requests

     - The total number of requests to the L2 to Not-hardware-Coherent (NC)
       memory allocations, per :ref:`normalization unit <normalization-units>`.
       See the :ref:`memory-type` for more information.

     - Requests per :ref:`normalization unit <normalization-units>`.

   * - Uncached Requests

     - The total number of requests to the L2 that go to Uncached (UC) memory
       allocations. See the :ref:`memory-type` for more information.

     - Requests per :ref:`normalization unit <normalization-units>`.

   * - Coherently Cached Requests

     - The total number of requests to the L2 that go to Coherently Cacheable (CC)
       memory allocations. See the :ref:`memory-type` for more information.

     - Requests per :ref:`normalization unit <normalization-units>`.

   * - Read/Write Coherent Requests

     - The total number of requests to the L2 that go to Read-Write coherent memory
       (RW) allocations. See the :ref:`memory-type` for more information.

     - Requests per :ref:`normalization unit <normalization-units>`.

.. note::

   All requests to the L2 are for a single cache line's worth of data. The size
   of a cache line may vary depending on the accelerator, however on an AMD
   Instinct CDNA2 :ref:`MI2XX <mixxx-note>` accelerator, it is 128B, while on
   an MI100, it is 64B.

.. _l2-fabric:

L2-Fabric transactions
======================

Requests/data that miss in the L2 must be routed to memory in order to
service them. The backing memory for a request may be local to this
accelerator (i.e., in the local high-bandwidth memory), in a remote
accelerator’s memory, or even in the CPU’s memory. Infinity Fabric
is responsible for routing these memory requests/data to the correct
location and returning any fetched data to the L2 cache. The
:ref:`l2-request-flow` describes the flow of these requests through
Infinity Fabric in more detail, as described by ROCm Compute Profiler metrics,
while :ref:`l2-request-metrics` give detailed definitions of
individual metrics.

.. _l2-request-flow:

Request flow
------------

The following is a diagram that illustrates how L2↔Fabric requests are reported
by ROCm Compute Profiler:

.. figure:: ../data/performance-model/l2perf_model.png
   :align: center
   :alt: L2-Fabric transaction flow on AMD Instinct MI-series accelerators
   :width: 800

   L2↔Fabric transaction flow on AMD Instinct MI-series accelerators.


Requests from the L2 Cache are broken down into two major categories, read
requests and write requests (at this granularity, atomic requests are treated
as writes).

From there, these requests can additionally subdivided in a number of ways.
First, these requests may be sent across Infinity Fabric as different
transaction sizes, 32B or 64B on current CDNA accelerators.

.. note::

   On current CDNA accelerators, the 32B read request path is expected to be
   unused and so is disconnected in the flow diagram.

In addition, the read and write requests can be further categorized as:

* Uncached read/write requests, for instance: for access to
  :ref:`fine-grained memory <memory-type>`

* Atomic requests, for instance: for atomic updates to
  :ref:`fine-grained memory <memory-type>`

* HBM read/write requests OR remote read/write requests, for instance: for
  requests to the accelerator’s local HBM OR requests to a remote accelerator’s
  HBM or the CPU’s DRAM

These classifications are not necessarily *exclusive*. For example, a
write request can be classified as an atomic request to the
accelerator’s local HBM, and an uncached write request. The request-flow
diagram marks *exclusive* classifications as a splitting of the flow,
while *non-exclusive* requests do not split the flow line. For example,
a request is either a 32B Write Request OR a 64B Write request, as the
flow splits at this point.

However, continuing along, the same request might be an atomic request and an
uncached write request, as reflected by a non-split flow.

Finally, we note that :ref:`uncached <memory-type>` read requests (e.g., to
:ref:`fine-grained memory <memory-type>`) are handled specially on CDNA
accelerators, as indicated in the request flow diagram. These are
expected to be counted as a 64B Read Request, and *if* they are requests
to uncached memory (denoted by the dashed line), they will also be
counted as *two* uncached read requests (that is, the request is split).


.. _l2-request-metrics:

Metrics
-------

 The following metrics are reported for the L2-Fabric interface:

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - L2-Fabric Read Bandwidth

     - The total number of bytes read by the L2 cache from Infinity Fabric per
       :ref:`normalization unit <normalization-units>`.

     - Bytes per :ref:`normalization unit <normalization-units>`.

   * - HBM Read Traffic

     - The percent of read requests generated by the L2 cache that are routed to
       the accelerator's local high-bandwidth memory (HBM). This breakdown does
       not consider the *size* of the request (meaning that 32B and 64B requests
       are both counted as a single request), so this metric only *approximates*
       the percent of the L2-Fabric Read bandwidth directed to the local HBM.

     - Percent

   * - Remote Read Traffic

     - The percent of read requests generated by the L2 cache that are routed to
       any memory location other than the accelerator's local high-bandwidth
       memory (HBM) -- for example, the CPU's DRAM or a remote accelerator's
       HBM. This breakdown does not consider the *size* of the request (meaning
       that 32B and 64B requests are both counted as a single request), so this
       metric only *approximates* the percent of the L2-Fabric Read bandwidth
       directed to a remote location.

     - Percent

   * - Uncached Read Traffic

     - The percent of read requests generated by the L2 cache that are reading
       from an :ref:`uncached memory allocation <memory-type>`. Note, as
       described in the :ref:`request flow <l2-request-flow>` section, a single
       64B read request is typically counted as two uncached read requests. So,
       it is possible for the Uncached Read Traffic to reach up to 200% of the
       total number of read requests. This breakdown does not consider the
       *size* of the request (i.e., 32B and 64B requests are both counted as a
       single request), so this metric only *approximates* the percent of the
       L2-Fabric read bandwidth directed to an uncached memory location.

     - Percent

   * - L2-Fabric Write and Atomic Bandwidth

     - The total number of bytes written by the L2 over Infinity Fabric by write
       and atomic operations per
       :ref:`normalization unit <normalization-units>`. Note that on current
       CDNA accelerators, such as the :ref:`MI2XX <mixxx-note>`, requests are
       only considered *atomic* by Infinity Fabric if they are targeted at
       non-write-cacheable memory, for example,
       :ref:`fine-grained memory <memory-type>` allocations or
       :ref:`uncached memory <memory-type>` allocations on the
       MI2XX.

     - Bytes per :ref:`normalization unit <normalization-units>`.

   * - HBM Write and Atomic Traffic

     - The percent of write and atomic requests generated by the L2 cache that
       are routed to the accelerator's local high-bandwidth memory (HBM). This
       breakdown does not consider the *size* of the request (meaning that 32B
       and 64B requests are both counted as a single request), so this metric
       only *approximates* the percent of the L2-Fabric Write and Atomic
       bandwidth directed to the local HBM. Note that on current CDNA
       accelerators, such as the :ref:`MI2XX <mixxx-note>`, requests are only
       considered *atomic* by Infinity Fabric if they are targeted at
       :ref:`fine-grained memory <memory-type>` allocations or
       :ref:`uncached memory <memory-type>` allocations.

     - Percent

   * - Remote Write and Atomic Traffic

     - The percent of read requests generated by the L2 cache that are routed to
       any memory location other than the accelerator's local high-bandwidth
       memory (HBM) -- for example, the CPU's DRAM or a remote accelerator's
       HBM. This breakdown does not consider the *size* of the request (meaning
       that 32B and 64B requests are both counted as a single request), so this
       metric only *approximates* the percent of the L2-Fabric Read bandwidth
       directed to a remote location. Note that on current CDNA
       accelerators, such as the :ref:`MI2XX <mixxx-note>`, requests are only
       considered *atomic* by Infinity Fabric if they are targeted at
       :ref:`fine-grained memory <memory-type>` allocations or
       :ref:`uncached memory <memory-type>` allocations.

     - Percent

   * - Atomic Traffic

     - The percent of write requests generated by the L2 cache that are atomic
       requests to *any* memory location. This breakdown does not consider the
       *size* of the request (meaning that 32B and 64B requests are both counted
       as a single request), so this metric only *approximates* the percent of
       the L2-Fabric Read bandwidth directed to a remote location. Note that on
       current CDNA accelerators, such as the :ref:`MI2XX <mixxx-note>`,
       requests are only considered *atomic* by Infinity Fabric if they are
       targeted at :ref:`fine-grained memory <memory-type>` allocations or
       :ref:`uncached memory <memory-type>` allocations.

     - Percent

   * - Uncached Write and Atomic Traffic

     - The percent of write and atomic requests generated by the L2 cache that
       are targeting :ref:`uncached memory allocations <memory-type>`. This
       breakdown does not consider the *size* of the request (meaning that 32B
       and 64B requests are both counted as a single request), so this metric
       only *approximates* the percent of the L2-Fabric read bandwidth directed
       to uncached memory allocations.

     - Percent

   * - Read Latency

     - The time-averaged number of cycles read requests spent in Infinity Fabric
       before data was returned to the L2.

     - Cycles

   * - Write Latency

     - The time-averaged number of cycles write requests spent in Infinity
       Fabric before a completion acknowledgement was returned to the L2.

     - Cycles

   * - Atomic Latency

     - The time-averaged number of cycles atomic requests spent in Infinity
       Fabric before a completion acknowledgement (atomic without return value)
       or data (atomic with return value) was returned to the L2.

     - Cycles

   * - Read Stall

     - The ratio of the total number of cycles the L2-Fabric interface was
       stalled on a read request to any destination (local HBM, remote PCIe®
       connected accelerator or CPU, or remote Infinity Fabric connected
       accelerator [#inf]_ or CPU) over the
       :ref:`total active L2 cycles <total-active-l2-cycles>`.

     - Percent

   * - Write Stall

     - The ratio of the total number of cycles the L2-Fabric interface was
       stalled on a write or atomic request to any destination (local HBM,
       remote accelerator or CPU, PCIe connected accelerator or CPU, or remote
       Infinity Fabric connected accelerator [#inf]_ or CPU) over the
       :ref:`total active L2 cycles <total-active-l2-cycles>`.

     - Percent

.. _l2-detailed-metrics:

Detailed transaction metrics
----------------------------

The following metrics are available in the detailed L2-Fabric
transaction breakdown table:

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - 32B Read Requests

     - The total number of L2 requests to Infinity Fabric to read 32B of data
       from any memory location, per
       :ref:`normalization unit <normalization-units>`. See
       :ref:`l2-request-flow` for more detail. Typically unused on CDNA
       accelerators.

     - Requests per :ref:`normalization unit <normalization-units>`.

   * - Uncached Read Requests

     - The total number of L2 requests to Infinity Fabric to read
       :ref:`uncached data <memory-type>` from any memory location, per
       :ref:`normalization unit <normalization-units>`. 64B requests for
       uncached data are counted as two 32B uncached data requests. See
       :ref:`l2-request-flow` for more detail.

     - Requests per :ref:`normalization unit <normalization-units>`.

   * - 64B Read Requests

     - The total number of L2 requests to Infinity Fabric to read 64B of data
       from any memory location, per
       :ref:`normalization unit <normalization-units>`. See
       :ref:`l2-request-flow` for more detail.

     - Requests per :ref:`normalization unit <normalization-units>`.

   * - HBM Read Requests

     - The total number of L2 requests to Infinity Fabric to read 32B or 64B of
       data from the accelerator's local HBM, per
       :ref:`normalization unit <normalization-units>`. See
       :ref:`l2-request-flow` for more detail.

     - Requests per :ref:`normalization unit <normalization-units>`.

   * - Remote Read Requests

     - The total number of L2 requests to Infinity Fabric to read 32B or 64B of
       data from any source other than the accelerator's local HBM, per
       :ref:`normalization unit <normalization-units>`. See
       :ref:`l2-request-flow` for more detail.

     - Requests per :ref:`normalization unit <normalization-units>`.

   * - 32B Write and Atomic Requests

     - The total number of L2 requests to Infinity Fabric to write or atomically
       update 32B of data to any memory location, per
       :ref:`normalization unit <normalization-units>`. See
       :ref:`l2-request-flow` for more detail.

     - Requests per :ref:`normalization unit <normalization-units>`.

   * - Uncached Write and Atomic Requests

     - The total number of L2 requests to Infinity Fabric to write or atomically
       update 32B or 64B of :ref:`uncached data <memory-type>`, per
       :ref:`normalization unit <normalization-units>`. See
       :ref:`l2-request-flow` for more detail.

     - Requests per :ref:`normalization unit <normalization-units>`.

   * - 64B Write and Atomic Requests

     - The total number of L2 requests to Infinity Fabric to write or atomically
       update 64B of data in any memory location, per
       :ref:`normalization unit <normalization-units>`. See
       :ref:`l2-request-flow` for more detail.

     - Requests per :ref:`normalization unit <normalization-units>`.

   * - HBM Write and Atomic Requests

     - The total number of L2 requests to Infinity Fabric to write or atomically
       update 32B or 64B of data in the accelerator's local HBM, per
       :ref:`normalization unit <normalization-units>`. See
       :ref:`l2-request-flow` for more detail.

     - Requests per :ref:`normalization unit <normalization-units>`.

   * - Remote Write and Atomic Requests

     - The total number of L2 requests to Infinity Fabric to write or atomically
       update 32B or 64B of data in any memory location other than the
       accelerator's local HBM, per
       :ref:`normalization unit <normalization-units>`. See
       :ref:`l2-request-flow` for more detail.

     - Requests per :ref:`normalization unit <normalization-units>`.

   * - Atomic Requests

     - The total number of L2 requests to Infinity Fabric to atomically update
       32B or 64B of data in any memory location, per
       :ref:`normalization unit <normalization-units>`. See
       :ref:`l2-request-flow` for more detail. Note that on current CDNA
       accelerators, such as the :ref:`MI2XX <mixxx-note>`, requests are only
       considered *atomic* by Infinity Fabric if they are targeted at
       non-write-cacheable memory, such as
       :ref:`fine-grained memory <memory-type>` allocations or
       :ref:`uncached memory <memory-type>` allocations on the MI2XX.

     - Requests per :ref:`normalization unit <normalization-units>`.

.. _l2-fabric-stalls:

L2-Fabric interface stalls
==========================

When the interface between the L2 cache and Infinity Fabric becomes backed up by
requests, it may stall, preventing the L2 from issuing additional requests to
Infinity Fabric until prior requests complete. This section gives a breakdown of
what types of requests in a kernel caused a stall (like read versus write), and
to which locations -- for instance, to the accelerator’s local memory, or to
remote accelerators or CPUs.

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - Read - PCIe Stall

     - The number of cycles the L2-Fabric interface was stalled on read requests
       to remote PCIe connected accelerators [#inf]_ or CPUs as a percent of the
       :ref:`total active L2 cycles <total-active-l2-cycles>`.

     - Percent

   * - Read - Infinity Fabric Stall

     - The number of cycles the L2-Fabric interface was stalled on read requests
       to remote Infinity Fabric connected accelerators [#inf]_ or CPUs as a
       percent of the :ref:`total active L2 cycles <total-active-l2-cycles>`.

     - Percent

   * - Read - HBM Stall

     - The number of cycles the L2-Fabric interface was stalled on read requests
       to the accelerator's local HBM as a percent of the
       :ref:`total active L2 cycles <total-active-l2-cycles>`.

     - Percent

   * - Write - PCIe Stall

     - The number of cycles the L2-Fabric interface was stalled on write or
       atomic requests to remote PCIe connected accelerators [#inf]_ or CPUs as
       a percent of the :ref:`total active L2 cycles <total-active-l2-cycles>`.

     - Percent

   * - Write - Infinity Fabric Stall

     - The number of cycles the L2-Fabric interface was stalled on write or
       atomic requests to remote Infinity Fabric connected accelerators [#inf]_
       or CPUs as a percent of the
       :ref:`total active L2 cycles <total-active-l2-cycles>`.

     - Percent

   * - Write - HBM Stall

     - The number of cycles the L2-Fabric interface was stalled on write or
       atomic requests to accelerator's local HBM as a percent of the
       :ref:`total active L2 cycles <total-active-l2-cycles>`.

     - Percent

   * - Write - Credit Starvation

     - The number of cycles the L2-Fabric interface was stalled on write or
       atomic requests to any memory location because too many write/atomic
       requests were currently in flight, as a percent of the
       :ref:`total active L2 cycles <total-active-l2-cycles>`.

     - Percent

.. warning::

   On current CDNA accelerators and GCN GPUs, these L2↔Fabric stalls can be undercounted in some circumstances.

.. rubric:: Footnotes

.. [#inf] In addition to being used for on-accelerator data-traffic, AMD
   `Infinity Fabric <https://www.amd.com/en/technologies/infinity-architecture>`_
   technology can be used to connect multiple accelerators to achieve advanced
   peer-to-peer connectivity and enhanced bandwidths over traditional PCIe
   connections. Some AMD Instinct MI-series accelerators like the MI250X feature coherent CPU-to-accelerator connections are built using AMD Infinity Fabric. For more information, see the `AMD CDNA2 white paper <https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf>`_.

.. rubric:: Disclaimer

PCIe® is a registered trademark of PCI-SIG Corporation.
