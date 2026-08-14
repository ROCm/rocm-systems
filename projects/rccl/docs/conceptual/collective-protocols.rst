.. meta::
   :description: Learn how RCCL's Simple, LL, and LL128 protocols control data packetization, synchronization, and bandwidth efficiency for collective operations on AMD GPUs.
   :keywords: RCCL, protocol, Simple protocol, LL protocol, LL128, low latency, NCCL_PROTO, RCCL_OVERRIDE_PROTO, bandwidth efficiency, memory fence, threadfence, gfx942, MI300X, gfx950, collective communication, ROCm

.. _collective-protocols:

*****************************
Collective protocols in RCCL
*****************************

When RCCL moves data between GPUs, it uses a **protocol** to control how that
data is packetized and how the receiver knows when data has arrived. The choice
of protocol is independent of the algorithm (Ring, Tree, Hierarchical, and so
on): any algorithm can run under any supported protocol, and RCCL selects the
combination that minimizes estimated execution time for the given message size
and hardware.

There are three protocols: **Simple**, **LL** (Low Latency), and **LL128**.
They differ in how they embed the completion signal alongside the data, which
in turn determines their latency, bandwidth efficiency, and hardware
requirements.

Understanding what each protocol does and why it exists is useful for
diagnosing unexpected performance, interpreting ``NCCL_DEBUG`` output, and
knowing when to override the default selection.

The core problem: ordering the flag after the data
==================================================

All three protocols solve the same fundamental problem. Consider a GPU sending
a payload to another GPU and then writing a flag to signal that the payload is
complete. The receiving GPU polls the flag and, once it sees it, reads the
payload. For correctness, the flag must never be visible to the receiver before
the entire payload has arrived.

This guarantee is harder to achieve than it appears, for three distinct
reasons.

**Compile-time reordering.** Languages with weakly ordered memory models —
including C++11, HIP, and CUDA — permit the compiler to reorder independent
stores. If the flag write is not explicitly ordered after the payload writes,
the compiler might emit the flag store first, allowing the receiver to observe
the flag before the data it protects.

The fix is an explicit memory fence instruction between the last payload write
and the flag write. In HIP, this is ``__threadfence()``. RCCL inserts this
fence in the Simple protocol kernel to enforce store ordering.

**Runtime reordering.** Even with a correct fence in the compiled code,
hardware can still reorder memory operations in flight. Stores might be buffered
in L2 caches, PCIe switches, or the data fabric and arrive at the destination
in a different order than they were issued. On some GPU-to-GPU paths, the
payload stores might drain through a different route than the flag store,
allowing the flag to arrive first.

The fix depends on the transport involved. For intra-node transfers over xGMI,
cache coherence is maintained by the fabric. For PCIe and network paths, an
explicit cache flush or non-temporal store might be required before the flag
write.

**Multi-device and multi-node coherency.** When data crosses a network
interface card (NIC), the ordering problem becomes multi-system. The NIC must
read the payload from GPU memory, transmit it, and deliver it to the remote
GPU before the flag is posted. This requires coordination across the GPU, the
PCIe switch, the NIC, the network fabric, and the remote memory system —
intermediate devices that have no inherent awareness of the flag-payload
relationship.

The Simple protocol handles all three cases with fences and flushes, at the
cost of requiring the receiver to wait for the entire payload before any data
is usable. The LL and LL128 protocols take a different approach: they embed
the flag inside the same atomic write as the data, eliminating the ordering
problem entirely.

Simple protocol
===============

The Simple protocol sends a large payload as a contiguous chunk followed by a
single completion flag. The receiver polls the flag; when it is set, the
receiver knows the payload is ready.

**Data efficiency.** Simple uses less than 100% of the wire bandwidth for data
because the flag adds a small fixed overhead per chunk. In practice, the
efficiency approaches 100% for large messages because the flag overhead is
negligible relative to the payload size.

**Why Simple wins for large messages.** For a 1 GB AllReduce, the bandwidth
term in the cost model dominates the latency term. Simple maximizes sustained
bandwidth because it sends data in large contiguous chunks with minimal
per-element overhead, allowing the memory system and network fabric to operate
at full throughput. For large messages, the fence overhead is a one-time cost
per chunk, not per element.

**Why Simple loses for small messages.** For small payloads, the fence
introduces latency that is proportionally large relative to the transfer time.
The receiver also cannot begin consuming data until the entire payload has
arrived, so there is no opportunity for pipelining within a single chunk.

**When RCCL selects Simple.** Simple is selected for large messages,
typically above several hundred kilobytes to a few megabytes depending on
hardware. It is also the only protocol available when the hardware does not
support the atomic transaction sizes required by LL or LL128.

LL (Low Latency) protocol
=========================

The LL protocol eliminates the fence-and-flag ordering problem by packing the
data and its completion flag into a single indivisible atomic write. Because
the data and flag arrive together in one transaction, the receiver can detect
when each individual data unit is ready without waiting for the entire payload.

**Packet structure.** Each LL packet is the size of the largest indivisible
atomic transaction the hardware supports for the relevant path. On AMD Instinct
MI300X (gfx942), this is 64 bytes, structured as 60 bytes of data plus a 4-byte
flag:

.. code-block:: text

   [ data : 60 bytes ][ flag : 4 bytes ]

The receiver polls the 4-byte flag of each 64-byte packet independently. As
soon as a packet's flag is set, the receiver knows those 60 bytes of data are
valid and can immediately process or forward them — without waiting for
subsequent packets to arrive.

**Data efficiency.** LL transfers 4 bytes of flag for every 60 bytes of data,
giving a wire efficiency of 60/64 = 93.75%... but the usable data fraction
relative to total bytes sent is only 4 bytes of flag per 8-byte "slot" in the
original NCCL LL formulation (4B data + 4B flag), giving 50% efficiency. The
exact efficiency depends on the data type and alignment.

**Why LL wins for small messages.** For a small AllReduce, the latency term
dominates the cost model. LL allows the receiver to begin reducing the first
packet while the sender is still transmitting subsequent packets, effectively
pipelining data movement and computation. The total time-to-first-result is
dramatically lower than Simple, which requires all data to arrive before
any processing begins.

**Why LL loses for large messages.** The flag overhead — 4 bytes per 60 bytes
of payload — reduces effective bandwidth by approximately 6.25% compared to
Simple. For large transfers this bandwidth tax outweighs the latency advantage,
and Simple becomes the better choice.

**Hardware dependency.** LL requires that the GPU support atomic writes of the
required packet size to the destination memory. On AMD hardware, the guaranteed
atomic transaction size varies by architecture and memory path (xGMI vs PCIe
vs network). RCCL's tuning model encodes these constraints per GPU generation.

LL128 protocol
==============

LL128 is a hybrid protocol that extends the LL approach to 128-byte aligned
atomic writes, achieving a much higher data efficiency while retaining the
flag-per-packet synchronization model.

**Packet structure.** Each LL128 packet is 128 bytes: 120 bytes of data plus
an 8-byte flag:

.. code-block:: text

   [ data : 120 bytes ][ flag : 8 bytes ]

The 120/128 ratio gives 93.75% wire efficiency — nearly the same as Simple for
large messages — while preserving the LL property that each packet can be
consumed as soon as it arrives, without a memory fence.

**Data efficiency.** LL128 achieves approximately 93.75% bandwidth efficiency,
versus 50% for LL and approaching 100% for Simple on large messages. This
makes LL128 effective across a wider message size range than LL, closing the
gap with Simple for medium-sized messages while retaining low per-packet
latency.

**The three-zone performance model.** LL128 fills the gap between LL and
Simple:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Message size
     - Protocol
     - Efficiency
     - Why it wins
   * - Small (< ~8 KB)
     - LL
     - ~50%
     - Lowest per-packet latency; fence overhead dominates at this scale
   * - Medium (~8 KB – ~512 KB)
     - LL128
     - ~93.75%
     - Near-Simple bandwidth with flag-based synchronization; no fence needed
   * - Large (> ~512 KB)
     - Simple
     - < 100%
     - Maximum sustained bandwidth; fence cost is negligible at scale

These thresholds are approximate. The exact crossover points are determined by
the hardware-specific tuning model for each GPU architecture and updated in
each RCCL release.

**Hardware requirements.** LL128 requires that the GPU and memory path support
128-byte aligned atomic stores. This is a stricter requirement than LL. On AMD
hardware, LL128 availability is architecture-specific:

- **gfx942 (MI300X):** enabled for configurations with 4 NICs, using a unified
  tuning table.
- **gfx950 (MI350):** enabled by default.
- **gfx90a (MI200) and earlier:** not enabled by default; LL128 support
  varies by configuration.

If 128-byte atomics are not supported for a given path, RCCL silently falls
back to LL or Simple rather than producing an error.

Protocol summary
================

The following table summarizes the key properties of all three protocols:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Protocol
     - Packet layout
     - Wire efficiency
     - Fence required
     - Best for
   * - Simple
     - Full payload + flag
     - < 100%
     - Yes (``__threadfence()``)
     - Large messages
   * - LL
     - 4B data + 4B flag per slot
     - ~50%
     - No
     - Small messages
   * - LL128
     - 120B data + 8B flag per packet
     - ~93.75%
     - No
     - Medium messages

How RCCL selects a protocol
===========================

Protocol selection is performed jointly with algorithm selection at each
collective launch by the function ``getAlgoInfo()`` in
``src/graph/tuning.cc``. The cost model evaluates every valid
algorithm-and-protocol combination and picks the pair with the lowest estimated
execution time:

.. code-block:: text

   time = (latency + nsteps × size ÷ bandwidth) × correctionFactor

The ``latency`` and ``bandwidth`` constants in this formula come from the
hardware-specific tuning model selected for the detected GPU architecture. Each
architecture has a ``hwLat`` table (per-hop latency per transport type) and a
``bwRatio`` table (fraction of raw link bandwidth each algorithm-protocol
combination achieves). The protocol grain size and dynamic shared memory
requirements are computed in ``src/enqueue.cc`` based on the target
architecture.

Not all algorithm-protocol combinations are valid. The Simple protocol is
available with all algorithms. LL and LL128 are available with Ring and Tree
but are not supported with CollNet or PAT (Port-Aggregated Topology), because
those algorithms offload work to network infrastructure that cannot participate
in the flag-based synchronization model.

Algorithm-protocol availability
===============================

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 20

   * - Algorithm
     - Simple
     - LL
     - LL128
   * - Ring
     - Yes
     - Yes
     - Yes (hardware-dependent)
   * - Tree
     - Yes
     - Yes
     - Yes (hardware-dependent)
   * - Hierarchical
     - Yes
     - Yes
     - Yes (hardware-dependent)
   * - CollNet (Direct / Chain)
     - Yes
     - No
     - No
   * - PAT
     - Yes
     - No
     - No
   * - DDA
     - Yes
     - No
     - No

Override protocol selection
===========================

Protocol selection is automatic and uses the tuning model's best estimate.
You should not need to override it in production. The following variables are
provided for diagnosis and performance investigation.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Effect
   * - ``NCCL_PROTO=LL,LL128,Simple``
     - Restricts the set of protocols the tuning model can choose from.
       Accepts a comma-separated list. The model re-runs over the enabled
       set and still picks the best combination. This is a debug variable;
       forcing a single protocol for all message sizes will degrade
       performance for sizes where it is suboptimal.
   * - ``RCCL_OVERRIDE_PROTO=Simple``
     - Forces a specific protocol unconditionally, bypassing the tuning model
       entirely. Use when you need to measure the performance of a specific
       protocol in isolation.
   * - ``NCCL_DEBUG=INFO`` + ``NCCL_DEBUG_SUBSYS=TUNING``
     - Prints the algorithm and protocol selected for each collective at
       communicator initialization, including the estimated latency and
       bandwidth for each combination evaluated.

.. note::

   ``NCCL_PROTO`` is a global override that applies to all message sizes and
   all collectives. Forcing ``NCCL_PROTO=LL`` on large messages will cut
   effective bandwidth nearly in half. Use it only to diagnose specific issues
   and always remove it before production runs.

Related topics
==============

- :doc:`Collective algorithms in RCCL <./collective-algorithms>` — how
  algorithm selection interacts with protocol selection
- :doc:`Hardware-specific optimizations <./hardware-specific-optimizations>` —
  per-architecture protocol availability, LL packet sizes, and the tuning model
  that drives selection
- :doc:`Collective operations in RCCL <./collective-operations>` — what each
  collective does and the communicator model
- :doc:`Run RCCL-Tests <../how-to/running-rccl-tests>` — how to measure
  protocol performance with ``NCCL_PROTO`` overrides
- :ref:`Environment variables <env-variables>` — complete reference for all
  RCCL runtime knobs
