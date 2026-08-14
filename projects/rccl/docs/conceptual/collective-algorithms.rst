.. meta::
   :description: Learn how RCCL selects between ring, tree, hierarchical, CollNet, PAT, and DDA algorithms for collective operations on AMD Instinct GPUs.
   :keywords: RCCL, ring algorithm, tree algorithm, hierarchical algorithm, CollNet, PAT, DDA, collective communication, AllReduce, algorithm selection, ROCm, tuning, gfx942, MI300X, cost model, NCCL_ALGO, xGMI

.. _collective-algorithms:

******************************
Collective algorithms in RCCL
******************************

RCCL does not use a single algorithm for all collective operations. For each
collective call, RCCL evaluates several candidate algorithms and selects the
one predicted to run fastest on the current hardware for the given message size
and node count. Understanding what each algorithm does, when it wins, and what
trade-offs it makes is the foundation for diagnosing unexpected performance and
for tuning workloads that push the edges of the default heuristics.

How RCCL chooses an algorithm
=============================

At communicator initialization, RCCL builds latency and bandwidth matrices from
its hardware-specific tuning models. For every collective launch, it evaluates
all valid algorithm-and-protocol combinations using the cost formula:

.. code-block:: text

   time = (latency + nsteps × size ÷ bandwidth) × correctionFactor

The combination with the lowest estimated time is selected. The two dominant
variables in this formula are:

- **latency** — the per-step fixed cost, which dominates at small message sizes.
- **nsteps × size ÷ bandwidth** — the data-movement cost, which dominates at
  large message sizes.

Algorithms differ in how many steps they take and how much data they move per
step. An algorithm with fewer steps wins at small sizes even if it moves more
total data; an algorithm that is bandwidth-optimal wins at large sizes even if
it takes more steps.

Default algorithm assignments
=============================

RCCL's default algorithm selection (when ``NCCL_ALGO=Ring/Tree``) maps
collectives to algorithms as follows:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Collective
     - Algorithm
     - Scale-up (intra-node)
     - Scale-out (multi-node)
   * - AllReduce
     - Ring, Tree
     - Yes
     - Yes
   * - AllGather
     - Ring
     - Yes
     - Yes
   * - ReduceScatter
     - Ring
     - Yes
     - Yes
   * - AllToAll
     - Direct
     - Yes
     - Yes
   * - Broadcast
     - Ring
     - Yes
     - Yes
   * - Reduce
     - Ring
     - Yes
     - Yes

RCCL also implements advanced algorithms — DDA (Direct Data Access),
Hierarchical, and PAT (Port-Aggregated Topology) — for specific hardware
configurations and scenarios. These are described in
`Advanced algorithms`_ below.

Ring algorithm
==============

The ring algorithm arranges all participating GPUs in a logical ring. Each GPU
simultaneously sends data to its successor and receives data from its
predecessor. After *N−1* steps (where *N* is the number of ranks), every GPU
has seen data from every other GPU.

For AllReduce, the ring operates in two phases:

1. **ReduceScatter.** In *N−1* steps, each GPU accumulates a partial sum for
   one slice of the tensor. After this phase, each GPU holds the fully reduced
   result for exactly 1/*N* of the output.
2. **AllGather.** In *N−1* more steps, each GPU broadcasts its fully reduced
   slice around the ring. After this phase, every GPU holds the complete result.

Total steps: 2(N−1). Total data transferred per GPU: 2(N−1)/N × input size.

**When ring wins.** Ring is bandwidth-optimal for large messages: as *N* grows,
each GPU sends and receives close to 2× the input size regardless of how many
GPUs are in the ring, so the per-step data movement per link saturates the
interconnect. Ring is the default for AllGather, ReduceScatter, Broadcast,
Reduce, and for large-message AllReduce.

**When ring loses.** For small messages, the *N−1* sequential steps mean that
total latency grows linearly with *N*. On a 256-rank job, a tiny AllReduce
requires 510 steps even if the data fits in a single packet. Tree algorithms
close this gap.

**Bus bandwidth correction factor.** Because ring sends more data than the
input size, the bus bandwidth (busBw) reported by RCCL-Tests is related to
algorithm bandwidth (algBw) by:

.. code-block:: text

   busBw = algBw × 2(N-1)/N

For 8 ranks this factor is 1.75; for 256 ranks it approaches 2.0. This is why
busBw can be nearly twice algBw on large clusters even though the application
only sees algBw.

**On switch-based topologies** (such as NVSwitch in NVIDIA systems or a
top-of-rack switch), any permutation of GPUs forms a valid ring that uses the
full switch bandwidth, because all traffic passes through the same switch fabric.
On a **fully connected point-to-point topology** (such as the AMD Instinct™
MI300X xGMI mesh, where every GPU has a direct link to every other GPU),
finding rings that do not collide on any link is a combinatorial problem. RCCL
solves this at initialization using pre-computed ring orderings stored in the
Rome model records. For an 8-GPU MI300X node, RCCL constructs 7 independent
non-colliding rings — one per xGMI link — and runs them in parallel as separate
channels, achieving approximately 7 × 46 GB/s ≈ 322 GB/s per-GPU bandwidth.

Tree algorithm
==============

The tree algorithm reduces data along a binary reduction tree in ``log₂(N)``
steps, rather than the *N−1* steps that ring requires. This dramatically reduces
latency for small messages on large GPU counts.

The cost is that tree sends *more* data per rank than ring: each rank sends and
receives approximately 2× the input size (versus ring's 2(N−1)/N × input size).
For small *N*, ring is more efficient on bandwidth; for large *N*, the difference
becomes negligible.

The step-count and data-multiplier trade-off, measured empirically:

.. list-table::
   :header-rows: 1
   :widths: 16 21 21 21 21

   * - Total ranks
     - Ring data multiplier
     - Tree data multiplier
     - Ring steps
     - Tree steps
   * - 8
     - 1.75×
     - 2.00×
     - 14
     - 6
   * - 16
     - 1.88×
     - 2.00×
     - 30
     - 8
   * - 32
     - 1.94×
     - 2.00×
     - 62
     - 10
   * - 64
     - 1.97×
     - 2.00×
     - 126
     - 12
   * - 128
     - 1.98×
     - 2.00×
     - 254
     - 14
   * - 256
     - 1.99×
     - 2.00×
     - 510
     - 16

At 8 ranks, tree uses 43% fewer steps than ring; at 256 ranks, tree uses 97%
fewer steps. The data-multiplier difference is only 14% at 8 ranks and
essentially zero at 256 ranks.

**When tree wins.** Tree is selected for small messages where the latency term
dominates, and for AllReduce when the message is latency-bound. The exact
crossover point is determined by the tuning model for the target hardware.

**Rail-optimized tree orderings.** On multi-node MI300X deployments with
rail-optimized networking, RCCL assigns tree nodes so that each subtree
alternates between NIC rails — the left subtree uses one set of NIC–switch
assignments, the right subtree uses another. This prevents tree-reduction
traffic from concentrating on a single switch rail, which would create a
bottleneck on the first-tier switches.

Hierarchical algorithm
======================

The hierarchical algorithm decomposes a flat communicator into two levels:

- **Intra-node (scale-up).** All GPUs within the same node participate in a
  fast intra-node collective using xGMI or direct GPU-to-GPU memory access.
- **Inter-node (scale-out).** One GPU per node (or one per NIC) participates in
  the inter-node collective using the network fabric.

This two-level decomposition lets the intra-node phase use the full xGMI
bandwidth (hundreds of GB/s) while the slower inter-node phase only involves
a subset of GPUs, reducing the amount of traffic on the network.

For AllGather, the hierarchical decomposition works as follows:

1. Each **inter-node communicator** groups GPUs with the same local rank across
   all nodes (for example, GPU 0 from every node). This communicator performs an
   inter-node AllGather using PAT (see `PAT algorithm`_ below) or the ring
   algorithm.
2. Each **intra-node communicator** groups all GPUs within the same node. This
   communicator performs an intra-node AllGather using one-shot, two-shot,
   direct, or ring.

On 16 nodes with 8× MI350 GPUs per node, hierarchical AllGather delivers up to
3.6× speedup over the default ring algorithm for small and medium messages.
Hierarchical ReduceScatter is also supported.

**When hierarchical is used.** Hierarchical AllGather is enabled by default for
multi-node runs with 8 or more nodes (ROCm 7.13 and later). The environment
variable ``RCCL_HIERARCHICAL_ALLGATHER`` controls this. Set it to ``0`` to fall
back to the flat ring algorithm, or to ``1`` to force hierarchical regardless of
node count.

CollNet algorithm
=================

CollNet (Collective Network) offloads the reduction to network infrastructure
that supports in-network computing — for example, a SHARP-enabled InfiniBand
switch. Instead of sending data to other GPUs and reducing on the GPU, the
switch fabric performs the reduction as data passes through it.

Two variants exist:

- **CollNetDirect** — GPUs post send and receive operations directly to and from
  the in-network compute engine, with no intermediate CPU staging.
- **CollNetChain** — data flows through a chain of switches that each apply a
  partial reduction.

CollNet reduces the round-trip latency for AllReduce because the reduction
happens inside the network rather than on a GPU. It is only available when the
network fabric explicitly supports in-network compute and RCCL can detect it at
initialization.

PAT algorithm
=============

PAT (Port-Aggregated Topology) aggregates bandwidth across multiple NIC ports to
serve traffic patterns that would saturate a single port. Rather than routing all
traffic for one collective through one NIC, PAT distributes the load across all
available NICs in a load-balanced way.

PAT is currently enabled for single-GPU-per-node configurations and is used as
the inter-node phase of hierarchical AllGather and ReduceScatter.

Advanced algorithms
===================

RCCL implements additional algorithms that go beyond the default Ring/Tree
selection. These are hardware-specific and require explicit enablement.

.. list-table::
   :header-rows: 1
   :widths: 20 25 20 15 20

   * - Collective
     - Algorithm
     - Environment variable
     - Scale-up
     - Scale-out
   * - AllReduce
     - DDA (Direct Data Access)
     - ``RCCL_DDA_ENABLE=1``
     - Yes
     - No
   * - AllGather
     - DDA
     - ``RCCL_DDA_ENABLE=1``
     - Yes
     - No
   * - AllGather
     - Direct, Hierarchical
     - ``RCCL_HIERARCHICAL_ALLGATHER``
     - Yes
     - Yes
   * - ReduceScatter
     - DDA
     - ``RCCL_DDA_ENABLE=1``
     - Yes
     - No
   * - ReduceScatter
     - Direct
     - (automatic)
     - Yes
     - Yes

DDA (Direct Data Access) algorithm
-----------------------------------

DDA is a symmetric-memory algorithm for single-node AllReduce, AllGather, and
ReduceScatter. It exploits the xGMI fully connected topology: rather than passing
data around a ring, every GPU reads directly from every other GPU's memory using
low-latency load instructions into shared symmetric memory windows registered
across all GPUs in the communicator.

DDA eliminates the sequential ring steps for intra-node communication, reducing
latency for small and medium messages significantly. Because it requires all GPUs
to have direct peer access to each other's memory, DDA is currently limited to
single-node configurations.

Enable DDA with ``RCCL_DDA_ENABLE=1``.

Direct algorithm
----------------

The Direct algorithm is the default for AllToAll and is also supported for
ReduceScatter in multi-node configurations. Each GPU posts send and receive
operations directly to every destination GPU without an intermediate reduction
tree or ring. This is efficient when every GPU has data for every other GPU
and the network can handle many simultaneous flows.

One-shot and two-shot algorithms
--------------------------------

For single-node small-message AllReduce, AllGather, ReduceScatter, and AllToAll,
RCCL supports one-shot and two-shot variants that reduce the number of kernel
launches and synchronization barriers:

- **One-shot** — performs the entire collective in a single kernel pass over
  symmetric memory, with each GPU reading all peer data in one sweep.
- **Two-shot** — splits the operation into a reduce phase and an all-gather
  phase with a single barrier between them, allowing pipeline overlap.

These variants are available for both registered buffers (user buffer
registration, requiring application changes) and unregistered buffers (standard
hipMalloc allocations, requiring no application changes). On 8× MI350, one-shot
and two-shot AllReduce deliver 2–4× speedup over the default ring algorithm for
messages below ~8 MB.

Protocol interaction with algorithm selection
=============================================

Algorithm selection and protocol selection are independent decisions that RCCL
makes jointly. The same ring pattern can execute under the Simple, LL, or LL128
protocol; the tuning model evaluates all valid combinations and picks the pair
with the lowest estimated time.

In general:

- **Small messages:** Tree or one-shot + LL (lowest latency).
- **Medium messages:** Tree or Ring + LL128 (low latency with near-peak
  bandwidth efficiency).
- **Large messages:** Ring + Simple (maximum sustained bandwidth).

For a detailed explanation of how the protocols work at the wire level, see
:doc:`Hardware-specific optimizations <./hardware-specific-optimizations>`.

Override algorithm selection
============================

Use the following environment variables to override RCCL's automatic selection.
These are primarily diagnostic tools; forcing a sub-optimal algorithm for all
message sizes will degrade performance.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Effect
   * - ``NCCL_ALGO=Ring``
     - Restrict selection to Ring only. RCCL re-runs the cost model over
       the enabled set, so the tuning model still picks the best protocol
       and channel count.
   * - ``NCCL_ALGO=Tree``
     - Restrict selection to Tree only.
   * - ``RCCL_OVERRIDE_ALGO=Ring``
     - Force Ring unconditionally, bypassing the tuning model entirely.
       Use this when you want to compare performance against the tuning
       model's choice.
   * - ``RCCL_OVERRIDE_ALGO=Tree``
     - Force Tree unconditionally.
   * - ``RCCL_DDA_ENABLE=1``
     - Enable DDA for supported single-node collectives.
   * - ``RCCL_HIERARCHICAL_ALLGATHER=0``
     - Disable hierarchical AllGather and fall back to flat ring.
   * - ``NCCL_DEBUG=INFO`` + ``NCCL_DEBUG_SUBSYS=TUNING``
     - Print the algorithm and protocol selected for each collective at
       communicator initialization.

Related topics
==============

- :doc:`Collective operations in RCCL <./collective-operations>` — what each
  collective does and the communicator model
- :doc:`Hardware-specific optimizations <./hardware-specific-optimizations>` —
  how tuning models, Rome models, and protocol selection work per GPU generation
- :doc:`Run RCCL-Tests <../how-to/running-rccl-tests>` — how to measure which
  algorithm is fastest on your hardware
- :doc:`RCCL usage tips <../how-to/rccl-usage-tips>` — environment variables
  and production tuning guidance
- :ref:`Environment variables <env-variables>` — complete reference for all
