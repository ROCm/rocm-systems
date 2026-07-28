.. meta::
   :description: Learn how collective communication operations work, why they are essential to HPC and distributed AI training, and how RCCL implements them on AMD GPUs.
   :keywords: RCCL, collective communication, AllReduce, AllGather, ReduceScatter, ring algorithm, tree algorithm, MPI, HPC, distributed training, ROCm, GPU communication

.. _collective-operations:

Collective operations in RCCL
==============================

Distributed workloads — whether training a large language model (LLM) across
hundreds of GPUs or running a fluid-dynamics simulation across a cluster —
cannot fit all their data on a single processor. Work is split across many
processes, each operating on a local slice. At regular intervals those
processes must communicate: sharing partial results, synchronizing state, or
exchanging data. The operations that coordinate all participants simultaneously
are called **collective operations**.

RCCL implements the full vocabulary of collective operations defined by the
Message Passing Interface (MPI) standard and extends it with AMD-specific
optimizations. Understanding what these operations do, why they matter, and
how RCCL executes them is the foundation for understanding RCCL's performance
characteristics and tuning options.

What collective communication is
---------------------------------

Traditional point-to-point communication moves data from one sender to one
receiver. Collective communication involves a *group* of processes that all
participate in the same operation at the same time. Every process in the group
both contributes data and receives a result (or contributes data that others
receive, depending on the operation).

A **communicator** defines the group. In MPI this is an ``MPI_Comm`` object;
in RCCL it is an ``ncclComm_t`` handle. All collective calls are scoped to a
communicator, and every rank (process or GPU) in that communicator must call
the operation for it to proceed. This synchronization contract is what
distinguishes collectives from ordinary sends and receives.

The Message Passing Interface
------------------------------

The `Message Passing Interface (MPI) <https://www.mpi-forum.org/>`_ is the
portable, vendor-neutral standard that defines how distributed processes
exchange data. First published in 1994 and updated most recently with MPI 4.0
in 2021, it is the de facto language of HPC. MPI specifies a rich set of
collective operations — Reduce, Broadcast, Scatter, Gather, AllReduce,
AllGather, ReduceScatter, and AllToAll — and the semantic contracts each must
satisfy.

GPU-centric libraries such as NCCL and RCCL inherit this vocabulary. The
operation names, semantics, and communicator model all trace back to MPI.
RCCL can interoperate with MPI-based applications: you can initialize RCCL
communicators from MPI ranks and layer RCCL collectives on top of MPI for the
GPU-communication portion of your workload.

Why collective communication is critical for HPC
-------------------------------------------------

Modern HPC applications use collective communication for several reasons:

**Gradient synchronization in data-parallel training.** In synchronous
data-parallel training, each GPU processes a different mini-batch and computes
local gradients. Before the optimizer step, every GPU must have the same
gradient — the average across all GPUs. An AllReduce sums the gradients across
all ranks and distributes the result back to every GPU. Without it, each GPU
would apply a different update and the replicas would diverge.

**Tensor parallelism.** Large models shard individual weight matrices across
GPUs. An AllReduce or ReduceScatter is required at each layer boundary to
recombine partial results before the next layer can run.

**Pipeline parallelism.** Different pipeline stages may live on different nodes.
Point-to-point Send/Recv operations carry activations forward and gradients
backward across stage boundaries.

**Scientific computing.** Finite-element solvers, molecular dynamics codes, and
climate models all distribute a spatial domain across processes. At each time
step, boundary values must be exchanged with neighboring partitions via
collectives or structured point-to-point operations.

The performance of the collective communication layer directly sets the ceiling
on how well any of these workloads can scale.

Collective operations defined
------------------------------

The following table summarizes the operations RCCL supports. In each case,
*N* denotes the number of GPUs (ranks) in the communicator.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Operation
     - What it does
   * - ``ncclAllReduce``
     - Applies a reduction operator (Sum, Prod, Min, Max, or custom) across
       the input buffers of all ranks, then delivers the identical result to
       every rank. This is the most commonly used collective in distributed
       deep learning.
   * - ``ncclAllGather``
     - Each rank contributes a chunk; every rank receives the concatenation
       of all chunks in rank order. The output on each rank is *N* times the
       size of its input.
   * - ``ncclReduceScatter``
     - Applies a reduction across all ranks and then scatters distinct
       portions of the result to each rank. The output on each rank is 1/*N*
       of the total reduced size.
   * - ``ncclBroadcast``
     - Copies the input buffer of one designated root rank to all other ranks.
   * - ``ncclReduce``
     - Reduces across all ranks and delivers the result only to the root rank.
   * - ``ncclGather``
     - Collects data from all ranks and assembles it at the root rank only.
   * - ``ncclScatter``
     - Distributes distinct portions of the root's buffer to each rank.
   * - ``ncclAllToAll``
     - Each rank sends a distinct chunk to every other rank and receives a
       distinct chunk from every other rank. Sometimes called a "transpose"
       or "personalized AllGather."
   * - ``ncclAllToAllv``
     - Variant of AllToAll where each send and receive count can differ per
       rank.
   * - ``ncclSend`` / ``ncclRecv``
     - Point-to-point primitives that send data from one specific rank to
       another. These can be grouped in an ``ncclGroupStart`` /
       ``ncclGroupEnd`` block to build custom communication patterns.

Reduction operators supported include ``ncclSum``, ``ncclProd``, ``ncclMin``,
``ncclMax``, and ``ncclAvg``. RCCL also exposes the RCCL-specific
``ncclAllReduceWithBias`` for fused AllReduce with elementwise bias
accumulation.

For the complete API signatures, see the :ref:`RCCL API reference <api-library>`.

How RCCL executes collective operations
----------------------------------------

RCCL's execution path has four main stages: topology discovery, algorithm
selection, protocol selection, and kernel dispatch.

Topology discovery
^^^^^^^^^^^^^^^^^^

When you call ``ncclCommInitRank``, RCCL builds a complete model of the
hardware. It enumerates GPUs, CPUs, network interface cards (NICs), and every
interconnect between them — xGMI (Infinity Fabric) links, PCIe switches, and
network fabric. This model is stored in the ``ncclTopoSystem`` structure and
is used to compute the optimal communication path between every pair of GPUs.

On AMD systems, RCCL matches the detected topology against predefined
``rcclRomeModel`` structures that encode hardware-specific ring orderings and
connectivity matrices for MI200 (gfx90a), MI300A, MI300X (gfx942), and MI350
(gfx950) platforms.

Algorithm selection
^^^^^^^^^^^^^^^^^^^^

For each collective call, RCCL evaluates several candidate algorithms and
selects the one with the lowest estimated execution time using the model:

.. code-block:: text

   time = (latency + nsteps × size ÷ bandwidth) × correctionFactor

The candidate algorithms are:

**Ring.** Data flows around a ring of GPUs. Each GPU simultaneously sends to
its successor and receives from its predecessor. After *N−1* steps every GPU
has contributed to and received the full result. Ring is the default for
bandwidth-bound workloads (large messages) because it fully utilizes every
link in the ring simultaneously.

AllReduce using Ring is decomposed into two phases:

1. **ReduceScatter** — each GPU reduces a slice of the data while passing
   partial sums around the ring. After *N−1* steps, every GPU holds the
   fully reduced result for one slice.
2. **AllGather** — each GPU broadcasts its fully reduced slice around the
   ring. After *N−1* more steps, every GPU holds all slices.

**Tree.** A binary reduction tree is used for latency-bound workloads (small
messages). The tree converges partial results in ``log₂(N)`` steps rather
than ``N−1``, so it out-performs Ring when the per-step latency dominates
over the bandwidth term.

**Hierarchical.** A two-level algorithm that uses fast intra-node
communication (xGMI) for the first level and inter-node network communication
for the second. This is the default for multi-node workloads with 8 or more
nodes for AllGather and ReduceScatter in recent RCCL releases.

**CollNet.** When an in-network computing switch is available (for example,
a SHARP-enabled InfiniBand switch), RCCL can offload the reduction to the
switch fabric, freeing GPU resources and reducing round-trip latency.
CollNetDirect and CollNetChain are two variants.

**PAT (Port-Aggregated Topology).** An algorithm that aggregates bandwidth
across multiple NIC ports for workloads that exhaust a single port's capacity.

The selection is performed by ``getAlgoInfo()`` in ``src/graph/tuning.cc`` and
takes into account message size, GPU generation, node count, and whether
specific hardware features (such as in-network compute) are present.

Protocol selection
^^^^^^^^^^^^^^^^^^

Independently of the algorithm, RCCL selects one of three wire protocols that
control how data is packetized, flagged, and acknowledged on each hop:

**LL (Low Latency).** Sends data in small, flag-tagged packets. Each 8-byte
packet carries 4 bytes of data and a 4-byte flag. The receiver polls the flag
to detect arrival without any memory fence. LL achieves the lowest possible
latency but at the cost of 2× bandwidth overhead (50% of each packet is
metadata). LL is selected for the smallest messages, typically under a few
kilobytes.

**LL128.** A hybrid protocol that uses 128-byte aligned atomic writes — 120
bytes of data and an 8-byte flag. The larger payload-to-flag ratio delivers
approximately 95% of peak hardware bandwidth while retaining flag-based
(fence-free) synchronization. LL128 is selected for medium-sized messages and
requires hardware support for 128-byte atomics. On AMD hardware, LL128 support
is architecture-specific; it is enabled for gfx942 with 4-NIC configurations
and gfx950 by default.

**Simple.** Divides the data into large chunks and uses standard memory fences
for synchronization. Simple maximizes sustained bandwidth for large messages
and is selected when the message is large enough that the fence overhead is
negligible compared to the transfer time.

The decision is made at enqueue time in ``src/enqueue.cc`` based on the
estimated time model and the target GPU architecture.

Channels
^^^^^^^^^

Each communicator has a set of **channels** — independent ring or tree
instances that can operate in parallel. Running multiple channels multiplies
effective bandwidth by keeping more links busy simultaneously. The default
channel count is architecture- and topology-dependent; for example, RCCL
defaults to 112 channels for single-node 8× gfx950 configurations, and 48
channels for gfx950 multi-node configurations.

You can override the channel count with ``NCCL_MIN_NCHANNELS`` and
``NCCL_MAX_NCHANNELS``. Fewer channels reduce initialization overhead and are
useful when GPU resources are scarce; more channels increase bandwidth
utilization for large messages.

Kernel dispatch and the proxy service
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the algorithm, protocol, and channel count are determined, RCCL builds a
``ncclKernelPlan`` and launches GPU kernels to carry out the data movement.
Device kernels (``ncclDevKernel_Generic_*``) perform the actual reduction and
memory copies on-GPU.

For multi-node operations, a CPU **proxy thread** runs alongside the GPU
kernel. The proxy handles network-side progress — posting RDMA send and
receive work requests to the InfiniBand or RoCE NIC — so the GPU kernel does
not need to block on network events. This design allows the GPU to stream data
through its memory system at full speed while the CPU proxy keeps the network
pipeline full.

Transport mechanisms
^^^^^^^^^^^^^^^^^^^^^

The transport layer determines the physical path used for each hop:

- **P2P (peer-to-peer)** — direct GPU-to-GPU transfer using xGMI (Infinity
  Fabric) or PCIe peer access. No CPU involvement.
- **SHM (shared memory)** — used for intra-node communication when P2P access
  is not available. Data is staged through CPU-visible shared memory.
- **NET (network)** — used for inter-node communication. RCCL supports
  InfiniBand Verbs, RoCE (RDMA over Converged Ethernet), and TCP/IP sockets.

For network transports, RCCL supports **GPUDirect RDMA (GDR)**, which allows
the NIC to read from and write to GPU memory directly without staging through
the CPU. GDR dramatically reduces inter-node latency. The related mechanism
**DMA-BUF** (Direct Memory Access Buffer) uses Linux kernel infrastructure to
export GPU memory as DMA-capable buffers for NIC access.

Transport selection is performed in ``src/graph/paths.cc`` and
``src/graph/connect.cc`` based on the topology model built at initialization.

MPI and RCCL together
-----------------------

Many HPC applications and distributed training frameworks combine MPI with
RCCL. The typical pattern is:

1. **MPI manages process launch.** ``mpirun`` (or a workload manager such as
   Slurm) spawns one process per GPU across all nodes.
2. **RCCL communicators are initialized from MPI ranks.** Each process uses
   its MPI rank and world size to call ``ncclCommInitRank``, establishing an
   RCCL communicator that matches the MPI world.
3. **GPU-to-GPU communication uses RCCL.** Collective operations on GPU tensors
   go through RCCL, which uses xGMI and InfiniBand directly without touching
   the CPU data path.
4. **CPU-to-CPU communication uses MPI.** Metadata, file I/O coordination, and
   non-GPU data continue to use MPI.

This division of labor lets each library do what it does best. RCCL achieves
near-peak GPU memory bandwidth and interconnect utilization because it owns the
full GPU communication path; MPI handles the control plane.

Tuning collective performance
-------------------------------

RCCL automatically selects algorithms and protocols, but several knobs let you
intervene when the defaults are suboptimal for your workload:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Environment variable
     - Effect
   * - ``NCCL_ALGO``
     - Force a specific algorithm (``Ring``, ``Tree``, ``CollNet``). Overrides
       automatic selection for all collectives.
   * - ``RCCL_OVERRIDE_ALGO``
     - Like ``NCCL_ALGO`` but enforces the selection without re-running the
       tuning model. Guarantees the algorithm is used regardless of heuristics.
   * - ``NCCL_PROTO``
     - Force a specific protocol (``LL``, ``LL128``, ``Simple``).
   * - ``RCCL_OVERRIDE_PROTO``
     - Like ``NCCL_PROTO`` but enforces the selection unconditionally.
   * - ``NCCL_MIN_NCHANNELS`` / ``NCCL_MAX_NCHANNELS``
     - Set the minimum and maximum number of channels.
   * - ``RCCL_CHANNEL_TUNING_ENABLE``
     - Enable channel tuning that overrides RCCL's internal
       ``threadThreshold``-based channel count adjustment.
   * - ``NCCL_DEBUG`` / ``NCCL_DEBUG_SUBSYS``
     - Enable diagnostic output. Set ``NCCL_DEBUG=INFO`` and
       ``NCCL_DEBUG_SUBSYS=TUNING`` to log the algorithm and protocol chosen
       for each collective.

For a comprehensive list of tuning variables, see the
:ref:`environment variables reference <env-variables>` and the
:doc:`RCCL usage tips <../how-to/rccl-usage-tips>` guide.

Measuring collective performance
----------------------------------

RCCL-Tests is the standard benchmarking suite for RCCL. It measures two key
metrics for each collective and message size:

- **Algorithm bandwidth (algBw)** — the effective data rate as seen by the
  application: ``data size ÷ time``.
- **Bus bandwidth (busBw)** — the actual bytes transferred on the interconnect,
  corrected for the number of hops the algorithm requires. For a Ring
  AllReduce with *N* ranks, ``busBw = algBw × 2(N−1)/N``. Bus bandwidth
  approaches the raw hardware limit as *N* grows.

Running RCCL-Tests on your system before tuning gives you a baseline that lets
you distinguish algorithm or configuration problems from hardware-level
constraints. See the
`RCCL-Tests documentation <https://deepwiki.com/ROCm/rccl-tests>`_ for usage
and interpretation guidance.

Related topics
---------------

- :doc:`What is RCCL? <../what-is-rccl>` — high-level overview of the library
- :ref:`RCCL API reference <api-library>` — full C API including all collective
  signatures, data types, and reduction operators
- :doc:`RCCL usage tips <../how-to/rccl-usage-tips>` — environment variables
  and performance tuning guidance
- :ref:`Environment variables <env-variables>` — complete list of runtime knobs
- :doc:`RCCL release notes <../release-notes>` — per-release algorithm and
  protocol changes

