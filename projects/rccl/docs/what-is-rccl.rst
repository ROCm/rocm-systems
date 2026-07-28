.. meta::
   :description: Learn about RCCL, AMD's collective communication library for multi-GPU and multi-node workloads on ROCm, and how it relates to NCCL.
   :keywords: RCCL, ROCm, collective communication, multi-GPU, NCCL, distributed training, AllReduce, AllGather, AMD GPU

.. _what-is:

What is RCCL?
=============

The ROCm Communication Collectives Library (RCCL, pronounced "rickle") is
AMD's open-source library for collective communication across multiple GPUs
and multiple nodes. It is the primary communication backbone for distributed
workloads on AMD hardware — from multi-GPU training runs on a single server
to large-scale cluster jobs spanning hundreds of nodes.

RCCL is forked from the NVIDIA Collective Communications Library (NCCL) and
implements an identical API, so applications that already use NCCL can run on
AMD hardware without source changes.

What RCCL does
--------------

RCCL provides the communication primitives that parallel and distributed
workloads need to coordinate data across GPUs. It handles two broad categories
of operations:

**Collective operations** involve all participating GPUs simultaneously:

- ``AllReduce`` — reduce a tensor across all GPUs and distribute the result
- ``AllGather`` — gather data from all GPUs and give every GPU the full tensor
- ``ReduceScatter`` — reduce across all GPUs and scatter the result
- ``Broadcast`` — send data from one GPU to all others
- ``Reduce`` — reduce across all GPUs and collect the result on one GPU
- ``AllToAll`` and ``AllToAllv`` — exchange distinct data between every pair of GPUs
- ``Gather`` and ``Scatter`` — collect or distribute data at a root GPU

**Point-to-point operations** transfer data directly between two GPUs:

- ``Send`` and ``Recv`` — direct GPU-to-GPU data transfer

For the full API, see the :ref:`API reference <api-library>`.

Why RCCL is useful
------------------

Training large neural networks or running large-scale simulations requires
distributing data and model state across many GPUs. Without a library like
RCCL, each application would need to implement its own communication logic,
handle hardware topology, manage network transports, and tune for bandwidth
and latency — a significant and error-prone engineering effort.

RCCL abstracts this complexity. It automatically:

- Discovers the hardware topology (GPUs, CPUs, NICs, and interconnects)
- Selects the best algorithm for each collective based on message size, node
  count, and topology
- Selects the best protocol (Low Latency, LL128, or Simple) for the target
  hardware
- Routes data over the highest-speed path available: xGMI for intra-node
  GPU-to-GPU, PCIe as a fallback, and InfiniBand, RoCE (RDMA over Converged
  Ethernet), or TCP/IP for inter-node traffic
- Uses GPUDirect RDMA (GDR) and DMA-BUF to move data directly between the
  NIC and GPU memory, bypassing the CPU

The result is that distributed frameworks such as PyTorch, JAX, and
TensorFlow-ROCm can offload all communication to RCCL and get near-peak
interconnect bandwidth without application-level tuning.

Who should use RCCL
-------------------

RCCL is the right choice for you if you are:

- **Running distributed deep learning** on AMD GPUs. RCCL integrates
  directly with PyTorch (via ``torch.distributed``), JAX, and
  TensorFlow-ROCm. You likely use RCCL already without calling it directly.

- **Building a distributed framework or runtime** on ROCm. RCCL exposes a
  C API that you can call from any language or runtime that supports C
  foreign-function interfaces.

- **Porting a NCCL-based application to AMD hardware**. Because RCCL is
  API-compatible with NCCL, a recompile against ``librccl.so`` is usually
  all that is required.

- **Developing or benchmarking MPI-based HPC applications** that need
  high-performance GPU-aware collectives alongside MPI.

RCCL is not a user-facing machine learning tool. You interact with it through
a framework or by linking against the library directly in your own code.

How RCCL works
--------------

When you initialize a communicator, RCCL performs topology discovery to build
a complete model of the hardware — which GPUs can communicate directly over
xGMI, which require PCIe, and which nodes are reachable over the network.
This model drives all subsequent decisions.

For each collective call, RCCL selects an algorithm from its supported set:

- **Ring** — data travels around a ring of GPUs; scales well for large messages
- **Tree** — a reduction tree that reduces latency for small messages
- **Hierarchical** — two-level approach for multi-node, combining fast
  intra-node communication with inter-node transfers
- **CollNet** — offloads the reduction to a smart network switch when
  available

Within the chosen algorithm, RCCL picks a protocol that matches the
latency-bandwidth tradeoff for the message size and hardware. Operations are
launched from the host and executed on the GPU; a CPU proxy thread handles
network-side progress for multi-node operations.

For a deeper look at the architecture, see the
`RCCL topology and collective operations <https://deepwiki.com/ROCm/rocm-systems/5.1-rccl-topology-and-collective-operations>`_
page on DeepWiki.

NCCL compatibility
------------------

RCCL tracks NCCL closely. Each RCCL release declares compatibility with one or
more NCCL versions, meaning the API, behavior, and environment variables from
those NCCL versions are supported. Applications do not need to be rewritten or
reconfigured to run on RCCL; in most cases, relinking against ``librccl.so``
is sufficient.

For the full compatibility history, see the
:doc:`RCCL release notes <./release-notes>`.

.. note::

   Some NCCL environment variables and features are NVIDIA-specific (for
   example, those that depend on NVLink Switch or CUDA driver internals) and
   have no effect or equivalent on AMD hardware. RCCL documents these
   differences where they are relevant.

Related topics
--------------

- :doc:`Install RCCL <./install/installation>` — get RCCL running on your system
- :doc:`Build RCCL from source <./install/building-installing>` — build options and CMake configuration
- :doc:`RCCL usage tips <./how-to/rccl-usage-tips>` — environment variables and tuning guidance
- :doc:`Performance optimization guide <./how-to/rccl-usage-tips>` — algorithm and protocol tuning
- :ref:`API reference <api-library>` — full C API documentation
- :ref:`Environment variables <env-variables>` — complete list of runtime knobs
