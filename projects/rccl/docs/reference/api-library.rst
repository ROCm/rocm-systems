.. meta::
   :description: Full Doxygen API reference for RCCL, AMD's collective communication library for multi-GPU and multi-node workloads on ROCm.
   :keywords: RCCL, ROCm, API reference, Doxygen, ncclAllReduce, ncclAllGather, ncclReduceScatter, ncclBroadcast, ncclSend, ncclRecv, collective communication, AMD GPU

.. _api-library:

***********
API library
***********

RCCL (pronounced "Rickle") is a stand-alone library of standard collective communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, gather, scatter, and all-to-all. There is also initial support for direct GPU-to-GPU send and receive operations. It has been optimized to achieve high bandwidth on platforms using PCIe, xGMI as well as networking using InfiniBand Verbs or TCP/IP sockets. RCCL supports an arbitrary number of GPUs installed in a single node or multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

The collective operations are implemented using ring and tree algorithms and have been optimized for throughput and latency. For best performance, small operations can be either batched into larger operations or aggregated through the API.

Operations
==========

.. doxygenindex::
