*********************
Collective Operations
*********************

Collective operations have to be called for each rank (hence CUDA device), using the same count and the same datatype, to form a complete collective operation.
Failure to do so will result in undefined behavior, including hangs, crashes, or data corruption.

.. _allreduce:

AllReduce
---------

The AllReduce operation performs reductions on data (for example, sum, min, max) across devices and stores the result in the receive buffer of every rank.

In a *sum* allreduce operation between *k* ranks, each rank will provide an array in of N values, and receive identical results in array out of N values,
where out[i] = in0[i]+in1[i]+…+in(k-1)[i].

.. figure:: images/allreduce.png
 :align: center

 All-Reduce operation: each rank receives the reduction of input values across ranks.

Related links: :c:func:`ncclAllReduce`.

.. _broadcast:

Broadcast
---------

The Broadcast operation copies an N-element buffer from the root rank to all the ranks.

.. figure:: images/broadcast.png
 :align: center

 Broadcast operation: all ranks receive data from a “root” rank.

Important note: The root argument is one of the ranks, not a device number, and is therefore impacted by a different rank to device mapping.

Related links: :c:func:`ncclBroadcast`.

.. _reduce:

Reduce
------

The Reduce operation performs the same operation as AllReduce, but stores the result only in the receive buffer of a specified root rank.

.. figure:: images/reduce.png
 :align: center

 Reduce operation: one rank receives the reduction of input values across ranks.

Important note: The root argument is one of the ranks (not a device number), and is therefore impacted by a different rank to device mapping.

Note: A Reduce, followed by a Broadcast, is equivalent to the AllReduce operation.

Related links: :c:func:`ncclReduce`.

.. _allgather:

AllGather
---------

The AllGather operation gathers N values from k ranks into an output buffer of size k*N, and distributes that result to all ranks.

The output is ordered by the rank index. The AllGather operation is therefore impacted by a different rank to device mapping.

.. figure:: images/allgather.png
 :align: center

 AllGather operation: each rank receives the aggregation of data from all ranks in the order of the ranks.

Note: Executing ReduceScatter, followed by AllGather, is equivalent to the AllReduce operation.

Related links: :c:func:`ncclAllGather`.

.. _reducescatter:

ReduceScatter
-------------

The ReduceScatter operation performs the same operation as Reduce, except that the result is scattered in equal-sized blocks between ranks,
each rank getting a chunk of data based on its rank index.

The ReduceScatter operation is impacted by a different rank to device mapping since the ranks determine the data layout.

.. figure:: images/reducescatter.png
 :align: center

 Reduce-Scatter operation: input values are reduced across ranks, with each rank receiving a subpart of the result.


Related links: :c:func:`ncclReduceScatter`

.. _alltoall:

AlltoAll
--------

In an AlltoAll operation between k ranks, each rank provides an input buffer of size k*N values, where the j-th chunk of N values is sent to destination rank j. Each rank receives an output buffer of size k*N values, where the i-th chunk of N values comes from source rank i.

.. figure:: images/alltoall.png
 :align: center

 AlltoAll operation: exchanges data between all ranks, where each rank sends different data to every other rank and receives different data from every other rank.

Related links: :c:func:`ncclAlltoAll`.

.. _gather:

Gather
------

The Gather operation gathers N values from k ranks into an output buffer on the root rank of size k*N.

.. figure:: images/gather.png
 :align: center

 Gather operation: root rank receives data from all ranks.

Important note: The root argument is one of the ranks, not a device number, and is therefore impacted by a different rank to device mapping.

Related links: :c:func:`ncclGather`.

.. _scatter:

Scatter
-------

The Scatter operation distributes a total of N*k values from the root rank to k ranks, each rank receiving N values.

.. figure:: images/scatter.png
 :align: center

 Scatter operation: root rank distributes data to all ranks.

Important note: The root argument is one of the ranks, not a device number, and is therefore impacted by a different rank to device mapping.

Related links: :c:func:`ncclScatter`.

.. _coll-config:

Per-collective configuration
----------------------------

Each collective has an ``nccl*Config`` entry point that takes a trailing ``ncclCollConfig_t*``.
``NULL`` is the same call as the plain collective. Initialize the struct with ``NCCL_COLLCONFIG_INITIALIZER``
and pass the same config on every rank.

.. code-block:: c++

   ncclCollConfig_t config = NCCL_COLLCONFIG_INITIALIZER;
   config.algSelection = "RING";
   config.minCTAs = 2;
   config.maxCTAs = 8;
   config.CTAPolicy = NCCL_CTA_POLICY_DEFAULT;
   config.userProfilerTag = 42;
   ncclAllReduceConfig(send, recv, count, ncclFloat, ncclSum, comm, stream, &config);

``algSelection`` limits that call to the named algorithms. The accepted names depend on the collective:
``RING`` and ``TREE`` for AllReduce, ``PAT`` for AllGather and ReduceScatter, plus the per-collective
``SYMK_*`` symmetric kernels.
An unknown name is an error unless ``forceAlgSelection`` is 0, in which case RCCL falls back to automatic selection.
``NCCL_ALGO`` and ``NCCL_PROTO`` still override the per-call string. ``minCTAs`` and ``maxCTAs`` bound the channel
count for that call. An unset ``CTAPolicy`` inherits the communicator policy and ``NCCL_CTA_POLICY`` overrides both;
note a per-call ``CTAPolicy`` steers the kernel-path tuners only, since the copy-engine backend is still chosen from
the communicator policy.
``userProfilerTag`` is copied into profiler events and does not change the algorithm. ``cgaClusterSize`` is accepted
and has no effect on HIP. Vendor options can be attached through ``ncclConfigExt_t`` on ``config.ext``; RCCL ignores
extensions it does not recognize.
