.. meta::
   :description: RCCL is a stand-alone library that provides multi-GPU and multi-node collective communication primitives optimized for AMD GPUs
   :keywords: RCCL, ROCm, library, API

.. _library-specification:

============================
RCCL library specification
============================

This document provides details of the API library. 

Communicator functions
----------------------

.. doxygenfunction:: ncclGetUniqueId

.. doxygenfunction:: ncclCommInitRank

.. doxygenfunction:: ncclCommInitAll

.. doxygenfunction:: ncclCommDestroy

.. doxygenfunction:: ncclCommAbort

.. doxygenfunction:: ncclCommCount

.. doxygenfunction:: ncclCommCuDevice

.. doxygenfunction:: ncclCommUserRank

.. _communicator-suspend-resume:

Communicator suspend and resume
-------------------------------

These functions release and reacquire the resources held by a communicator
that is temporarily idle, and report per-communicator memory statistics. They
are useful when an application wants to free GPU memory held by an inactive
communicator, and then later resume collective operations on the same communicator.

Releasing the physical backing of a suspended communicator requires virtual
memory management (VMM) support enabled through ``NCCL_CUMEM_ENABLE``. Without
it, ``ncclCommSuspend`` and ``ncclCommResume`` succeed but don't release GPU
memory. See :ref:`suspend-resume` for the full list of prerequisites.

Use ``ncclGroupStart`` and ``ncclGroupEnd`` when suspending or resuming multiple
communicators from one thread. Requests run at ``ncclGroupEnd`` after all ranks
of each affected communicator synchronize.

.. doxygendefine:: NCCL_SUSPEND_MEM

.. doxygenfunction:: ncclCommSuspend

.. doxygenfunction:: ncclCommResume

.. doxygenenum:: ncclCommMemStat_t

.. doxygenfunction:: ncclCommMemStats

Collective communication operations
-----------------------------------

Collective communication operations must be called separately for each communicator in a communicator clique.

They return when operations have been enqueued on the hipstream.

Since they may perform inter-CPU synchronization, each call has to be done from a different thread or process, or need to use Group Semantics (see below).

.. doxygenfunction:: ncclReduce

.. doxygenfunction:: ncclBcast

.. doxygenfunction:: ncclBroadcast

.. doxygenfunction:: ncclAllReduce

.. doxygenfunction:: ncclReduceScatter

.. doxygenfunction:: ncclAllGather

.. doxygenfunction:: ncclSend

.. doxygenfunction:: ncclRecv

.. doxygenfunction:: ncclGather

.. doxygenfunction:: ncclScatter

.. doxygenfunction:: ncclAllToAll

Group semantics
---------------
When managing multiple GPUs from a single thread, and since NCCL collective
calls may perform inter-CPU synchronization, we need to "group" calls for
different ranks/devices into a single call.

Grouping NCCL calls as being part of the same collective operation is done
using ncclGroupStart and ncclGroupEnd. ncclGroupStart will enqueue all
collective calls until the ncclGroupEnd call, which will wait for all calls
to be complete. Note that for collective communication, ncclGroupEnd only
guarantees that the operations are enqueued on the streams, not that
the operation is effectively done.

Both collective communication and ncclCommInitRank can be used in conjunction
of ncclGroupStart/ncclGroupEnd.

.. doxygenfunction:: ncclGroupStart

.. doxygenfunction:: ncclGroupEnd

Library functions
-----------------

.. doxygenfunction:: ncclGetVersion

.. doxygenfunction:: ncclGetErrorString

Types
-----

There are few data structures that are internal to the library. The pointer types to these
structures are given below. The user would need to use these types to create handles and pass them
between different library functions.

.. doxygentypedef:: ncclComm_t

.. doxygenstruct:: ncclUniqueId



Enumerations
------------

This section provides all the enumerations used.

.. doxygenenum:: ncclResult_t

.. doxygenenum:: ncclRedOp_t

.. _rccl-supported-data-types:

.. doxygenenum:: ncclDataType_t
