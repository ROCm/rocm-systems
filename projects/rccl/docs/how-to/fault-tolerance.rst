.. meta::
   :description: How RCCL handles errors and supports fault tolerance for multi-GPU and multi-node collective communication on AMD GPUs
   :keywords: RCCL, ROCm, AMD, fault tolerance, error handling, communicator abort, shrink, grow, revoke

.. _fault-tolerance:

***************************
Fault tolerance in RCCL
***************************

Large-scale jobs running across many AMD GPUs and nodes must survive failures
such as a network link going down, an ECC error, a node crash, or a process that
exits unexpectedly. RCCL provides APIs to detect these conditions, release the
affected resources, and keep running without restarting the whole job.

RCCL inherits the NCCL error-handling model and adds AMD-specific extensions on
top of it. The APIs are declared in ``rccl.h`` and use the standard NCCL
signatures.

.. note::

   Fault tolerance relies on communicators created in non-blocking mode. With
   ``config.blocking = 0``, every RCCL call (except :cpp:func:`ncclCommDestroy`
   and :cpp:func:`ncclCommAbort`) returns immediately, so the application can
   react to a hang or a failure instead of being stuck inside a collective.

.. _ft-error-codes:

Error handling and communicator abort
=====================================

Every RCCL call returns an ``ncclResult_t`` code. Set ``NCCL_DEBUG=WARN`` to
print a human-readable message on error, or ``NCCL_DEBUG=INFO`` to also print the
call stack.

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Error
     - Description
     - Handling
   * - ``ncclSuccess``
     - No error.
     - None.
   * - ``ncclUnhandledCudaError``
     - Error during a HIP (ROCm runtime) call.
     - Fatal. Abort the communicator and re-create it.
   * - ``ncclSystemError``
     - Error during a system call, for example a network failure.
     - Fatal. Abort the communicator and re-create it.
   * - ``ncclInternalError``
     - A bug inside RCCL.
     - Fatal. Abort the communicator and report the issue to AMD.
   * - ``ncclInvalidArgument``
     - An argument is invalid, for example a ``NULL`` pointer.
     - Non-fatal. The call had no effect; the communicator is still usable.
   * - ``ncclInvalidUsage``
     - The sequence of RCCL calls is invalid.
     - Fatal. Abort the communicator and re-create it.
   * - ``ncclRemoteError``
     - A remote process exited or a network error occurred.
     - Fatal. Abort the communicator, then shrink out the failed ranks or re-create it.
   * - ``ncclInProgress``
     - The call is still running (non-blocking mode).
     - Poll with :cpp:func:`ncclCommGetAsyncError`.
   * - ``ncclTimeout``
     - An operation exceeded its time limit, for example an unresponsive peer.
     - Fatal. Abort the communicator, then shrink out the failed ranks or re-create it.

A fatal error applies to all communicators in the same group. To recover, call
:cpp:func:`ncclCommAbort` on every affected communicator and re-create it.

.. _async-errors:

Asynchronous errors
===================

Network failures are not reported by the originating call; they surface later
through :cpp:func:`ncclCommGetAsyncError`. In non-blocking mode, poll it until the
communicator leaves the ``ncclInProgress`` state, yielding the CPU between checks.

.. code-block:: cpp

   #include <sched.h>

   // Wait for a non-blocking communicator to reach a terminal state.
   ncclResult_t waitForAsyncResult(ncclComm_t comm) {
       ncclResult_t state = ncclInProgress;
       while (state == ncclInProgress) {
           ncclResult_t err = ncclCommGetAsyncError(comm, &state);
           if (err != ncclSuccess) return err;   // failed to query the state
           if (state == ncclInProgress) sched_yield();
       }
       return state;
   }

When waiting for a collective to finish, query the stream and poll for
asynchronous errors at the same time, instead of blocking in
``hipStreamSynchronize``, which can hang forever if a peer has failed.

.. code-block:: cpp

   int streamSyncWithAbort(hipStream_t stream, ncclComm_t comm) {
       while (true) {
           hipError_t hipErr = hipStreamQuery(stream);
           if (hipErr == hipSuccess) return 0;                 // collective done
           if (hipErr != hipErrorNotReady) return 1;           // HIP failure

           ncclResult_t asyncErr = ncclSuccess;
           ncclCommGetAsyncError(comm, &asyncErr);
           if (asyncErr != ncclSuccess) {                      // peer/network died
               ncclCommAbort(comm);
               return 2;
           }
           sched_yield();
       }
   }

.. _ft-recovery:

Recovering from a failure
=========================

RCCL offers several recovery strategies, depending on how much of the job you
want to preserve:

* **Abort and re-create** the whole communicator -- a coarse, job-wide restart.
* **Shrink** to drop the failed ranks and keep running with the survivors.
* **Grow** to bring replacement ranks back in and return to full size.
* **Revoke** (RCCL-specific) to abort in-flight work without destroying the
  communicator, so it can be reused as a parent for a later shrink or grow.

Abort requires non-blocking communicators, and no thread may be inside an RCCL
call when :cpp:func:`ncclCommAbort` is invoked.

.. _ft-shrink:

Shrinking a communicator
========================

:cpp:func:`ncclCommShrink` creates a new communicator by removing ranks from an
existing one. Only the ranks that remain call it; the excluded ranks must not.
Remaining ranks are renumbered contiguously.

.. code-block:: cpp

   // Drop one failed rank and continue on the survivors.
   int        excludeList[] = { failedRank };
   int        excludeCount  = 1;
   ncclComm_t newComm       = nullptr;

   if (myRank != failedRank) {
       ncclResult_t res = ncclCommShrink(comm, excludeList, excludeCount,
                                         &newComm, nullptr, NCCL_SHRINK_ABORT);
       if (res != ncclSuccess) {
           ncclCommAbort(comm);
           return res;
       }

       // newComm has (nRanks - excludeCount) ranks, renumbered 0..N-1.
       int newRank = -1, newSize = 0;
       ncclCommUserRank(newComm, &newRank);
       ncclCommCount(newComm, &newSize);

       // ... run collectives on newComm ...

       ncclCommDestroy(newComm);
   }
   ncclCommDestroy(comm);   // parent no longer needed

Shrink flags:

* ``NCCL_SHRINK_DEFAULT`` -- the parent has no in-flight work, or it was already
  aborted by :cpp:func:`ncclCommRevoke`.
* ``NCCL_SHRINK_ABORT`` -- abort in-flight work on the parent first, then shrink.
  Use this when shrinking directly after a failure, without a preceding revoke.

.. _ft-grow:

Growing a communicator
======================

:cpp:func:`ncclCommGrow` creates a new communicator by adding ranks. A
coordinator generates a unique ID with :cpp:func:`ncclCommGetUniqueId` and
distributes it to the new ranks out of band (here using ``MPI_Bcast``).

* Existing root (coordinator): ``comm`` set, ``uniqueId = &growId``, ``rank = -1``.
* Existing non-root ranks: ``comm`` set, ``uniqueId = NULL``, ``rank = -1``.
* New ranks: ``comm = NULL``, ``uniqueId = &growId``, ``rank =`` the assigned rank
  (a non-NULL ``uniqueId`` is required here).

.. code-block:: cpp

   // Grow a communicator of `existing` ranks up to `newTotal` ranks.
   ncclComm_t newComm = nullptr;

   // 1. Coordinator generates the grow ID; the new ranks need it out of band.
   ncclUniqueId growId{};
   if (myRank == 0) {
       ncclCommGetUniqueId(comm, &growId);
   }
   MPI_Bcast(&growId, sizeof(growId), MPI_BYTE, 0, MPI_COMM_WORLD);

   // 2. Each rank calls grow with the arguments for its role.
   if (myRank == 0) {
       // Existing root: passes the ID.
       ncclCommGrow(comm, newTotal, &growId, -1, &newComm, nullptr);
   } else if (myRank < existing) {
       // Existing non-root: pass NULL (grow contract in rccl.h).
       ncclCommGrow(comm, newTotal, nullptr, -1, &newComm, nullptr);
   } else {
       // New rank: comm = NULL, passes the ID and its assigned rank.
       hipSetDevice(localDevice);
       ncclCommGrow(nullptr, newTotal, &growId, myRank, &newComm, nullptr);
   }

For a non-blocking grow, pass a config with ``blocking = 0`` and poll for
completion before using the new communicator.

.. code-block:: cpp

   ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
   config.blocking = 0;

   if (myRank == 0) {
       ncclCommGrow(comm, newTotal, &growId, -1, &newComm, &config);
   } else if (myRank < existing) {
       ncclCommGrow(comm, newTotal, nullptr, -1, &newComm, &config);
   } else {
       ncclCommGrow(nullptr, newTotal, &growId, myRank, &newComm, &config);
   }

   ncclResult_t asyncErr = ncclInProgress;
   while (asyncErr == ncclInProgress) {
       ncclCommGetAsyncError(newComm, &asyncErr);
   }
   // asyncErr == ncclSuccess once the grow has completed.

   ncclCommDestroy(comm);   // parent can be released after a successful grow

Notes:

* Existing ranks keep their rank numbers; new ranks are numbered from the parent
  size upward.
* The grow ID is single-use.
* The parent must have no outstanding operations when grow is called.

.. _ft-revoke:

Revoking a communicator
=======================

:cpp:func:`ncclCommRevoke` aborts in-flight collectives **without** destroying
the communicator, so it can recover from a peer that failed mid-collective.
Output buffers of an aborted collective contain undefined data.

The typical flow is *collective, revoke, shrink, continue*. Because revoke
already aborts in-flight work, the shrink uses ``NCCL_SHRINK_DEFAULT``.

.. code-block:: cpp

   // A collective is in flight on the parent when a peer fails.
   ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, parent, stream);

   // 1. Revoke aborts the in-flight collective but keeps `parent` usable.
   ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT);
   MPI_Barrier(MPI_COMM_WORLD);   // all healthy ranks agree to recover

   // 2. Build a smaller communicator from the survivors (DEFAULT, not ABORT).
   ncclComm_t child = NCCL_COMM_NULL;
   if (myRank != failedRank) {
       ncclCommShrink(parent, excludeList, excludeCount,
                      &child, nullptr, NCCL_SHRINK_DEFAULT);
   }
   MPI_Barrier(MPI_COMM_WORLD);

   // 3. Continue on the child communicator.
   if (child != NCCL_COMM_NULL) {
       ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, child, stream);
       hipStreamSynchronize(stream);
       ncclCommDestroy(child);
   }
   ncclCommDestroy(parent);

Behavior:

* New collectives on a revoked communicator return ``ncclInvalidUsage``.
* A revoked communicator stays valid as a parent for :cpp:func:`ncclCommSplit`
  and :cpp:func:`ncclCommShrink`, and can be torn down with
  :cpp:func:`ncclCommDestroy`.
* Revoking twice returns ``ncclInvalidArgument``.
* :cpp:func:`ncclCommFinalize` on a revoked communicator returns
  ``ncclInvalidUsage`` -- use :cpp:func:`ncclCommDestroy` instead.
* ``revokeFlags`` must be ``NCCL_REVOKE_DEFAULT``.

.. _ft-finalize-destroy:

Finalizing and destroying
=========================

For a clean shutdown, call :cpp:func:`ncclCommFinalize` to drain outstanding
operations, then :cpp:func:`ncclCommDestroy` to free resources.

.. code-block:: cpp

   // Clean shutdown: drain, then free.
   ncclCommFinalize(comm);                 // moves comm to ncclInProgress

   ncclResult_t state = ncclInProgress;
   while (state == ncclInProgress) {       // wait until globally quiescent
       ncclCommGetAsyncError(comm, &state);
   }

   ncclCommDestroy(comm);                  // non-blocking once state is ncclSuccess

Do not access a communicator after it is destroyed. Use
:cpp:func:`ncclCommAbort` instead when the communicator is in a bad state and
outstanding operations cannot be drained.
