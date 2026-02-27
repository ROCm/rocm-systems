/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "barrier.h"
#include "collectives.h"
#include "debug.h"
#include "param.h"
#include "group.h"
#include "enqueue.h"
#include "alloc.h"
#include <hip/hip_runtime.h>


// Environment variable: RCCL_INSERT_BARRIER (default: 0 = disabled)
RCCL_PARAM(InsertBarrier, "INSERT_BARRIER", 0);

// Thread-local flag to prevent recursive barrier insertion
static __thread bool rcclBarrierInProgress = false;

// Thread-local flag to track if barrier was already inserted for this group
static __thread bool rcclBarrierInsertedForGroup = false;

// Thread-local to save the current group depth
static __thread int rcclSavedGroupDepth = 0;

// External reference to ncclGroupDepth from group.cc
extern __thread int ncclGroupDepth;

ncclResult_t rcclInsertProfilingBarrier(ncclComm_t comm, hipStream_t stream) {
  // Check if barrier insertion is enabled
  if (!rcclParamInsertBarrier()) {
    return ncclSuccess;
  }

  // Prevent recursive barrier insertion (barrier uses AllReduce internally)
  if (rcclBarrierInProgress) {
    return ncclSuccess;
  }

  // Check if barrier was already inserted for this group
  if (rcclBarrierInsertedForGroup) {
    return ncclSuccess;
  }

  // Validate communicator
  if (comm == nullptr) {
    return ncclSuccess;
  }

  // Allocate barrier buffer if not already done
  const size_t barrierSize = 1;
  if (comm->barrierBuff == nullptr) {
    NCCLCHECK(ncclCudaCalloc(&comm->barrierBuff, barrierSize));
  }

  ncclResult_t ret = ncclSuccess;

  INFO(NCCL_COLL, "RCCL: Inserting profiling barrier (rcclProfilingBarrier) "
       "for comm %p rank %d/%d stream %p nChannels %d",
       comm, comm->rank, comm->nRanks, stream, comm->nChannels);

  static bool warnedOnce = false;
  if (!warnedOnce) {
    WARN("RCCL_INSERT_BARRIER is enabled.\n"
       "NOTE: The RCCL team has NO CONTROL over the skew observed in this barrier.\n"
       "Skew in this barrier reflects application-level startup skew between ranks, NOT RCCL performance.\n"
       "WARNING: This barrier adds latency due to stream synchronization.");
    warnedOnce = true;
  }

  // Set flags to prevent recursive insertion
  rcclBarrierInProgress = true;
  rcclBarrierInsertedForGroup = true;

  // Save the current group depth and temporarily reset to 0
  // This forces the barrier to execute as a standalone operation
  rcclSavedGroupDepth = ncclGroupDepth;
  ncclGroupDepth = 0;

  // Start a dedicated group for the barrier
  NCCLCHECKGOTO(ncclGroupStartInternal(), ret, fail);

  // Execute ring AllReduce to synchronize all ranks
  // Use a meaningful size to engage all channels/CUs
  {
    struct ncclInfo info = { ncclFuncAllReduce, "rcclProfilingBarrier",
      comm->barrierBuff, comm->barrierBuff, barrierSize, ncclInt8, ncclSum, 0, comm, stream,
      ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, nullptr
      };

    NCCLCHECKGOTO(ncclEnqueueCheck(&info), ret, fail);
  }

  // End the group and launch the barrier
  NCCLCHECKGOTO(ncclGroupEndInternal(), ret, fail);

  rcclBarrierInProgress = false;

  // Restore the original group depth
  ncclGroupDepth = rcclSavedGroupDepth;

  return ncclSuccess;

fail:
  ncclGroupDepth = rcclSavedGroupDepth;
  rcclBarrierInProgress = false;
  return ret;
}

// Reset the barrier group flag - called when a group ends
void rcclResetBarrierGroupFlag() {
  rcclBarrierInsertedForGroup = false;
}

// Check if barrier is currently in progress (for internal use)
bool rcclIsBarrierInProgress() {
  return rcclBarrierInProgress;
}
