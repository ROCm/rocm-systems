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

/*
 * RCCL Profiling Barrier Implementation
 *
 * This file implements a profiling barrier that helps users identify timing
 * skew caused by uneven rank startup times vs actual RCCL collective time.
 *
 * The barrier uses a ring-based AllReduce named "rcclProfilingBarrier"
 * to synchronize all ranks before the actual collective operations begin.
 *
 * To ensure the barrier completes BEFORE the actual collective (and isn't
 * aggregated with it), we:
 *   1. Launch the barrier AllReduce outside of any user group
 *   2. Force stream synchronization to ensure completion
 *
 * IMPORTANT NOTE TO USERS:
 * =======================
 * The RCCL library team has NO CONTROL over the timing skew observed across
 * ranks in rcclProfilingBarrier. This skew reflects:
 *   - Uneven workload distribution across ranks before the collective
 *   - System-level scheduling variations (OS, GPU driver, etc.)
 *   - Memory allocation/deallocation timing differences
 *   - Other application-level synchronization issues
 *
 * If you see significant skew between ranks in rcclProfilingBarrier
 * (i.e., some ranks complete much earlier than others), you should investigate
 * your application's workload balance BEFORE the RCCL collective calls.
 * This skew is NOT an RCCL performance issue.
 *
 * USAGE:
 * ======
 * Set environment variable: RCCL_INSERT_BARRIER=1
 *
 * After enabling, your profiling traces will show:
 *   1. rcclProfilingBarrier (absorbs startup time variations)
 *   2. Your actual collective operations (now with aligned start times)
 */

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
  // Use a larger size to engage more CUs during the AllReduce
  const size_t barrierSize = 1; // 1MB to ensure all channels are used
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

  // Synchronize the stream to ensure barrier completes before actual collective
  // This is necessary to prevent the barrier from being pipelined with the next collective
  CUDACHECKGOTO(hipStreamSynchronize(stream), ret, fail);

  // Restore the original group depth
  ncclGroupDepth = rcclSavedGroupDepth;

  rcclBarrierInProgress = false;
  // Reset the group flag so the next collective gets its own barrier
  rcclBarrierInsertedForGroup = false;
  return ncclSuccess;

fail:
  ncclGroupDepth = rcclSavedGroupDepth;
  rcclBarrierInProgress = false;
  // Reset the group flag so the next collective gets its own barrier
  rcclBarrierInsertedForGroup = false;
  return ret;
}

// Called when a group ends to reset the barrier-inserted flag
void rcclResetBarrierGroupFlag() {
  rcclBarrierInsertedForGroup = false;
}

// Check if barrier is currently being inserted (to prevent recursion)
bool rcclIsBarrierInProgress() {
  return rcclBarrierInProgress;
}
