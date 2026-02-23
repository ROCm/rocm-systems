/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "barrier.h"
#include "collectives.h"
#include "debug.h"
#include "param.h"
#include "group.h"
#include "enqueue.h"
#include <hip/hip_runtime.h>

/*
 * RCCL Profiling Barrier Implementation
 *
 * This file implements a profiling barrier that helps users identify timing
 * skew caused by uneven rank startup times vs actual RCCL collective time.
 *
 * The barrier works by executing a minimal 1-byte AllReduce named
 * "rcclWaitForAllRanksBarrier" to synchronize all ranks before the actual
 * collective operations begin.
 *
 * IMPORTANT NOTE TO USERS:
 * =======================
 * The RCCL library team has NO CONTROL over the timing skew observed across
 * ranks in rcclWaitForAllRanksBarrier. This skew reflects:
 *   - Uneven workload distribution across ranks before the collective
 *   - System-level scheduling variations (OS, GPU driver, etc.)
 *   - Memory allocation/deallocation timing differences
 *   - Other application-level synchronization issues
 *
 * If you see significant skew between ranks in rcclWaitForAllRanksBarrier
 * (i.e., some ranks complete much earlier than others), you should investigate
 * your application's workload balance BEFORE the RCCL collective calls.
 * This skew is NOT an RCCL performance issue.
 *
 * USAGE:
 * ======
 * Set environment variable: RCCL_INSERT_BARRIER=1
 *
 * After enabling, your profiling traces will show:
 *   1. rcclWaitForAllRanksBarrier (absorbs startup time variations)
 *   2. Your actual collective operations (now with aligned start times)
 *
 * The timing skew between ranks in rcclWaitForAllRanksBarrier represents
 * the "startup skew" that was previously being incorrectly attributed to
 * RCCL collective performance.
 */

// Environment variable: RCCL_INSERT_BARRIER (default: 0 = disabled)
RCCL_PARAM(InsertBarrier, "INSERT_BARRIER", 0);

// Thread-local flag to prevent recursive barrier insertion
static __thread bool rcclBarrierInProgress = false;

// Thread-local flag to track if barrier was already inserted for this group
static __thread bool rcclBarrierInsertedForGroup = false;

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
    return ncclSuccess; // Silently skip if no communicator
  }

  // Allocate barrier buffer if not already done
  if (comm->barrierBuff == nullptr) {
    NCCLCHECK(ncclCudaCallocAsync(&comm->barrierBuff, 1024, stream));
  }

  INFO(NCCL_COLL, "RCCL: Inserting profiling barrier (rcclWaitForAllRanksBarrier) "
       "for comm %p rank %d/%d stream %p",
       comm, comm->rank, comm->nRanks, stream);

  // Print one-time warning about barrier behavior
  static bool warnedOnce = false;
  if (!warnedOnce) {
    WARN("RCCL_INSERT_BARRIER is enabled. rcclWaitForAllRanksBarrier will appear in profiling traces.\n"
         "NOTE: The RCCL team has NO CONTROL over how long this barrier takes.\n"
         "Time spent in this barrier reflects application-level startup skew, NOT RCCL performance.\n"
         "To reduce this time, ensure your application has balanced workloads before collectives.");
    warnedOnce = true;
  }

  // Set flags to prevent recursive insertion
  rcclBarrierInProgress = true;
  rcclBarrierInsertedForGroup = true;

  ncclResult_t ret = ncclSuccess;

  // Execute 1-byte AllReduce to synchronize all ranks
  // We use ncclSum with ncclInt8 for minimal overhead
  // The buffer content doesn't matter - we just need synchronization
  // The operation name "rcclWaitForAllRanksBarrier" will appear in profilers
  {
    struct ncclInfo info = { ncclFuncAllReduce, "rcclWaitForAllRanksBarrier",
      comm->barrierBuff, comm->barrierBuff, 1024, ncclInt8, ncclSum, 0, comm, stream,
      ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, nullptr,
      true /* isBarrier - use all channels and bypass optimizations */ };

    // Enqueue the barrier AllReduce
    ret = ncclEnqueueCheck(&info);
  }

  rcclBarrierInProgress = false;
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
