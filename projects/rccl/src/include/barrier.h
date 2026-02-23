/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_BARRIER_H_
#define RCCL_BARRIER_H_

#include "nccl.h"
#include "comm.h"
#include <cuda_runtime.h>

/*
 * RCCL Profiling Barrier
 *
 * This module provides a mechanism to insert a synchronization barrier before
 * collective operations to help users identify the source of timing variations
 * in their profiling data.
 *
 * When RCCL_INSERT_BARRIER=1 is enabled, a lightweight barrier (1-byte AllReduce)
 * is inserted before each group of collectives. The barrier operation is named
 * "rcclWaitForAllRanksBarrier" to be easily identifiable in profiling tools.
 *
 * IMPORTANT: The RCCL team has NO CONTROL over how long rcclWaitForAllRanksBarrier
 * takes. This time reflects application-level synchronization issues that users
 * must address in their own code.
 *
 * Usage:
 *   export RCCL_INSERT_BARRIER=1
 *   # Run your application with profiler (rocprof, omniperf, etc.)
 *   # Look for "rcclWaitForAllRanksBarrier" in the trace
 */

// Check if barrier insertion is enabled
int64_t rcclParamInsertBarrier();

// Insert a barrier before collective operations
// This should be called from ncclEnqueueCheck for the first collective in a group
// Parameters:
//   comm: The communicator to use for the barrier
//   stream: The stream on which to execute the barrier
// Returns: ncclSuccess on success, error code on failure
ncclResult_t rcclInsertProfilingBarrier(ncclComm_t comm, hipStream_t stream);

// Reset the barrier-inserted flag when a group ends
void rcclResetBarrierGroupFlag();

// Check if barrier is currently being inserted (to prevent recursion)
bool rcclIsBarrierInProgress();

#endif // RCCL_BARRIER_H_
