/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_BARRIER_H_
#define RCCL_BARRIER_H_

#include "nccl.h"
#include "comm.h"
#include <hip/hip_runtime.h>

/*
 * RCCL Profiling Barrier
 *
 * This module provides a mechanism to insert a synchronization barrier before
 * collective operations to help users identify the source of timing variations
 * in their profiling data.
 *
 * When RCCL_INSERT_BARRIER=1 is enabled, a ring-based AllReduce barrier
 * is inserted before each group of collectives. The barrier:
 *   1. Uses a 1MB AllReduce to engage all channels/CUs
 *   2. Synchronizes all ranks across nodes
 *   3. Forces stream completion before the actual collective
 *
 * The barrier operation appears as "rcclProfilingBarrier" in profilers.
 *
 * IMPORTANT: The RCCL team has NO CONTROL over how long rcclProfilingBarrier
 * takes. This time reflects application-level synchronization issues that users
 * must address in their own code.
 *
 * NOTE: This barrier adds latency due to stream synchronization. It is intended
 * for profiling/debugging only, not production use.
 *
 * Usage:
 *   export RCCL_INSERT_BARRIER=1
 *   # Run your application with profiler (rocprof, omniperf, etc.)
 *   # Look for "rcclProfilingBarrier" in the trace
 */

// Get the parameter value for RCCL_INSERT_BARRIER
int64_t rcclParamInsertBarrier();

// Insert a barrier before collective operations
// This executes a ring AllReduce and waits for completion
ncclResult_t rcclInsertProfilingBarrier(ncclComm_t comm, hipStream_t stream);

// Reset the barrier-inserted flag when a group ends
void rcclResetBarrierGroupFlag();

// Check if barrier is currently being inserted (to prevent recursion)
bool rcclIsBarrierInProgress();

#endif // RCCL_BARRIER_H_
