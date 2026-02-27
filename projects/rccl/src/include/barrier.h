/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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
 * is inserted before each group of collectives. The barrier forces stream completion before the actual collective
 *
 * The barrier operation appears as "rcclProfilingBarrier" in profilers.
 *
 * The RCCL library team has NO CONTROL over the timing skew observed across
 * ranks in rcclProfilingBarrier. This time reflects application-level straggler issues that users
 * must address in their own code.
 *
 *
 * Usage:
 *   export RCCL_INSERT_BARRIER=1
 *   # Run your application with profiler (rocprof, omniperf, etc.)
 */

// Get the parameter value for RCCL_INSERT_BARRIER
int64_t rcclParamInsertBarrier();

// Insert a barrier before collective operations
// This executes a ring AllReduce and waits for completion
ncclResult_t rcclInsertProfilingBarrier(ncclComm_t comm, hipStream_t stream);

#endif // RCCL_BARRIER_H_
