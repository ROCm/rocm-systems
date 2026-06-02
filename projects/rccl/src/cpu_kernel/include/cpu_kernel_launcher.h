/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Optional CPU execution path for RCCL device collective kernels.
 *
 * Enable at runtime:
 *   export RCCL_CPU_KERNEL_ENABLE=1
 *
 * Optional tuning:
 *   export RCCL_CPU_KERNEL_THREADS=<max parallel blocks, default: channel count>
 *
 * See src/cpu_kernel/README.md for semantics and limitations.
 ************************************************************************/

#ifndef RCCL_CPU_KERNEL_LAUNCHER_H_
#define RCCL_CPU_KERNEL_LAUNCHER_H_

#include "comm.h"
#include "enqueue.h"

#include <hip/hip_runtime.h>

// Returns true when RCCL_CPU_KERNEL_ENABLE is non-zero.
bool rcclCpuKernelEnabled();

// True when this plan can be executed on the CPU path.
bool rcclCpuKernelPlanSupported(struct ncclComm* comm, struct ncclKernelPlan* plan);

// Launch collective kernel work on CPU threads with MI300 ordering, enqueueing
// completion on launchStream (same ordering contract as cuLaunchKernel).
ncclResult_t rcclLaunchKernelCpu(
    struct ncclComm* comm,
    struct ncclKernelPlan* plan,
    unsigned gridDimX,
    unsigned blockDimX,
    hipStream_t launchStream);

#endif  // RCCL_CPU_KERNEL_LAUNCHER_H_
