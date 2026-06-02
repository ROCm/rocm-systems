/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#ifndef RCCL_CPU_KERNEL_INTERNAL_H_
#define RCCL_CPU_KERNEL_INTERNAL_H_

#include "cpu_device_context.h"
#include "cpu_dev_comm_mirror.h"
#include "comm.h"
#include "device.h"

ncclResult_t rcclCpuLoadWorkBatch(
    struct rcclCpuBlockContext* ctx,
    struct ncclDevKernelArgs const* args,
    int batchIx,
    struct rcclCpuBlockBarrier* bar);

ncclResult_t rcclCpuDispatchWork(struct rcclCpuBlockContext* ctx, struct rcclCpuBlockBarrier* bar, int tid, int tn);

ncclResult_t rcclCpuExecuteBlock(
    struct ncclComm* comm,
    struct rcclCpuCommMirrorState* mirror,
    struct ncclKernelComm* hostComm,
    struct ncclDevKernelArgs* args,
    int blockId,
    int threadCount,
    int warpSize);

#endif
