/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#ifndef RCCL_CPU_COLL_EXEC_H_
#define RCCL_CPU_COLL_EXEC_H_

#include "cpu_device_context.h"
#include "cpu_func_decode.h"

ncclResult_t rcclCpuExecuteCollWork(
    struct rcclCpuBlockContext* ctx,
    struct rcclCpuBlockBarrier* bar,
    int tid, int tn,
    struct ncclDevWorkColl* work,
    struct rcclCpuFuncDesc const& desc);

ncclResult_t rcclCpuExecuteP2pWork(
    struct rcclCpuBlockContext* ctx,
    struct rcclCpuBlockBarrier* bar,
    int tid, int tn,
    struct ncclDevWorkP2p* work,
    struct rcclCpuFuncDesc const& desc);

#endif
