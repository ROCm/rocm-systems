/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host path: launch meta::comms::ddaReduceScatterFlatIpc from ncclReduceScatter.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_REDUCE_SCATTER_IPC_H_
#define DDA_REDUCE_SCATTER_IPC_H_

#include "nccl.h"

struct ncclComm;

/**
 * Check if DDA reduce-scatter is eligible for the given parameters
 */
bool ncclReduceScatterDdaIpcEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op);

/**
 * Execute DDA reduce-scatter operation using IPC
 */
ncclResult_t ncclReduceScatterDdaIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream);

#endif
