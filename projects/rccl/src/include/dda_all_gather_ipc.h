/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host path: launch meta::comms::ddaAllGatherFlatIpc from ncclAllGather.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_ALL_GATHER_IPC_H_
#define DDA_ALL_GATHER_IPC_H_

#include "nccl.h"

struct ncclComm;

/**
 * Check if DDA allgather is eligible for the given parameters
 */
bool ncclAllGatherDdaIpcEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype);

/**
 * Execute DDA allgather operation using IPC
 */
ncclResult_t ncclAllGatherDdaIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm* comm,
    cudaStream_t stream);

#endif
