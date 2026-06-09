/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host path: launch meta::comms::ddaAllReduceFlatIpc from ncclAllReduce.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_ALL_REDUCE_IPC_H_
#define DDA_ALL_REDUCE_IPC_H_

#include "nccl.h"

struct ncclComm;

bool ncclAllReduceDdaIpcEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op);

ncclResult_t ncclAllReduceDdaIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream);

#endif
