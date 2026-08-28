/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the DDA all-gather paths launched from ncclAllGather.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_ALL_GATHER_H_
#define DDA_ALL_GATHER_H_

#include "nccl.h"

struct ncclComm;

/**
 * Check if DDA allgather is eligible for the given parameters
 */
bool ncclAllGatherDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                 ncclDataType_t datatype);

/**
 * Execute DDA allgather operation using IPC
 */
ncclResult_t ncclAllGatherDdaIpc(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                 ncclComm* comm, cudaStream_t stream);

/**
 * Check if DDA allgather is eligible for the fabric/VMM path (runtime nRanks
 * up to kDdaMaxNranks, single- or multi-node within an MNNVL clique).
 */
bool ncclAllGatherDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                    ncclDataType_t datatype);

/**
 * Execute DDA allgather operation using the fabric/VMM path
 */
ncclResult_t ncclAllGatherDdaFabric(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                    ncclComm* comm, cudaStream_t stream);

#endif
