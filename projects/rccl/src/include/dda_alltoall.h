/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the DDA all-to-all paths launched from ncclAllToAll.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_ALLTOALL_H_
#define DDA_ALLTOALL_H_

#include "nccl.h"

struct ncclComm;

/**
 * Check if DDA alltoall is eligible for the given parameters
 */
bool ncclAllToAllDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                ncclDataType_t datatype);

/**
 * Execute DDA alltoall operation using IPC
 */
ncclResult_t ncclAllToAllDdaIpc(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                ncclComm* comm, cudaStream_t stream);

/**
 * Check if DDA alltoall is eligible for the fabric/VMM path (runtime nRanks
 * up to kDdaMaxNranks, single- or multi-node within an MNNVL clique).
 */
bool ncclAllToAllDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                   ncclDataType_t datatype);

/**
 * Execute DDA alltoall operation using the fabric/VMM path
 */
ncclResult_t ncclAllToAllDdaFabric(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclComm* comm, cudaStream_t stream);

#endif
