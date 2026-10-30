/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the DDA broadcast paths launched from ncclBroadcast.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_BROADCAST_H_
#define DDA_BROADCAST_H_

#include "nccl.h"

#include <cstdint>

struct ncclComm;

// Rank-uniform inputs only: no buffers (sendbuff is unused on non-roots), no root.
bool ncclBroadcastDdaFabricLLEligible(ncclComm* comm, size_t count, ncclDataType_t datatype);
bool ncclBroadcastDdaFabricLL128Eligible(ncclComm* comm, size_t count, ncclDataType_t datatype);

// Total CTAs each launcher would use.
uint32_t ncclBroadcastDdaFabricLLBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype);
uint32_t ncclBroadcastDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype);

ncclResult_t ncclBroadcastDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                      int root, ncclComm* comm, cudaStream_t stream);

ncclResult_t ncclBroadcastDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         int root, ncclComm* comm, cudaStream_t stream);

#endif
