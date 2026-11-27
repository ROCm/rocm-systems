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

// Upper bound for the DDA Broadcast block-size env overrides.
constexpr unsigned kDdaBcMaxThreads = 512;

// Per-rank block budget: CU-derived (fabric: min across ranks; IPC: this GPU),
// lowered by RCCL_DDA_BC_MAX_BLOCKS. The root still gets one block per peer.
size_t ncclBroadcastDdaBlockBudget(const ncclComm* comm);

// Total CTAs each launcher would use.
uint32_t ncclBroadcastDdaFabricLLBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype);
uint32_t ncclBroadcastDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype);

ncclResult_t ncclBroadcastDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                      int root, ncclComm* comm, cudaStream_t stream);

ncclResult_t ncclBroadcastDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         int root, ncclComm* comm, cudaStream_t stream);

// Two-phase IPC Broadcast (single node, 8 ranks): root scatters 1/8 slices, then the DDA all-gather body runs.
// Its sizes are whole units of 8 ranks x 16 B, so each slice is whole 16-byte units and the all-gather needs no tail.
constexpr size_t kDdaBcIpcGranuleBytes = 128;
bool ncclBroadcastDdaIpcEligible(ncclComm* comm, size_t count, ncclDataType_t datatype);
uint32_t ncclBroadcastDdaIpcBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype);
ncclResult_t ncclBroadcastDdaIpc(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                                 ncclComm* comm, cudaStream_t stream);

#endif
