/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the DDA all-to-all paths launched from ncclAllToAll.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_ALLTOALL_H_
#define DDA_ALLTOALL_H_

#include "nccl.h"

#include <cstdint>

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

// Total CTAs (grid blocks) each DDA alltoall launcher would use for the given
// operands. Mirrors the launch grid math so reporting reflects real occupancy.
uint32_t ncclAllToAllDdaIpcBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype);
uint32_t ncclAllToAllDdaFabricBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype);
uint32_t ncclAllToAllDdaFabricLLBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype);
uint32_t ncclAllToAllDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype);

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

// LL-protocol fabric path (small-chunk fast lane, 16B lines, no barrier).
bool ncclAllToAllDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                     ncclDataType_t datatype);

ncclResult_t ncclAllToAllDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                     ncclComm* comm, cudaStream_t stream);

// LL128-protocol fabric path (mid-chunk fast lane, 128B lines, no barrier).
bool ncclAllToAllDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                        ncclDataType_t datatype);

ncclResult_t ncclAllToAllDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                        ncclComm* comm, cudaStream_t stream);

#endif
