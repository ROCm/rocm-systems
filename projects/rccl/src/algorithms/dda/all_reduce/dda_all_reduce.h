/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the DDA all-reduce paths launched from ncclAllReduce
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_ALL_REDUCE_H_
#define DDA_ALL_REDUCE_H_

#include "nccl.h"

#include <cstdint>

struct ncclComm;

// True when RCCL_DDA_NRANKS_RELAX=1 (allow 2..8-rank DDA IPC AllReduce). Default 0.
bool ncclDdaNranksRelaxEnabled();

// True when nRanks is a supported single-node DDA IPC participant count: exactly
// kDdaNranks by default, or any 2..kDdaNranks when RCCL_DDA_NRANKS_RELAX=1. Shared by
// the DDA IPC collectives (AllReduce / AllGather / ReduceScatter / AllToAll).
bool ncclDdaNranksSupported(int nRanks);

// IPC path (single node, kDdaNranks ranks by default; any 2..kDdaNranks when relax is set).
bool ncclAllReduceDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceDdaIpc(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

// Fabric path (runtime nRanks up to kDdaMaxNranks, single- or multi-node).
bool ncclAllReduceDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                    ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceDdaFabric(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

// LL-protocol fabric path (small-message fast lane, flag-based sync, no barrier).
//
// The tier has two variants -- one-shot, and a two-shot that transports only the
// shard each rank owns (count/nRanks per peer instead of count). Picking between
// them, including the DDA_LL / DDA_LL_TWOSHOT enables and thresholds, is internal
// to dda_all_reduce_fabric_ll.cu: Eligible reports whether either variant claims
// the message, and the entry point launches whichever one did.
bool ncclAllReduceDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op);

// The per-variant predicates the gate above is the disjunction of. Declared here
// so unit tests can exercise one tier without the other masking it; hidden in
// Release by -fvisibility=hidden, as the rest of the internal surface is.
bool ddaLLArOneShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op);

bool ddaLLArTwoShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                      ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

// LL128-protocol fabric path (mid-message fast lane, 128B lines, no barrier).
bool ncclAllReduceDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                         ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

// Total CTAs (grid blocks) each DDA allreduce launcher would use for the given
// operands. Mirrors the launch grid math so reporting reflects real occupancy.
uint32_t ncclAllReduceDdaIpcBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype);
uint32_t ncclAllReduceDdaFabricBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype);
uint32_t ncclAllReduceDdaFabricLLBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype);
uint32_t ncclAllReduceDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype);

#endif
