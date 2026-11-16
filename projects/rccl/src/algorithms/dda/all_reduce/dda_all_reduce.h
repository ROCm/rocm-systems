/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the DDA all-reduce paths launched from ncclAllReduce
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_ALL_REDUCE_H_
#define DDA_ALL_REDUCE_H_

#include "nccl.h"
#include "rccl_decision.h"

#include <cstdint>

struct ncclComm;

// IPC path (single node, fixed kDdaNranks ranks).
bool ncclAllReduceDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceDdaIpc(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

// Fabric path (runtime nRanks up to kDdaMaxNranks, single- or multi-node).
bool ncclAllReduceDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                    ncclDataType_t datatype, ncclRedOp_t op);

// `cap` (rcclCollDecision::ddaCap) bounds the grid of a call that replaces another backend; the launchers below too.
ncclResult_t ncclAllReduceDdaFabric(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm* comm, cudaStream_t stream, rcclDdaLaunchCap cap = {});

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

// The LL128 tiers hang off the same gate. Their slot geometry is derived from
// the scratch bank rather than a fixed constant, so unlike the LL tiers their
// size cap moves with comm->ddaScratchBytes -- which is what the unit tests
// exercise by shrinking the scratch to a known number of slices.
bool ddaLL128ArOneShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op);

bool ddaLL128ArTwoShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                      ncclRedOp_t op, ncclComm* comm, cudaStream_t stream,
                                      rcclDdaLaunchCap cap = {});

// LL128-protocol fabric path (mid-message fast lane, 128B lines, no barrier).
bool ncclAllReduceDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                         ncclDataType_t datatype, ncclRedOp_t op);

// Whether the LL128 peer-split takes the message. *bound is the grid under `cap` of the launch it replaces, else `cap`.
bool ncclAllReduceDdaFabricLL128SplitEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                              ncclDataType_t datatype, ncclRedOp_t op, rcclDdaLaunchCap cap,
                                              bool vmmFirst, rcclDdaLaunchCap* bound);

// Whether the peer-split takes the message only through uneven shards where no DDA tier would run it (the native
// launch would): the caller then bounds the split by that launch's grid where it is Ring.
bool ncclAllReduceDdaFabricLL128SplitReplacesNonDda(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                                    ncclDataType_t datatype, ncclRedOp_t op);

// `split` runs the peer-split two-shot within `cap`, the bound ncclAllReduceDdaFabricLL128SplitEligible gave.
ncclResult_t ncclAllReduceDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         ncclRedOp_t op, ncclComm* comm, cudaStream_t stream, rcclDdaLaunchCap cap = {},
                                         bool split = false);

// Total CTAs (grid blocks) each DDA allreduce launcher would use for the given
// operands. Mirrors the launch grid math so reporting reflects real occupancy.
uint32_t ncclAllReduceDdaIpcBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype);
uint32_t ncclAllReduceDdaFabricBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype, rcclDdaLaunchCap cap = {});
uint32_t ncclAllReduceDdaFabricLLBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype,
                                        rcclDdaLaunchCap cap = {});
uint32_t ncclAllReduceDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype,
                                           rcclDdaLaunchCap cap = {}, bool split = false);
// Blocks and threads per block of the ncclAllReduceDdaFabric launch.
rcclDdaLaunchCap ncclAllReduceDdaFabricGrid(ncclComm* comm, size_t count, ncclDataType_t datatype,
                                            rcclDdaLaunchCap cap = {});

#endif
