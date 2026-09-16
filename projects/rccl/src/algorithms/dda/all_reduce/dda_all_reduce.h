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

// Single source of truth for "is nRanks a supported DDA IPC participant count".
// Exactly kDdaNranks always; any 2..kDdaNranks when RCCL_DDA_NRANKS_RELAX=1.
// The comm-init gate and the per-collective eligibility gate must agree on this,
// otherwise comm init allocates IPC resources the eligibility gate then refuses.
bool ncclDdaIpcNranksSupported(int nRanks);

// True when nRanks sits in the range where ncclDdaIpcNranksSupported()'s answer
// actually depends on RCCL_DDA_NRANKS_RELAX: 2..kDdaNranks-1. Outside that range
// (exactly kDdaNranks, or too few/many ranks) the supported/unsupported answer
// is the same whether or not the knob is set, so a per-rank mismatch in the env
// var cannot cause ranks to disagree on whether to enter the DDA IPC path. Comm
// init uses this to decide whether checking RCCL_DDA_NRANKS_RELAX agreement
// across ranks is even relevant for a given communicator.
bool ncclDdaNranksRelaxConsensusMatters(int nRanks);

// IPC path (single node, kDdaNranks ranks by default; any 2..kDdaNranks when relax is set).
bool ncclAllReduceDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype, ncclRedOp_t op);

// Single source of "can the IPC tree (two-shot) kernel take this count/nRanks/
// typeSize combination": count must divide evenly across nRanks, and the
// resulting per-rank slice must itself be 16-byte aligned, since the tree
// kernel does 16-byte (uint4) vectorized loads over that slice. Shared by
// ncclAllReduceDdaIpcEligible() (the caller-facing gate) and the IPC launch
// path (which is templated on a compile-time NRANKS and so re-derives this
// independently of the gate rather than trusting it) -- kept as one function
// so the two cannot drift apart. Declared here so unit tests can exercise it
// without a GPU; hidden in Release by -fvisibility=hidden, as the rest of the
// internal surface is.
bool ncclAllReduceDdaIpcTreeEligible(size_t count, int nRanks, size_t typeSize);

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

// The LL128 tiers hang off the same gate. Their slot geometry is derived from
// the scratch bank rather than a fixed constant, so unlike the LL tiers their
// size cap moves with comm->ddaScratchBytes -- which is what the unit tests
// exercise by shrinking the scratch to a known number of slices.
bool ddaLL128ArOneShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op);

bool ddaLL128ArTwoShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
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
