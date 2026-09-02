/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "nccl.h"

#include <cstddef>

struct ncclComm;

constexpr size_t RCCL_DIRECT_A2A_DEFAULT_ONESHOT_THRESHOLD_BYTES = 1ULL << 16;
constexpr size_t RCCL_DIRECT_A2A_TWO_RANK_MAX_BYTES = 1ULL << 22;
constexpr size_t RCCL_DIRECT_A2A_MAX_BYTES = 1ULL << 24;
constexpr int RCCL_DIRECT_A2A_MIN_RANKS = 2;
constexpr int RCCL_DIRECT_A2A_MAX_RANKS = 4;

// Initialize the per-communicator receive staging buffer when the strictly
// gated gfx1151 direct-A2A path is enabled.
ncclResult_t rcclDirectA2aAllReduceCommInit(struct ncclComm* comm);
ncclResult_t rcclDirectA2aAllReduceCommFini(struct ncclComm* comm);

// Checks hardware/topology, operation and resource requirements. Callers must
// separately reject grouped, implicit-order and graph-captured launches.
bool rcclDirectA2aAllReduceEligible(struct ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op);

// Two-rank AllReduce uses one-shot through its rank-specific maximum. Other
// supported rank counts use one-shot for small messages and two-shot
// ReduceScatter + AllGather up to 16 MiB by default (override with
// RCCL_DIRECT_A2A_MAX_BYTES, capped at 32 MiB).
ncclResult_t rcclDirectA2aAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t op, struct ncclComm* comm, cudaStream_t stream);
