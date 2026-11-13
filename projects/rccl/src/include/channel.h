/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CHANNEL_H_
#define NCCL_CHANNEL_H_
#include "comm.h"
#include "utils.h"
#include "param.h"
#include "archinfo.h"

bool rcclUseAinic();
int rcclEffectiveP2pBatchEnable(struct ncclComm* comm);
int rcclP2pTaskParts(struct ncclComm* comm, ncclFunc_t collAPI);

RCCL_PARAM_DECLARE(
  PxnOptQpUsage); // RCCL_PXN_OPT_QP_USAGE: uses batch stride of comm->maxLocalRanks instead of 1 to reduce QP usage when p2p-batching is disabled

#include <algorithm>

ncclResult_t initChannel(struct ncclComm* comm, int channelid);
ncclResult_t initNvlsChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share);
ncclResult_t initCollnetChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share);
ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks, int collnetNRanks, int nvlsNRanks,
                         struct ncclComm* comm);

inline uint8_t ncclP2pChannelBaseForRound(struct ncclComm* comm, int p2pRound, int p2pBatchEnable = 0) {
  int base;
  if (comm->nNodes > 1) {
    int nodeDelta = p2pRound / comm->maxLocalRanks;
    int localDelta = p2pRound % comm->maxLocalRanks;
    int fallbackBatch = (!ncclPxnDisable(comm) && rcclParamPxnOptQpUsage() && rcclUseAinic()) ? comm->maxLocalRanks : 1;
    int batchSize = (comm->nNodes > 2 && p2pBatchEnable) ? NCCL_MAX_DEV_WORK_P2P_PER_BATCH : fallbackBatch;
    base = nodeDelta * divUp(comm->maxLocalRanks, batchSize);
    base += localDelta / batchSize;
  } else {
    base = p2pRound;
  }
  return base & 0xff;
}

// True when the single-node MLOPart halving loop in paths.cc shrank the per-peer count.
inline bool rcclP2pHalvedSingleNode(struct ncclComm* comm) {
  return comm->nNodes == 1 && comm->hasMloPart && comm->p2pnChannelsPerPeerBase > comm->p2pnChannelsPerPeer;
}

// Single-node MLOPart gather/scatter keep the unhalved per-peer count (the halving
// loop in paths.cc sizes alltoall, and was also cutting gather to 1 part per peer).
inline bool rcclP2pPartitionedGather(struct ncclComm* comm, ncclFunc_t collAPI) {
  return rcclP2pHalvedSingleNode(comm) && (collAPI == ncclFuncGather || collAPI == ncclFuncScatter);
}

// Parts (channels per peer) one collective places. The device recovers a part with
// p2pnChannelsPerPeer as the stride, so this only changes how many parts are used.
inline int rcclP2pPartCount(struct ncclComm* comm, ncclFunc_t collAPI) {
  return rcclP2pPartitionedGather(comm, collAPI) ? comm->p2pnChannelsPerPeerBase : comm->p2pnChannelsPerPeer;
}

// Multiplier on the round base. Channel = base * stride + part, so scaling the base by
// parts/stride gives each round its own block of parts (gather: round*8 + part).
inline int rcclP2pBaseScale(struct ncclComm* comm, ncclFunc_t collAPI) {
  int stride = comm->p2pnChannelsPerPeer > 0 ? comm->p2pnChannelsPerPeer : 1;
  return rcclP2pPartCount(comm, collAPI) / stride;
}

inline bool rcclP2pBatchEligible(int enabled, ssize_t sendBytes, ssize_t recvBytes, ssize_t threshold) {
  return enabled && sendBytes == recvBytes && sendBytes <= threshold && recvBytes <= threshold;
}

// Fusion size cutoff. A negative threshold, including an explicit
// RCCL_P2P_BATCH_THRESHOLD=-1, means "arch default": the parameter default is
// -1, and the wrapper passes that value through with no separate "was set" bit.
// Arch default is 64 KiB on gfx950 at 16+ nodes and no cutoff elsewhere, so an
// explicit enable stays fused. A non-negative threshold is the cutoff on every
// architecture. The channel map still follows the batching flag, so rounds that
// miss the cutoff share a channel and run one after another.
inline ssize_t rcclP2pBatchCutoff(const char* gcn, int nNodes, ssize_t sendBytes, ssize_t recvBytes,
                                  ssize_t threshold) {
  if (threshold < 0) {
    if (IsArchMatch(gcn, "gfx950") && nNodes >= 16) return 1 << 16; // 64k per-rank message size
    return std::max(sendBytes, recvBytes);
  } else {
    return threshold;
  }
}

#endif
