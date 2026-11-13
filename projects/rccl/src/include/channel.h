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
#include "transport.h"

bool rcclUseAinic();
int rcclEffectiveP2pBatchEnable(struct ncclComm* comm);
int rcclP2pTaskParts(struct ncclComm* comm, ncclFunc_t collAPI);

// Marks the connIndex connectors of the first nParts channels of a peer's P2P
// mapping for preconnect. Returns whether any connector still needs setup.
inline bool rcclP2pMarkPreconnectChannels(struct ncclComm* comm, int peer, bool isSendNotRecv, uint8_t base,
                                          int nP2pChannels, int nP2pChannelsPerPeer, int nParts, int connIndex) {
  bool marked = false;
  base %= nP2pChannels;
  for (int c = 0; c < nParts; c++) {
    int channelId = ncclP2pChannelForPart(nP2pChannels, base, c, nP2pChannelsPerPeer, comm->nNodes,
                                          comm->p2pChannelShiftSize);
    struct ncclConnector* conn = isSendNotRecv ? &comm->channels[channelId].peers[peer]->send[connIndex]
                                               : &comm->channels[channelId].peers[peer]->recv[connIndex];
    if (conn->hasSeen != 0) continue;
    // P2P uses only one connector. hasSeen also prevents duplicate setup
    // when split shared communicators enqueue the same connection.
    conn->hasSeen = 1;
    if (connIndex == RCCL_CONN_IDX_P2P_SHM && conn->connected) continue;
    if (connIndex != NCCL_CONN_IDX_P2P_NET) conn->p2pOnly = 1;
    int maskPeer = peer + CHANNEL_MASK_OFFSET(comm->nRanks, connIndex);
    struct channelMasks* connect = isSendNotRecv ? &comm->connectSend[maskPeer] : &comm->connectRecv[maskPeer];
    connect->masks[channelId / 64] |= (1ULL << (channelId % 64));
    marked = true;
  }
  return marked;
}

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
