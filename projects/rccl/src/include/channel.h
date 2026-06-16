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

bool rcclUseAinic();

RCCL_PARAM_DECLARE(PxnOptQpUsage);  // RCCL_PXN_OPT_QP_USAGE: uses batch stride of comm->maxLocalRanks instead of 1 to reduce QP usage when p2p-batching is disabled

#include <algorithm>

ncclResult_t initChannel(struct ncclComm* comm, int channelid);
ncclResult_t initNvlsChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share);
ncclResult_t initCollnetChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share);
ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks, int collnetNRanks, int nvlsNRanks, struct ncclComm* comm);

inline uint8_t ncclP2pChannelBaseForRound(struct ncclComm* comm, int p2pRound, int p2pBatchEnable = 0) {
  int base;
  if (comm->nNodes > 1) {
    int nodeDelta = p2pRound/comm->maxLocalRanks;
    int localDelta = p2pRound%comm->maxLocalRanks;
    int fallbackBatch = (!ncclPxnDisable(comm) && rcclParamPxnOptQpUsage() && rcclUseAinic()) ? comm->maxLocalRanks : 1;
    int batchSize = (comm->nNodes > 2 && p2pBatchEnable)
        ? NCCL_MAX_DEV_WORK_P2P_PER_BATCH
        : fallbackBatch;
    base = nodeDelta*divUp(comm->maxLocalRanks, batchSize);
    base += localDelta/batchSize;
  } else {
    base = p2pRound;
  }
  return base & 0xff;
}

#endif
