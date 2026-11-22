/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef RCCL_TRANSPORT_NET_IB_IB_MULTISEG_MSEG_H_
#define RCCL_TRANSPORT_NET_IB_IB_MULTISEG_MSEG_H_

#include <stdint.h>

#include "nccl.h"
#include "seg_rma_mseg.h" // RCCL: shared multi-segment limits; this header keeps the classic-IB-only parts.

// a freed request returns its sequence id only if no later request took one.
static inline void rcclRmaReleaseSeqId(uint64_t id, uint64_t* postedSeq) {
  if (id == *postedSeq) (*postedSeq)--;
}

struct IbMsegRequest;
ncclResult_t rcclRmaIbProxyConnectFinish(void** collComm, int nranks);
ncclResult_t rcclRmaIbProxyRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags,
                                    void** mhandle);
ncclResult_t rcclRmaIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                          uint64_t mr_flags, void** mhandle);
ncclResult_t rcclRmaIbProxyDeregMrSym(void* collComm, void* mhandle);
ncclResult_t rcclRmaIbProxyIPut(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                uint64_t dstOff, void* dstMhandle, uint32_t rank, uint32_t optFlags, void** request);
ncclResult_t rcclRmaIbProxyIGet(void* rmaCtx, int context, uint64_t remoteOffset, void* remoteMhandle, size_t size,
                                uint64_t localOffset, void* localMhandle, uint32_t rank, uint32_t optFlags,
                                void** request);
ncclResult_t rcclRmaIbProxyIPutSignal(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                      uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff,
                                      void* signalMhandle, uint64_t signalValue, uint32_t signalOp, bool isStrongSignal,
                                      uint32_t optFlags, void** request);
ncclResult_t rcclRmaIbProxyIFlush(void* rmaCtx, int context, void* mhandle, uint32_t rank, void** request);
ncclResult_t rcclRmaIbProxyTestFailed(struct IbMsegRequest* req, int* done);

int rcclRmaIbProxyMrSegmentCount(void* mhandle);

// RCCL: segmented-builder counters per send comm, for tests. laterSegPosts counts posts that addressed a segment
// past the first on either side, which only a working segment walk can do.
struct rcclIbSegStats {
  uint64_t posts;
  uint64_t wrs;
  uint64_t laterSegPosts;
};
ncclResult_t rcclIbSendCommSegStats(void* sendComm, struct rcclIbSegStats* out);
ncclResult_t rcclRmaIbProxySegStats(void* rmaCtx, int context, int rank, struct rcclIbSegStats* out);

// RCCL: one connect-time decision. multiSegOn is the core's fd -1 gate, so the CTS follows registration; GIN has no
// CTS. Resiliency replays a whole slot, which a send resumed across tests cannot do yet.
static inline bool IbMsegUseMultiSeg(bool multiSegOn, bool forGin, bool resilient) {
  return multiSegOn && !forGin && !resilient;
}

#endif // RCCL_TRANSPORT_NET_IB_IB_MULTISEG_MSEG_H_
