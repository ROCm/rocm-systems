/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_NET_IB_RMA_MULTISEG_H_
#define NCCL_NET_IB_RMA_MULTISEG_H_

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include "nccl.h"

// Cap on physical segments per GIN/RMA symmetric buffer. HIP dma-buf export
// describes only the first physical segment, so registration allocates one MR
// per segment up to this limit.
#ifndef NCCL_RMA_MAX_SEGMENTS
#define NCCL_RMA_MAX_SEGMENTS 16
#endif

// 32 WRs cover aligned 4/8-GPU 8 GiB windows; 16x8 GiB needs 48. Fail closed past 64.
#define NCCL_RMA_MAX_DATA_WRS (4 * NCCL_RMA_MAX_SEGMENTS)
#define NCCL_RMA_MAX_SIGNAL_WRS (NCCL_RMA_MAX_DATA_WRS + 1)
#define NCCL_RMA_MAX_FLUSH_WRS NCCL_RMA_MAX_SEGMENTS
// Multi-segment sender QP depth, NCCL_IB_MAX_SEND_WRS, spelled without common.h; connect.cc asserts they match.
#define NCCL_RMA_SEND_QP_WRS (2 * 256 + 8 * 2 * NCCL_RMA_MAX_SEGMENTS + 1)

static inline size_t ncclRmaSegmentSliceBytes(size_t remaining, size_t localRemaining, size_t remoteRemaining) {
  size_t chunk = remaining;
  if (localRemaining < chunk) chunk = localRemaining;
  if (remoteRemaining < chunk) chunk = remoteRemaining;
  if ((size_t)UINT32_MAX < chunk) chunk = (size_t)UINT32_MAX;
  return chunk;
}

static inline int ncclRmaDataWrBudgetFull(int n, int maxWr) {
  return n >= maxWr;
}

// Paired data WRs to move `size` inside one local and one remote segment
// (UINT32_MAX SGE splits only). Returns maxWr+1 if the chain does not fit.
static inline int ncclRmaCountPairedDataWrs(size_t size, int maxWr) {
  int n = 0;
  size_t rem = size;
  while (rem > 0) {
    if (ncclRmaDataWrBudgetFull(n, maxWr)) {
      return maxWr + 1;
    }
    rem -= ncclRmaSegmentSliceBytes(rem, rem, rem);
    n++;
  }
  return n;
}

static inline int ncclRmaSegIndexOf(const size_t* segOff, int nSeg, uint64_t off) {
  for (int s = 0; s < nSeg; s++) {
    if (off < segOff[s + 1]) return s;
  }
  return nSeg - 1;
}

// Bytes of the next WR at (lOff, rOff) and the segments it lands in; the WR builder and the counter share it.
// Flush mode (localOff == NULL) takes the rest of the remote segment. Returns 0 past the end of either table.
static inline size_t ncclRmaNextSlice(const size_t* localOff, int nLocal, const size_t* remoteOff, int nRemote,
                                      uint64_t lOff, uint64_t rOff, size_t rem, int* ls, int* rs) {
  *ls = 0;
  *rs = ncclRmaSegIndexOf(remoteOff, nRemote, rOff);
  if (rOff >= remoteOff[*rs + 1]) return 0;
  size_t remoteRemaining = remoteOff[*rs + 1] - rOff;
  if (localOff == NULL) return rem < remoteRemaining ? rem : remoteRemaining;
  *ls = ncclRmaSegIndexOf(localOff, nLocal, lOff);
  if (lOff >= localOff[*ls + 1]) return 0;
  return ncclRmaSegmentSliceBytes(rem, localOff[*ls + 1] - lOff, remoteRemaining);
}

// Count WRs for explicit local/remote segOff tables. Returns maxWr+1 if the chain does not fit.
static inline int ncclRmaCountLayoutDataWrs(const size_t* localOff, int nLocal, const size_t* remoteOff, int nRemote,
                                            uint64_t lOff, uint64_t rOff, size_t size, int maxWr) {
  int n = 0;
  size_t rem = size;
  while (rem > 0) {
    if (ncclRmaDataWrBudgetFull(n, maxWr)) return maxWr + 1;
    int ls, rs;
    size_t chunk = ncclRmaNextSlice(localOff, nLocal, remoteOff, nRemote, lOff, rOff, rem, &ls, &rs);
    if (chunk == 0) return maxWr + 1;
    lOff += chunk;
    rOff += chunk;
    rem -= chunk;
    n++;
  }
  return n;
}

static inline int ncclRmaWrIsSignaled(int wrIndex, int nWrs) {
  return nWrs > 0 && wrIndex == nWrs - 1;
}

static inline int ncclRmaWrCreditsAvailable(int outstanding, int requested, int capacity) {
  return outstanding >= 0 && requested >= 0 && requested <= capacity && outstanding <= capacity - requested;
}

// Request ids are dense per comm and at most NCCL_RMA_CREDIT_IDS are live, so a chain's WR count
// lives at id % NCCL_RMA_CREDIT_IDS. Ids without an entry are single-MR ops at a fixed cost.
#define NCCL_RMA_CREDIT_IDS 256
struct ncclRmaWrCredits {
  uint64_t releasedSeq;
  int trackedWrs;
  int trackedIds;
  uint8_t wrsById[NCCL_RMA_CREDIT_IDS];
};

static inline void ncclRmaWrCreditsRelease(struct ncclRmaWrCredits* c, uint64_t completedSeq) {
  for (; c->releasedSeq < completedSeq; c->releasedSeq++) {
    uint8_t* wrs = &c->wrsById[(c->releasedSeq + 1) % NCCL_RMA_CREDIT_IDS];
    if (*wrs == 0) continue;
    c->trackedWrs -= *wrs;
    c->trackedIds--;
    *wrs = 0;
  }
}

static inline int ncclRmaWrCreditsInUse(const struct ncclRmaWrCredits* c, uint64_t postedSeq, int untrackedCost) {
  uint64_t live = postedSeq > c->releasedSeq ? postedSeq - c->releasedSeq : 0;
  return c->trackedWrs + ((int)live - c->trackedIds) * untrackedCost;
}

static inline void ncclRmaWrCreditsRecord(struct ncclRmaWrCredits* c, uint64_t id, int nWrs) {
  if (id <= c->releasedSeq || nWrs <= 0) return;
  uint8_t* wrs = &c->wrsById[id % NCCL_RMA_CREDIT_IDS];
  if (*wrs == 0) c->trackedIds++;
  c->trackedWrs += nWrs - *wrs;
  *wrs = (uint8_t)nWrs;
}
// True when ibv_post_send accepted a prefix that did not include the signaled last WR.
static inline int ncclRmaPrefixPostLostSignaledTail(int posted, int nWr) {
  return posted > 0 && posted < nWr;
}

static inline int ncclRmaSignalOffsetValid(size_t signalOff, size_t segmentEnd) {
  return (signalOff & (sizeof(uint64_t) - 1)) == 0 && signalOff <= segmentEnd &&
         sizeof(uint64_t) <= segmentEnd - signalOff;
}

// Equal nSegments in [1, NCCL_RMA_MAX_SEGMENTS]. Terminal sizes may differ.
static inline int ncclRmaSegmentCountsMatch(int lhsSegments, int rhsSegments) {
  return lhsSegments == rhsSegments && lhsSegments >= 1 && lhsSegments <= NCCL_RMA_MAX_SEGMENTS;
}

static inline int ncclRmaLayoutsMatch(int lhsSegments, const size_t* lhsOffsets, int rhsSegments,
                                      const size_t* rhsOffsets) {
  if (!ncclRmaSegmentCountsMatch(lhsSegments, rhsSegments)) return 0;
  for (int s = 0; s <= lhsSegments; s++) {
    if (lhsOffsets[s] != rhsOffsets[s]) return 0;
  }
  return 1;
}

// Peer segOff tables feed unsigned offset math: they must start at 0 and never decrease.
static inline int ncclRmaSegOffTableValid(const size_t* segOff, int nSegments) {
  if (segOff == NULL || nSegments < 1 || nSegments > NCCL_RMA_MAX_SEGMENTS || segOff[0] != 0) return 0;
  for (int s = 0; s < nSegments; s++) {
    if (segOff[s + 1] < segOff[s]) return 0;
  }
  return 1;
}

// Per-rank segOff table from registration allgather; falls back to the local map.
static inline const size_t* ncclRmaPeerSegOff(const size_t* rankSegOff, const size_t* localSegOff, int rank) {
  if (rankSegOff == NULL || rank < 0) return localSegOff;
  return rankSegOff + (size_t)rank * (NCCL_RMA_MAX_SEGMENTS + 1);
}

// Count WRs the HCA accepted when ibv_post_send fails at badWr. Walk a
// next-linked chain of nWr entries. badWr == NULL counts the whole chain.
static inline int ncclRmaPostedWrCount(const void* wr, int nWr, const void* badWr, size_t nextOffset) {
  int posted = 0;
  const char* cur = (const char*)wr;
  while (cur != NULL && posted < nWr) {
    if (cur == (const char*)badWr) break;
    posted++;
    cur = *(char* const*)(cur + nextOffset);
  }
  return posted;
}

// A failed handle calloc must not memcpy segOff before the status AllGather.
static inline int ncclRmaRegistrationHandleReady(const void* handle, int nSeg) {
  return handle != NULL && nSeg >= 1 && nSeg <= NCCL_RMA_MAX_SEGMENTS;
}

// After a prefix post, keep the request and return success so Test() drains.
// Callers NCCLCHECK the complete helper and never reach test() on error.
static inline ncclResult_t ncclRmaPostedRequestStatus(ncclResult_t postRet, int posted) {
  if (postRet != ncclSuccess && posted > 0) return ncclSuccess;
  return postRet;
}

// Keep-or-free after ibv_post_send. posted==0 and error: free the slot.
// Otherwise keep *request so Test() drains; mark FAILED if the signaled tail
// was lost. gin.cc applies the IB side effects from these flags.
static inline ncclResult_t ncclRmaCompletePostedRequest(ncclResult_t postRet, int posted, int nWr, int* keepRequest,
                                                        int* markFailed) {
  if (keepRequest) *keepRequest = 0;
  if (markFailed) *markFailed = 0;
  if (postRet != ncclSuccess && posted == 0) return postRet;
  if (keepRequest) *keepRequest = 1;
  if (markFailed) *markFailed = ncclRmaPrefixPostLostSignaledTail(posted, nWr);
  return ncclRmaPostedRequestStatus(postRet, posted);
}

// A freed request returns its sequence id only if no later request took one.
static inline void ncclRmaReleaseSeqId(uint64_t id, uint64_t* postedSeq) {
  if (id == *postedSeq) (*postedSeq)--;
}

// Per-segment DMA-BUF MRs for the RMA proxy. NCCL_MULTI_SEGMENT_REGISTER=0 keeps the
// upstream single-MR path; the value must match on every rank.
int64_t ncclParamMultiSegmentRegister();
static inline bool rcclRmaMultiSegEnabled() { return ncclParamMultiSegmentRegister() != 0; }

struct ncclIbRequest;
ncclResult_t rcclRmaIbProxyConnectFinish(void** collComm, int nranks);
ncclResult_t rcclRmaIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                   uint64_t mr_flags, void** mhandle);
ncclResult_t rcclRmaIbProxyDeregMrSym(void* collComm, void* mhandle);
ncclResult_t rcclRmaIbProxyIPut(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                         uint64_t dstOff, void* dstMhandle, uint32_t rank, uint32_t optFlags, void** request);
ncclResult_t rcclRmaIbProxyIGet(void* rmaCtx, int context, uint64_t remoteOffset, void* remoteMhandle, size_t size,
                         uint64_t localOffset, void* localMhandle, uint32_t rank, uint32_t optFlags, void** request);
ncclResult_t rcclRmaIbProxyIPutSignal(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                               uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff,
                               void* signalMhandle, uint64_t signalValue, uint32_t signalOp, bool isStrongSignal,
                               uint32_t optFlags, void** request);
ncclResult_t rcclRmaIbProxyIFlush(void* rmaCtx, int context, void* mhandle, uint32_t rank, void** request);
ncclResult_t rcclRmaIbProxyTestFailed(struct ncclIbRequest* req, int* done);

#endif
