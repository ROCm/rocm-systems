/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_NET_IB_GIN_H_
#define _NCCL_NET_IB_GIN_H_

#include <stddef.h>
#include <stdint.h>
#include <limits.h>
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

static_assert(NCCL_RMA_MAX_DATA_WRS == 4 * NCCL_RMA_MAX_SEGMENTS,
              "data WR budget must cover boundary splits plus UINT32_MAX SGE splits");
static_assert(NCCL_RMA_MAX_SIGNAL_WRS == NCCL_RMA_MAX_DATA_WRS + 1,
              "signal WR budget must cover the data chain plus the atomic");
static_assert(NCCL_RMA_MAX_FLUSH_WRS == NCCL_RMA_MAX_SEGMENTS,
              "flush WR budget must cover one RDMA_READ per physical segment");

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

// Count WRs for explicit local/remote segOff tables. Returns maxWr+1 if the chain does not fit.
static inline int ncclRmaSegIndexOf(const size_t* segOff, int nSeg, uint64_t off) {
  for (int s = 0; s < nSeg; s++) {
    if (off < segOff[s + 1]) return s;
  }
  return nSeg - 1;
}

static inline int ncclRmaCountLayoutDataWrs(const size_t* localOff, int nLocal, const size_t* remoteOff, int nRemote,
                                            uint64_t lOff, uint64_t rOff, size_t size, int maxWr) {
  int n = 0;
  size_t rem = size;
  while (rem > 0) {
    if (ncclRmaDataWrBudgetFull(n, maxWr)) return maxWr + 1;
    int ls = ncclRmaSegIndexOf(localOff, nLocal, lOff);
    int rs = ncclRmaSegIndexOf(remoteOff, nRemote, rOff);
    size_t chunk = ncclRmaSegmentSliceBytes(rem, localOff[ls + 1] - lOff, remoteOff[rs + 1] - rOff);
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

// Compact have/status recv: heap registrations, else the unused 64-record stack.
static inline void* ncclRmaCompactConsensusRecv(void* heapRegs, void* stackRegs, size_t stackBytes, int nranks,
                                                size_t elemBytes) {
  if (heapRegs != NULL) return heapRegs;
  if (stackRegs == NULL || nranks < 1 || elemBytes == 0) return NULL;
  if ((size_t)nranks > stackBytes / elemBytes) return NULL;
  return stackRegs;
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

struct ncclGinIbCollComm {
  void* ctx;
  int rank;
  int nranks;
  int connectionId;
  int nConnections;
  int queueDepth;
  void* recvComm;
  void* sendComm;
  void** fullRecvComm;
  void** fullSendComm;
  int dev;
  void* ginCtx;
  struct {
    struct ibv_context* context;
    struct ibv_pd* pd;
  } ib;
  ncclResult_t (*getProperties)(int dev, void* props);
  ncclResult_t (*allGather)(struct ncclGinIbCollComm* cComm, void* srcBuf, void* recvBuf, size_t len);
  ncclResult_t (*allToAll)(struct ncclGinIbCollComm* cComm, void* srcBuf, void* recvBuf, size_t len);
  ncclResult_t (*getGidIndex)(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                              int* gidIndex);
};

#endif
