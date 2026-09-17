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

// Shared limits and dependency-free boundary helpers for classic and CAST
// NET/IB RMA. Keep these here so host tests exercise the exact production math.
#ifndef NCCL_RMA_MAX_SEGMENTS
#define NCCL_RMA_MAX_SEGMENTS 16
#endif

// 32 WRs cover aligned 4/8-GPU 8 GiB windows; 16x8 GiB needs 48. Fail closed past 64.
#define NCCL_RMA_MAX_DATA_WRS (4 * NCCL_RMA_MAX_SEGMENTS)
#define NCCL_RMA_MAX_SIGNAL_WRS (NCCL_RMA_MAX_DATA_WRS + 1)
#define NCCL_RMA_MAX_FLUSH_WRS NCCL_RMA_MAX_SEGMENTS
// Equals NET_IB_MAX_REQUESTS (NCCL_NET_MAX_REQUESTS * NCCL_NET_IB_MAX_RECVS).
// common_cast.h static_asserts the production macros against this.
#define NCCL_RMA_MAX_INFLIGHT_REQUESTS 256
#define NCCL_RMA_MAX_SEND_QP_WRS (2 * NCCL_RMA_MAX_INFLIGHT_REQUESTS + NCCL_RMA_MAX_SIGNAL_WRS - 2)
#define NCCL_RMA_MAX_FLUSH_QP_WRS (NCCL_RMA_MAX_INFLIGHT_REQUESTS + NCCL_RMA_MAX_FLUSH_WRS - 1)

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

static inline int ncclRmaWrIsSignaled(int wrIndex, int nWrs) {
  return nWrs > 0 && wrIndex == nWrs - 1;
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

// True when ibv_post_send accepted a prefix that did not include the signaled last WR.
static inline int ncclRmaPrefixPostLostSignaledTail(int posted, int nWr) {
  return posted > 0 && posted < nWr;
}

static inline int ncclRmaWrCreditsAvailable(int outstanding, int requested, int capacity) {
  return outstanding >= 0 && requested >= 0 && requested <= capacity && outstanding <= capacity - requested;
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

// Compact have/status recv: heap registrations, else the unused 64-record stack.
static inline void* ncclRmaCompactConsensusRecv(void* heapRegs, void* stackRegs, size_t stackBytes, int nranks,
                                                size_t elemBytes) {
  if (heapRegs != NULL) return heapRegs;
  if (stackRegs == NULL || nranks < 1 || elemBytes == 0) return NULL;
  if ((size_t)nranks > stackBytes / elemBytes) return NULL;
  return stackRegs;
}

// Registration stores MRs/rkeys in recvComm device-slot order. Posts use a
// peer QP whose remDevIdx/devIndex may name a different physical HCA.
// Translate physical ibDevN to that registration slot. Unused slots are -1.
static inline int ncclRmaDevSlotOf(const int* ibDevNs, int rank, int ibDevN, int maxDevsPerNic) {
  if (ibDevNs == NULL || ibDevN < 0 || rank < 0 || maxDevsPerNic < 1) return -1;
  const int* slots = ibDevNs + (size_t)rank * (size_t)maxDevsPerNic;
  for (int d = 0; d < maxDevsPerNic; d++) {
    if (slots[d] == ibDevN) return d;
  }
  return -1;
}

// Data WRs look up the remote QP's physical device. Flush RDMA_READs look up
// the local QP's physical device in the same remote handle.
static inline int ncclRmaRkeyIbDevN(int localIbDevN, int remoteIbDevN, int flushMode) {
  return flushMode ? localIbDevN : remoteIbDevN;
}

#endif // NCCL_NET_IB_RMA_MULTISEG_H_
