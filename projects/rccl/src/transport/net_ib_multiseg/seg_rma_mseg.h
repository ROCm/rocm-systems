/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

#ifndef RCCL_TRANSPORT_MSEG_RMA_MSEG_H_
#define RCCL_TRANSPORT_MSEG_RMA_MSEG_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "nccl.h"
#include "seg_mseg.h"
#include "transport/rccl_mseg_rma.h" // RCCL: the segment-table bound and the in-flight cap shared with the core.

// RCCL: multi-segment RMA limits and helpers for this plugin's RMA backend.

// One round of a multi-segment op: up to RCCL_RMA_MAX_DATA_WRS data WRs, plus the signal on the last round. A wider
// op posts its next round after this one completes, so these do not depend on the segment count.
#define RCCL_RMA_MAX_DATA_WRS 64
#define RCCL_RMA_MAX_SIGNAL_WRS (RCCL_RMA_MAX_DATA_WRS + 1)
// Live request ids per IB comm. The plugin static_asserts it equals NET_IB_MAX_REQUESTS.
#define RCCL_RMA_MAX_REQUESTS 256
// Sender QP depth with RMA multi-segment on: every in-flight request may have one full round posted.
#define RCCL_RMA_SEND_QP_WRS (RCCL_RMA_MAX_INFLIGHT_REQUESTS * RCCL_RMA_MAX_SIGNAL_WRS)

// RCCL: the RMA offset table is the shared segment table. size_t and uint64_t match on the platforms we build.
static_assert(sizeof(size_t) == sizeof(uint64_t), "segment offsets are stored as size_t and read as uint64_t");

static inline int rcclRmaSegIndexOf(const size_t* segOff, int nSeg, uint64_t off) {
  return rcclIbSegIndexOf((const uint64_t*)segOff, nSeg, off);
}

static inline int rcclRmaWrIsSignaled(int wrIndex, int nWrs) {
  return nWrs > 0 && wrIndex == nWrs - 1;
}

// true when ibv_post_send accepted a prefix that did not include the signaled last WR.
static inline int rcclRmaPrefixPostLostSignaledTail(int posted, int nWr) {
  return posted > 0 && posted < nWr;
}

static inline int rcclRmaSignalOffsetValid(size_t signalOff, size_t segmentEnd) {
  return (signalOff & (sizeof(uint64_t) - 1)) == 0 && signalOff <= segmentEnd &&
         sizeof(uint64_t) <= segmentEnd - signalOff;
}

// per-rank segOff table from registration allgather, nSegments + 1 entries per rank; falls back to the local map.
static inline const size_t* rcclRmaPeerSegOff(const size_t* rankSegOff, const size_t* localSegOff, int rank,
                                              int nSegments) {
  if (rankSegOff == NULL || rank < 0 || nSegments < 1) return localSegOff;
  return rankSegOff + (size_t)rank * ((size_t)nSegments + 1);
}

// WRs the HCA accepted from a next-linked chain of nWr when ibv_post_send fails at badWr (NULL: the whole chain).
static inline int rcclRmaPostedWrCount(const void* wr, int nWr, const void* badWr, size_t nextOffset) {
  int posted = 0;
  const char* cur = (const char*)wr;
  while (cur != NULL && posted < nWr) {
    if (cur == (const char*)badWr) break;
    posted++;
    cur = *(char* const*)(cur + nextOffset);
  }
  return posted;
}

// Keep-or-free after ibv_post_send: free the slot only if nothing posted, and mark FAILED if the signaled tail
// was lost or the doorbell failed. After a prefix post the request is kept and success returned so Test() drains.
static inline ncclResult_t rcclRmaCompletePostedRequest(ncclResult_t postRet, int posted, int nWr, int* keepRequest,
                                                        int* markFailed) {
  if (keepRequest) *keepRequest = 0;
  if (markFailed) *markFailed = 0;
  if (postRet != ncclSuccess && posted == 0) return postRet;
  if (keepRequest) *keepRequest = 1;
  // A failed batch doorbell stops at or before the chain's signaled tail, even when posted == nWr.
  if (markFailed) *markFailed = postRet != ncclSuccess || rcclRmaPrefixPostLostSignaledTail(posted, nWr);
  return ncclSuccess;
}

// RCCL: per-segment MRs follow the core fd -1 gate (switch and DMA-BUF), as comm->rcclMultiSegRma does.
bool rcclMultiSegOn(bool pluginRegistersPerSegment);
static inline bool rcclRmaMultiSegEnabled() { return rcclMultiSegOn(/*pluginRegistersPerSegment=*/true); }

// First field of the multi-segment MR. The single-MR handle does not carry it.
static inline uint32_t rcclRmaSegMrTag(void) { return 0xC3C3A5A5u; }

// Segment count, or 0 when the handle is empty or is the single-MR layout.
static inline int rcclRmaMrSegmentCount(const void* mhandle) {
  if (mhandle == NULL) return 0;
  uint32_t tag = 0;
  int nSegments = 0;
  memcpy(&tag, mhandle, sizeof(tag));
  memcpy(&nSegments, (const char*)mhandle + sizeof(tag), sizeof(nSegments));
  if (tag != rcclRmaSegMrTag()) return 0;
  return nSegments;
}

#endif // RCCL_TRANSPORT_MSEG_RMA_MSEG_H_
