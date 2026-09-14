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

// A paired data transfer can split at every local and remote boundary. A
// segment slice may split once more at the verbs 32-bit SGE length limit; the
// fixed budget deliberately rejects larger chains before posting.
#define NCCL_RMA_MAX_DATA_WRS (2 * NCCL_RMA_MAX_SEGMENTS)
#define NCCL_RMA_MAX_SIGNAL_WRS (NCCL_RMA_MAX_DATA_WRS + 1)
#define NCCL_RMA_MAX_FLUSH_WRS NCCL_RMA_MAX_SEGMENTS

static_assert(NCCL_RMA_MAX_DATA_WRS == 2 * NCCL_RMA_MAX_SEGMENTS,
              "data WR budget must cover a split on every local and remote boundary");
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

static inline int ncclRmaSignalOffsetValid(size_t signalOff, size_t segmentEnd) {
  return (signalOff & (sizeof(uint64_t) - 1)) == 0 && signalOff <= segmentEnd &&
         sizeof(uint64_t) <= segmentEnd - signalOff;
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

struct ncclIbMrHandle;
struct ncclRmaIbProxyMrHandle {
  int nSegments;
  // segOff[0]==0, segOff[nSegments]==size; per-segment local MRs.
  // base_vas indexed [rank*nSegments + seg].
  // rkeys indexed (rank * nSegments + seg) * NCCL_IB_MAX_DEVS_PER_NIC + remDevIdx
  size_t segOff[NCCL_RMA_MAX_SEGMENTS + 1];
  struct ncclIbMrHandle* mrHandle[NCCL_RMA_MAX_SEGMENTS];
  uintptr_t* base_vas;
  uint32_t* rkeys;
};

static inline int ncclRmaHandleNSegments(const void* mhandle) {
  const struct ncclRmaIbProxyMrHandle* h = (const struct ncclRmaIbProxyMrHandle*)mhandle;
  return h != NULL ? h->nSegments : 0;
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
