/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLTOALLV_META_H_
#define NCCL_ALLTOALLV_META_H_

#include <stddef.h>

#include "nccl.h"

// Gathered AlltoAllv metadata layout (bytes):
// per-rank block of [sendSizes, sendDispls, recvSizes, recvDispls] x nRanks.
static inline size_t ncclAlltoAllvMetaBlockOffset(int rank, int nRanks) {
  return (size_t)rank * 4u * (size_t)nRanks;
}

static inline size_t* ncclAlltoAllvRankMetaBlock(size_t* gathered, int rank, int nRanks) {
  return gathered + ncclAlltoAllvMetaBlockOffset(rank, nRanks);
}

static inline size_t* ncclAlltoAllvSendSizes(size_t* gathered, int rank, int nRanks) {
  return ncclAlltoAllvRankMetaBlock(gathered, rank, nRanks);
}

static inline size_t* ncclAlltoAllvSendDispls(size_t* gathered, int rank, int nRanks) {
  return ncclAlltoAllvSendSizes(gathered, rank, nRanks) + nRanks;
}

static inline size_t* ncclAlltoAllvRecvSizes(size_t* gathered, int rank, int nRanks) {
  return ncclAlltoAllvSendDispls(gathered, rank, nRanks) + nRanks;
}

static inline size_t* ncclAlltoAllvRecvDispls(size_t* gathered, int rank, int nRanks) {
  return ncclAlltoAllvRecvSizes(gathered, rank, nRanks) + nRanks;
}

static inline void ncclAlltoAllvPackLocalSizes(size_t* sizes, int nRanks, const size_t* sendcounts,
                                               const size_t* sdispls, const size_t* recvcounts, const size_t* rdispls) {
  for (int i = 0; i < nRanks; ++i) {
    sizes[i] = sendcounts[i];
    sizes[nRanks + i] = sdispls[i];
    sizes[2 * nRanks + i] = recvcounts[i];
    sizes[3 * nRanks + i] = rdispls[i];
  }
}

static inline size_t ncclAlltoAllvTrafficBytes(const size_t* gathered, int rank, int nRanks) {
  size_t bytes = 0;
  const size_t* sendSizes = ncclAlltoAllvSendSizes((size_t*)gathered, rank, nRanks);
  for (int r = 0; r < nRanks; ++r) {
    bytes += sendSizes[r];
  }
  return bytes;
}

// Verdict is a pure function of the gathered matrix - identical on every rank.
static inline ncclResult_t ncclAlltoAllvValidateSizeMatrix(const size_t* gathered, int nRanks) {
  for (int src = 0; src < nRanks; ++src) {
    const size_t* s = ncclAlltoAllvSendSizes((size_t*)gathered, src, nRanks);
    for (int dst = 0; dst < nRanks; ++dst) {
      const size_t* r = ncclAlltoAllvRecvSizes((size_t*)gathered, dst, nRanks);
      if (s[dst] != r[src]) {
        return ncclInvalidUsage;
      }
    }
  }
  return ncclSuccess;
}

#endif /* NCCL_ALLTOALLV_META_H_ */
