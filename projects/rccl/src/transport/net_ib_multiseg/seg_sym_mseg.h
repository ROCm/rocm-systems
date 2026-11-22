/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

#ifndef RCCL_TRANSPORT_MSEG_SYM_MSEG_H_
#define RCCL_TRANSPORT_MSEG_SYM_MSEG_H_

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "alloc.h"
#include "debug.h"
#include "nccl.h"

// RCCL: per-rank layout of a symmetric multi-segment registration, shared by every backend that splits operations at
// segment boundaries. keys holds nKeys opaque values per (rank, segment), e.g. one rkey per NIC port.
struct rcclIbSegSymTable {
  size_t* rankOff; // [rank * (nSeg + 1) + s]: each rank's own offsets, starting at 0
  uintptr_t* vas;  // [rank * nSeg + s]
  uint64_t* keys;  // [(rank * nSeg + s) * nKeys + k]
};

// Consensus record per rank; the backend keeps nranks of these allocated before registration starts.
struct rcclIbSegSymRecord {
  ncclResult_t status;
  int nSegments;
};

typedef ncclResult_t (*rcclIbSegSymAllGatherFn)(void* ctx, void* src, void* dst, size_t bytes);

static inline uintptr_t rcclIbSegSymVa(const struct rcclIbSegSymTable* t, int nSeg, int rank, int seg) {
  return t->vas[(size_t)rank * nSeg + (size_t)seg];
}

static inline uint64_t rcclIbSegSymKey(const struct rcclIbSegSymTable* t, int nSeg, int nKeys, int rank, int seg,
                                       int k) {
  return t->keys[((size_t)rank * nSeg + (size_t)seg) * nKeys + (size_t)k];
}

static inline void rcclIbSegSymTableFree(struct rcclIbSegSymTable* t) {
  free(t->rankOff);
  free(t->vas);
  free(t->keys);
  t->rankOff = NULL;
  t->vas = NULL;
  t->keys = NULL;
}

// Gathers one status per rank; a failure anywhere fails every rank with the first failing rank's error.
static inline ncclResult_t rcclIbSegSymAgree(rcclIbSegSymAllGatherFn allGather, void* ctx, int nranks,
                                             struct rcclIbSegSymRecord* records, ncclResult_t status, int nSeg,
                                             const char* who, bool warn) {
  struct rcclIbSegSymRecord local = {status, nSeg};
  NCCLCHECK(allGather(ctx, &local, records, sizeof(local)));
  for (int r = 0; r < nranks; r++) {
    if (records[r].status != ncclSuccess) {
      if (warn) {
        WARN("%s: symmetric registration rejected because rank %d failed with error %d", who, r, records[r].status);
      }
      return records[r].status;
    }
  }
  return ncclSuccess;
}

// RCCL: every rank calls this after its local walk, even after a local failure, so all ranks fail together.
// Builds t from each rank's offsets, segment starts and nKeys keys per segment; t is freed by the caller.
static inline ncclResult_t rcclIbSegSymExchange(rcclIbSegSymAllGatherFn allGather, void* ctx, int nranks,
                                                struct rcclIbSegSymRecord* records, ncclResult_t localStatus,
                                                int nSeg, int nKeys, size_t* localOff, uintptr_t* localVas,
                                                uint64_t* localKeys, const void* data, const char* who,
                                                struct rcclIbSegSymTable* t) {
  NCCLCHECK(rcclIbSegSymAgree(allGather, ctx, nranks, records, localStatus, nSeg, who, true));
  // The offset, VA and key gathers below use one nSeg. Terminal sizes may differ (asymmetric windows).
  for (int r = 0; r < nranks; r++) {
    if (records[r].nSegments != nSeg) {
      WARN("%s: buffer %p segment count differs on rank %d (%d vs %d)", who, data, r, records[r].nSegments, nSeg);
      return ncclInvalidUsage;
    }
  }

  // Reconcile allocation failures too, before entering the offset, VA and key gathers.
  ncclResult_t ret = ncclCalloc(&t->rankOff, (size_t)nranks * ((size_t)nSeg + 1));
  if (ret == ncclSuccess) ret = ncclCalloc(&t->vas, (size_t)nranks * nSeg);
  if (ret == ncclSuccess) ret = ncclCalloc(&t->keys, (size_t)nranks * nSeg * nKeys);
  NCCLCHECK(rcclIbSegSymAgree(allGather, ctx, nranks, records, ret, nSeg, who, false));

  // Every rank checks every table, so a bad table fails all ranks together.
  NCCLCHECK(allGather(ctx, localOff, t->rankOff, sizeof(size_t) * (nSeg + 1)));
  for (int r = 0; r < nranks; r++) {
    const size_t* off = t->rankOff + (size_t)r * (nSeg + 1);
    bool ordered = off[0] == 0;
    for (int s = 0; ordered && s < nSeg; s++) ordered = off[s + 1] >= off[s];
    if (!ordered) {
      WARN("%s: buffer %p segment offsets from rank %d are not ordered from 0", who, data, r);
      return ncclInvalidUsage;
    }
  }
  NCCLCHECK(allGather(ctx, localVas, t->vas, sizeof(uintptr_t) * nSeg));
  NCCLCHECK(allGather(ctx, localKeys, t->keys, sizeof(uint64_t) * nSeg * nKeys));
  return ncclSuccess;
}

#endif // RCCL_TRANSPORT_MSEG_SYM_MSEG_H_
