/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

#ifndef RCCL_TRANSPORT_MSEG_REG_MSEG_H_
#define RCCL_TRANSPORT_MSEG_REG_MSEG_H_

#include <stddef.h>
#include <stdint.h>

#include "nccl.h"

// RCCL: per-segment register and deregister for IB-MSEG; the walk scratch is one block sized from the count.

struct rcclIbSegRegOps {
  ncclResult_t (*regOne)(void* ctx, int index, void** mr);
  ncclResult_t (*deregOne)(void* ctx, int index, void* mr);
};

// Skips NULL slots, clears each slot it releases, and returns the first deregister error.
static inline ncclResult_t rcclIbSegDeregister(const struct rcclIbSegRegOps* ops, void* ctx, void** mrs, int n) {
  ncclResult_t ret = ncclSuccess;
  if (mrs == NULL || n < 1) return ncclSuccess;
  for (int i = 0; i < n; i++) {
    if (mrs[i] == NULL) continue;
    if (ops != NULL && ops->deregOne != NULL) {
      ncclResult_t r = ops->deregOne(ctx, i, mrs[i]);
      if (ret == ncclSuccess && r != ncclSuccess) ret = r;
    }
    mrs[i] = NULL;
  }
  return ret;
}

// On failure, releases every slot in [0, index] through deregOne, including a handle regOne stored before failing.
static inline ncclResult_t rcclIbSegRegister(const struct rcclIbSegRegOps* ops, void* ctx, void** mrs, int n) {
  if (ops == NULL || ops->regOne == NULL || ops->deregOne == NULL || mrs == NULL || n < 1) return ncclInvalidUsage;
  for (int i = 0; i < n; i++) {
    mrs[i] = NULL;
    ncclResult_t r = ops->regOne(ctx, i, &mrs[i]);
    if (r != ncclSuccess) {
      (void)rcclIbSegDeregister(ops, ctx, mrs, i + 1);
      return r;
    }
  }
  return ncclSuccess;
}

// addrs[n], lens[n], offs[n], fds[n] in one zeroed block. 0 when n < 1.
static inline size_t rcclIbSegWalkBytes(int n) {
  if (n < 1) return 0;
  size_t count = (size_t)n;
  return count * (sizeof(void*) + sizeof(size_t) + sizeof(uint64_t) + sizeof(int));
}

struct rcclIbSegWalk {
  void** addrs;
  size_t* lens;
  uint64_t* offs;
  int* fds;
};

// block is rcclIbSegWalkBytes(n) bytes and malloc-aligned. The four pointers alias that block.
static inline void rcclIbSegWalkBind(void* block, int n, struct rcclIbSegWalk* out) {
  char* p = (char*)block;
  size_t count = n < 1 ? 0 : (size_t)n;
  out->addrs = (void**)p;
  p += count * sizeof(void*);
  out->lens = (size_t*)p;
  p += count * sizeof(size_t);
  out->offs = (uint64_t*)p;
  p += count * sizeof(uint64_t);
  out->fds = (int*)p;
}

// RCCL: classic net entry, one dma-buf MR per physical segment when the export fd is -1.
// IbMsegRegMrDmaBuf calls it; tests call it directly. Defined in reg.cc.
ncclResult_t IbMsegRegMrDmaBufMultiSeg(void* comm, int nSeg, void** segAddrs, size_t* segLens, uint64_t* segOffsets,
                                       int* segFds, int type, void** mhandle);

#endif
