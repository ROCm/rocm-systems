/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#include "cpu_kernel_internal.h"
#include "cpu_device_guard.h"
#include "cpu_mem.h"

#include <algorithm>
#include <cstring>

static void rcclCpuCopy16(int tid, int tn, void* dst, void const* src, int bytes) {
  for (int offset = 16 * tid; offset < bytes; offset += 16 * tn) {
    std::memcpy(static_cast<char*>(dst) + offset, static_cast<char const*>(src) + offset, 16);
  }
}

ncclResult_t rcclCpuLoadWorkBatch(
    struct rcclCpuBlockContext* ctx,
    struct ncclDevKernelArgs const* args,
    int batchIx,
    struct rcclCpuBlockBarrier* bar) {
  struct ncclDevWorkBatch const* batches = reinterpret_cast<struct ncclDevWorkBatch const*>(args + 1);
  struct ncclDevWorkBatch batch = batches[batchIx];

  int nWorks = 0;
  if (ctx->warpSize == 64) {
    nWorks = __builtin_popcountll(static_cast<uint64_t>(batch.offsetBitset));
  } else {
    uint32_t low = static_cast<uint32_t>(batch.offsetBitset);
    uint32_t high = static_cast<uint32_t>(batch.offsetBitset >> 32);
    nWorks = __builtin_popcount(low) + __builtin_popcount(high);
  }

  int workSize = 0;
  switch (batch.workType) {
  case ncclDevWorkTypeP2p:
    workSize = sizeof(struct ncclDevWorkP2p);
    break;
  case ncclDevWorkTypeColl:
    workSize = sizeof(struct ncclDevWorkColl);
    break;
  case ncclDevWorkTypeCollReg:
    workSize = sizeof(struct ncclDevWorkCollReg);
    break;
  default:
    return ncclInternalError;
  }

  ctx->workType = batch.workType;
  ctx->workSize = workSize;
  ctx->nWorks = nWorks;
  ctx->funcId = batch.funcId;
  ctx->nextBatchIx = (batch.nextJump == 0) ? -1 : static_cast<int>(batchIx + batch.nextJump);

  char const* workBase = nullptr;
  if (args->workStorageType == ncclDevWorkStorageTypeArgs) {
    workBase = reinterpret_cast<char const*>(args) + batch.offsetBase;
  } else if (args->workBuf != nullptr) {
    workBase = static_cast<char const*>(args->workBuf) + (batch.offsetBase & args->workMask);
  } else {
    return ncclInternalError;
  }

  int totalBytes = nWorks * workSize;
  if (totalBytes > static_cast<int>(sizeof(ctx->workStorage))) return ncclInternalError;

  if (args->workStorageType == ncclDevWorkStorageTypeArgs) {
    int tn = std::min(ctx->threadCount, std::max(1, (totalBytes + 15) / 16));
    for (int t = 0; t < tn; t++) {
      rcclCpuCopy16(t, tn, ctx->workStorage, workBase, totalBytes);
    }
  } else {
    NCCLCHECK(rcclCpuCopyBytes(ctx->cudaDev, ctx->workStorage, workBase, totalBytes));
  }
  rcclCpuBlockBarrierWait(bar, 0, 1);
  return ncclSuccess;
}
