/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#include "cpu_device_context.h"

#include "bitops.h"

#include <cstring>

void rcclCpuBlockBarrierInit(struct rcclCpuBlockBarrier* b) {
  pthread_mutex_init(&b->mutex, nullptr);
  pthread_cond_init(&b->cond, nullptr);
  b->arrived = 0;
  b->generation = 0;
  b->expected = 0;
}

void rcclCpuBlockBarrierDestroy(struct rcclCpuBlockBarrier* b) {
  pthread_cond_destroy(&b->cond);
  pthread_mutex_destroy(&b->mutex);
}

void rcclCpuBlockBarrierWait(struct rcclCpuBlockBarrier* b, int tid, int tn) {
  if (tn <= 1) {
    rcclCpuFenceBlock();
    return;
  }
  pthread_mutex_lock(&b->mutex);
  if (tid == 0) b->expected = tn;
  b->arrived += 1;
  int gen = b->generation;
  if (b->arrived == b->expected) {
    b->arrived = 0;
    b->generation = gen + 1;
    pthread_cond_broadcast(&b->cond);
    pthread_mutex_unlock(&b->mutex);
    rcclCpuFenceBlock();
    return;
  }
  while (b->generation == gen) pthread_cond_wait(&b->cond, &b->mutex);
  pthread_mutex_unlock(&b->mutex);
  rcclCpuFenceBlock();
}

void rcclCpuBlockContextInit(struct rcclCpuBlockContext* ctx, int warpSize) {
  std::memset(ctx, 0, sizeof(*ctx));
  ctx->warpSize = warpSize;
  ctx->cudaDev = -1;
  ctx->comm = nullptr;
  ctx->channel = nullptr;
  ctx->hostAbortFlag = nullptr;
  ctx->nextBatchIx = -1;
  ctx->aborted = 0;
}

int rcclCpuMapBlockToChannel(struct ncclDevKernelArgs const* args, int blockId, int warpSize) {
  int total = 0;
  int num = MAXCHANNELS/64 > 0 ? MAXCHANNELS/64 : 1;
  for (int i = 0; i < num; i++) {
    uint64_t mask = args->channelMask.masks[i];
    int nSet = countOneBits(mask);
    for (int x = 0; x < 64; x++) {
      if (mask & (1ull << x)) {
        if (blockId == total + __builtin_popcountll(mask & ((1ull << x) - 1))) {
          return x + total;
        }
      }
      if (warpSize < 64 && x == warpSize - 1) break;
    }
    total += nSet;
  }
  return blockId;
}
