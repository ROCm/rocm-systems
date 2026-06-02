/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host port of ncclKernelMain orchestration (channel mapping, batch loop).
 ************************************************************************/

#include "cpu_kernel_internal.h"
#include "cpu_dev_comm_mirror.h"

#include "checks.h"

#include <cstring>
#include <memory>

ncclResult_t rcclCpuExecuteBlock(
    struct ncclComm* comm,
    struct rcclCpuCommMirrorState* mirror,
    struct ncclKernelComm* hostComm,
    struct ncclDevKernelArgs* args,
    int blockId,
    int threadCount,
    int warpSize) {
  // HIP host-function callbacks run on a small stack; keep the large block context on the heap.
  auto ctx = std::make_unique<rcclCpuBlockContext>();
  struct rcclCpuBlockBarrier bar;
  rcclCpuBlockContextInit(ctx.get(), warpSize);
  rcclCpuBlockBarrierInit(&bar);

  constexpr int kCpuLogicalThreads = 1;
  (void)threadCount;
  ctx->threadCount = kCpuLogicalThreads;
  ctx->cudaDev = comm->cudaDev;
  ctx->hostAbortFlag = comm->abortFlag;
  std::memcpy(&ctx->args, args, sizeof(ctx->args));
  ctx->comm = hostComm;
  ctx->channelId = rcclCpuMapBlockToChannel(args, blockId, warpSize);
  if (ctx->channelId < 0 || ctx->channelId >= MAXCHANNELS) return ncclInternalError;
  ctx->channel = &hostComm->channels[ctx->channelId];

  if (ctx->hostAbortFlag && rcclCpuLoadSeqCstU32(const_cast<uint32_t*>(ctx->hostAbortFlag))) {
    ctx->aborted = 1;
    goto finish;
  }

  NCCLCHECK(rcclCpuLoadWorkBatch(ctx.get(), args, blockId, &bar));

  while (ctx->aborted == 0) {
    NCCLCHECK(rcclCpuDispatchWork(ctx.get(), &bar, 0, kCpuLogicalThreads));
    if (ctx->nextBatchIx < 0) break;
    int batchIx = ctx->nextBatchIx;
    rcclCpuBlockBarrierWait(&bar, 0, kCpuLogicalThreads);
    NCCLCHECK(rcclCpuLoadWorkBatch(ctx.get(), args, batchIx, &bar));
  }

  NCCLCHECK(rcclCpuWritebackChannelCounters(comm, mirror, ctx->channelId, ctx->channel->workCounter));

finish:
  rcclCpuBlockBarrierDestroy(&bar);
  return ncclSuccess;
}
