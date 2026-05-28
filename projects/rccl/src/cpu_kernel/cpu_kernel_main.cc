/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host port of ncclKernelMain orchestration (channel mapping, batch loop).
 ************************************************************************/

#include "cpu_kernel_internal.h"
#include "cpu_dev_comm_mirror.h"

#include "checks.h"

#include <cstring>

ncclResult_t rcclCpuExecuteBlock(
    struct ncclComm* comm,
    struct ncclKernelComm* hostComm,
    struct ncclDevKernelArgs* args,
    int blockId,
    int threadCount,
    int warpSize) {
  struct rcclCpuBlockContext ctx;
  struct rcclCpuBlockBarrier bar;
  rcclCpuBlockContextInit(&ctx, warpSize);
  rcclCpuBlockBarrierInit(&bar);

  ctx.threadCount = threadCount;
  std::memcpy(&ctx.args, args, sizeof(ctx.args));
  ctx.comm = *hostComm;
  ctx.channelId = rcclCpuMapBlockToChannel(args, blockId, warpSize);
  ctx.channel = hostComm->channels[ctx.channelId];
  ctx.channel.workCounter = hostComm->channels[ctx.channelId].workCounter;

  if (ctx.comm.abortFlag && rcclCpuLoadSeqCstU32(const_cast<uint32_t*>(ctx.comm.abortFlag))) {
    ctx.aborted = 1;
    goto finish;
  }

  NCCLCHECK(rcclCpuLoadWorkBatch(&ctx, args, blockId, &bar));

  while (ctx.aborted == 0) {
    NCCLCHECK(rcclCpuDispatchWork(&ctx, &bar));
    if (ctx.nextBatchIx < 0) break;
    int batchIx = ctx.nextBatchIx;
    rcclCpuBlockBarrierWait(&bar, 0, threadCount);
    NCCLCHECK(rcclCpuLoadWorkBatch(&ctx, args, batchIx, &bar));
  }

  NCCLCHECK(rcclCpuWritebackChannelCounters(comm, ctx.channelId, ctx.channel.workCounter));

finish:
  rcclCpuBlockBarrierDestroy(&bar);
  return ncclSuccess;
}
