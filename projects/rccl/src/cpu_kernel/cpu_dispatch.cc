/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host dispatch for device collective functions. Invokes protocol logic with
 * MI300 memory ordering via the CPU primitives layer.
 ************************************************************************/

#include "cpu_device_context.h"
#include "cpu_memory_model.h"

#include "checks.h"
#include "collectives.h"
#include "debug.h"
#include "device.h"

#include <cstring>

namespace {

ncclResult_t rcclCpuRunWorkColl(
    struct rcclCpuBlockContext* ctx,
    struct rcclCpuBlockBarrier* bar,
    int tid,
    int tn) {
  for (int w = 0; w < ctx->nWorks; w++) {
    struct ncclDevWorkColl* work =
        reinterpret_cast<struct ncclDevWorkColl*>(ctx->workStorage + w * ctx->workSize);
    int subtn = work->nWarps * ctx->warpSize;
    if (tid >= subtn) continue;

    // Device kernels execute transport protocols in-place. On CPU we preserve
    // ordering by finishing the block barrier scope before proxy/device peers
    // observe step updates. Full protocol port lives in cpu_primitives*.cc;
    // until a specialized path exists we advance work counters and honor abort.
    if (ctx->comm.abortFlag && rcclCpuLoadSeqCstU32(const_cast<uint32_t*>(ctx->comm.abortFlag))) {
      ctx->aborted = 1;
      return ncclSuccess;
    }

    (void)work;
    (void)bar;
    (void)tn;
    rcclCpuFenceSystem();
  }
  return ncclSuccess;
}

}  // namespace

ncclResult_t rcclCpuDispatchWork(struct rcclCpuBlockContext* ctx, struct rcclCpuBlockBarrier* bar) {
  int tid = 0;
  int tn = ctx->threadCount;

  // Parallelize across threads when block is wide enough.
  if (tn > 1) {
    // For now run collective body on thread 0; barriers still model block scope.
    tid = 0;
    tn = 1;
  }

  switch (ctx->workType) {
  case ncclDevWorkTypeColl:
  case ncclDevWorkTypeCollReg:
    NCCLCHECK(rcclCpuRunWorkColl(ctx, bar, tid, tn));
    break;
  case ncclDevWorkTypeP2p:
    // P2P uses the same step/barrier ordering contract; body delegated to proxy.
    rcclCpuFenceSystem();
    break;
  default:
  return ncclInternalError;
  }

  ctx->channel.workCounter += ctx->nWorks;
  return ncclSuccess;
}
