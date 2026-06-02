/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host dispatch for device collective functions. Invokes protocol logic with
 * MI300 memory ordering via the CPU primitives layer.
 ************************************************************************/

#include "cpu_kernel_internal.h"
#include "cpu_coll_exec.h"
#include "cpu_func_decode.h"
#include "cpu_memory_model.h"

#include "checks.h"
#include "collectives.h"
#include "debug.h"
#include "device.h"

namespace {

ncclResult_t rcclCpuRunWorkColl(
    struct rcclCpuBlockContext* ctx,
    struct rcclCpuBlockBarrier* bar,
    int tid,
    int tn) {
  struct rcclCpuFuncDesc desc{};
  if (!rcclCpuDecodeFuncId(ctx->funcId, &desc)) {
    WARN("rcclCpuRunWorkColl: unknown funcId %u", ctx->funcId);
    return ncclInternalError;
  }

  if (ctx->workType == ncclDevWorkTypeCollReg) {
    auto* reg = reinterpret_cast<struct ncclDevWorkCollReg*>(ctx->workStorage);
    int subtn = std::min(tn, reg->coll.nWarps * ctx->warpSize);
    if (tid < subtn) {
      NCCLCHECK(rcclCpuExecuteCollWork(ctx, bar, tid, subtn, &reg->coll, desc));
    }
    return ncclSuccess;
  }

  for (int w = 0; w < ctx->nWorks; w++) {
    if (ctx->workSize <= 0 ||
        (w + 1) * ctx->workSize > static_cast<int>(sizeof(ctx->workStorage))) {
      return ncclInternalError;
    }
    struct ncclDevWorkColl* work =
        reinterpret_cast<struct ncclDevWorkColl*>(ctx->workStorage + w * ctx->workSize);
    if (work < reinterpret_cast<struct ncclDevWorkColl*>(ctx->workStorage) ||
        work >= reinterpret_cast<struct ncclDevWorkColl*>(ctx->workStorage + sizeof(ctx->workStorage))) {
      return ncclInternalError;
    }
    int subtn = std::min(tn, static_cast<int>(work->nWarps) * ctx->warpSize);
    if (tid >= subtn) continue;

  if (w != 0) {
    struct ncclDevWorkColl* workPrev =
        reinterpret_cast<struct ncclDevWorkColl*>(ctx->workStorage + (w - 1) * ctx->workSize);
    if (work->nWarps != workPrev->nWarps) rcclCpuBlockBarrierWait(bar, tid, tn);
  }

  NCCLCHECK(rcclCpuExecuteCollWork(ctx, bar, tid, subtn, work, desc));
  }
  return ncclSuccess;
}

ncclResult_t rcclCpuRunWorkP2p(
    struct rcclCpuBlockContext* ctx,
    struct rcclCpuBlockBarrier* bar,
    int tid,
    int tn) {
  struct rcclCpuFuncDesc desc{};
  if (!rcclCpuDecodeFuncId(ctx->funcId, &desc)) {
    desc.coll = ncclFuncSendRecv;
    desc.proto = NCCL_PROTO_SIMPLE;
    desc.datatype = ncclInt8;
    desc.devRedOp = ncclDevSum;
    desc.valid = true;
  }

  for (int w = 0; w < ctx->nWorks; w++) {
    struct ncclDevWorkP2p* work =
        reinterpret_cast<struct ncclDevWorkP2p*>(ctx->workStorage + w * ctx->workSize);
    NCCLCHECK(rcclCpuExecuteP2pWork(ctx, bar, tid, tn, work, desc));
  }
  return ncclSuccess;
}

}  // namespace

ncclResult_t rcclCpuDispatchWork(struct rcclCpuBlockContext* ctx, struct rcclCpuBlockBarrier* bar, int tid, int tn) {
  if (ctx->hostAbortFlag && rcclCpuLoadSeqCstU32(const_cast<uint32_t*>(ctx->hostAbortFlag))) {
    ctx->aborted = 1;
    return ncclSuccess;
  }

  switch (ctx->workType) {
  case ncclDevWorkTypeColl:
  case ncclDevWorkTypeCollReg:
    NCCLCHECK(rcclCpuRunWorkColl(ctx, bar, tid, tn));
    break;
  case ncclDevWorkTypeP2p:
    NCCLCHECK(rcclCpuRunWorkP2p(ctx, bar, tid, tn));
    break;
  default:
    return ncclInternalError;
  }

  if (ctx->channel) ctx->channel->workCounter += ctx->nWorks;
  return ncclSuccess;
}
