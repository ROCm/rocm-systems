/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Symmetric: kernels over symmetric (registered) windows, picked by the tuning cost model.

#include "debug.h"
#include "dev_runtime.h"
#include "enqueue.h"
#include "rccl_scheduler/rccl_scheduler.h"
#include "sym_kernels.h"
#include "tuning.h"

static int symkHostRedOpToDev(ncclRedOp_t op) {
  switch ((int)op) {
  case ncclSum:
    return (int)ncclDevSum;
  case ncclProd:
    return (int)ncclDevProd;
  case ncclMin:
  case ncclMax:
    return (int)ncclDevMinMax;
  case ncclAvg:
    return (int)ncclDevSumPostDiv;
  default:
    return -1;
  }
}

// Fills (algo, protocol, maxChannels) for the symmetric kernel that would run
// for these operands; returns false if symk is unavailable or no kernel is
// picked. Single source shared by rcclSymKGetInfo and the query-mode symmetric
// reporting in rcclSelectAllReduce/rcclSelectAllGather/rcclSelectReduceScatter.
// algo is RCCL_SYMMETRIC_TDM when the pick is a Tensor Data Mover kernel, else
// RCCL_SYMMETRIC. symAligned16B is ncclSymAligned16B for the operands; rcclSymKGetInfo
// has no buffers and leaves it true.
bool rcclSymkQuery(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
                          ncclRedOp_t op, int* algo, int* protocol, int* maxChannels, bool symAligned16B) {
  // Symmetric kernels need symmetric-window support (cuMem). Without it the
  // windows are never registered symmetric and symk cannot run, so do not report
  // it -- rcclSymKGetInfo's caller then falls back to the actual backend.
  if (comm == nullptr || !comm->symmetricSupport) return false;
  if (coll != ncclFuncAllReduce && coll != ncclFuncAllGather && coll != ncclFuncReduceScatter) return false;
  int devOp = (coll == ncclFuncAllGather) ? (int)ncclDevSum : symkHostRedOpToDev(op);
  if (devOp < 0) return false;
  if (ncclSymkInitOnce(comm) != ncclSuccess) return false;
  if (!ncclSymkAvailable(comm, coll, devOp, dataType, (size_t)count)) return false;
  // NCCL 2.31 replaced ncclSymkPickKernel() with the tuning cost model; restricting
  // the tuning mask to the symmetric kernels reproduces the old query.
  struct ncclTuningInput_t input = {};
  input.comm = comm;
  input.tuningMask = NCCL_TUNING_MASK_SYM_KERNELS;
  input.func = coll;
  input.redOp = op;
  input.devRedOp = (ncclDevRedOp_t)devOp;
  input.datatype = dataType;
  input.nBytes = (size_t)count * ncclTypeSize(dataType);
  input.count = (size_t)count;
  input.countMax = (size_t)count;
  input.nWorks = 1;
  input.winRegType = ncclSymSendRegRecvReg;
  input.minCTAs = comm->config.minCTAs;
  input.maxCTAs = comm->config.maxCTAs;
  input.CTAPolicy = comm->config.CTAPolicy;
  input.nvlsSupport = comm->nvlsSupport && (ncclNvlsSupported(devOp, dataType) || coll == ncclFuncAllGather);
  input.symAligned16B = symAligned16B;
  struct ncclTuningResult_t bestTuning = NCCL_TUNING_RESULT_INIT;
  if (ncclTuningCompute(&input, &bestTuning) != ncclSuccess) return false;
  ncclSymkKernelId kernelId = (ncclSymkKernelId)bestTuning.symKernelId;
  if (kernelId == ncclSymkKernelId_Count) return false;
  *maxChannels = bestTuning.maxChannels;
  *algo = rcclSymkKernelIdIsTma((int)kernelId) ? (int)rcclAddonAlgos_t::RCCL_SYMMETRIC_TDM
                                                : (int)rcclAddonAlgos_t::RCCL_SYMMETRIC;
  *protocol = rcclSymkKernelIdIsLL((int)kernelId) ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE;
  return true;
}
bool rcclSymkRequested(rcclSchedCtx& ctx) {
  struct ncclComm* comm = ctx.comm;
  const ncclRedOp_t op = ctx.op;
  int devOp = (int)ncclDevSum;
  switch (ctx.func) {
  case ncclFuncAllReduce:
    if (op != ncclSum) return false;
    break;
  case ncclFuncReduceScatter:
    if (op != ncclSum && op != ncclAvg) return false;
    if (op == ncclAvg) devOp = (int)ncclDevSumPostDiv;
    break;
  case ncclFuncAllGather:
  case ncclFuncAlltoAll: break;
  default: return false;
  }
  return isSymmetricKernelRequestedWin(comm, ctx.func, devOp, ctx.datatype, ctx.count, ctx.sendWin, ctx.recvWin);
}

namespace {

// AllGather: the TDM window admits symk when the model picks a TDM kernel; across trays it is the only way in.
bool allGatherEligible(rcclSchedCtx& ctx) {
  struct ncclComm* comm = ctx.comm;
  const size_t bytes = ctx.totalBytes;
  const rcclArchThresholds* t = ctx.archTable;
  bool tdmAdmits = false;
  if (rcclSymTdmWindowContains(t, ncclFuncAllGather, bytes, comm->spansTrays, ctx.capturing)) {
    int algo, proto, channels;
    tdmAdmits = rcclSymkQuery(comm, ncclFuncAllGather, ctx.count, ctx.datatype, ncclSum, &algo, &proto, &channels,
                              ncclSymAligned16B(ncclFuncAllGather, comm->rank, ctx.sendbuff, ctx.recvbuff,
                                                ctx.sendWin, ctx.recvWin, ctx.countBytes)) &&
                algo == (int)rcclAddonAlgos_t::RCCL_SYMMETRIC_TDM;
  }
  ctx.base.symkTdmOnly = tdmAdmits;
  if (tdmAdmits) {
    if (!ctx.query) INFO(NCCL_TUNING, "AG symk admitted by the TDM window: totalBytes=%zu", bytes);
    return true;
  }
  const size_t symMin = rcclAgSymMinR2Cap(comm, t);
  const size_t symMax = rcclAgSymMaxR2Cap(comm, t, ctx.capturing);
  const bool eligible = !(symMin > 0 && bytes < symMin) && bytes <= symMax && !comm->spansTrays;
  if (!eligible && !ctx.query)
    INFO(NCCL_TUNING, "AG symk withdrawn: totalBytes=%zu window=[%zu, %zu] spansTrays=%d", bytes, symMin, symMax,
         (int)comm->spansTrays);
  return eligible;
}

}  // namespace

bool rcclSymkEligible(rcclSchedCtx& ctx) {
  if (!rcclSymkRequested(ctx)) {
    if (ctx.func == ncclFuncAllGather) ctx.base.symkTdmOnly = false;
    return false;
  }
  if (ctx.func == ncclFuncAllGather) return allGatherEligible(ctx);
  const size_t bytes = ctx.totalBytes;
  const bool graphMode = ctx.func == ncclFuncAllReduce && ctx.capturing;
  const size_t symMin = ctx.func == ncclFuncAlltoAll ? 0 : rcclSymMinR2CapTab(ctx.archTable, ctx.func);
  const size_t symMax = rcclSymMaxR2CapTab(ctx.archTable, ctx.func, graphMode);
  const bool eligible = !(symMin > 0 && bytes < symMin) && bytes <= symMax;
  if (!eligible && !ctx.query)
    INFO(NCCL_TUNING, "%s symk withdrawn: totalBytes=%zu window=[%zu, %zu]", ctx.name, bytes, symMin, symMax);
  return eligible;
}

namespace {

// Live, enqueue extracts the symk task (symkExtract); the query reports the kernel the model picks.
ncclResult_t kernelChoose(rcclSchedCtx& c, rcclCollDecision* d, bool* ok) {
  d->algo = (int)rcclAddonAlgos_t::RCCL_SYMMETRIC;
  *ok = true;
  if (!c.query) return ncclSuccess;
  int algo, proto, channels;
  *ok = rcclSymkQuery(c.comm, c.func, c.count, c.datatype, c.op, &algo, &proto, &channels,
                      ncclSymAligned16B(c.func, c.comm->rank, c.sendbuff, c.recvbuff, c.sendWin, c.recvWin,
                                        c.countBytes));
  if (*ok) {
    d->algo = algo;
    d->protocol = proto;
    d->nMaxChannels = channels;
  }
  return ncclSuccess;
}

}  // namespace

const rcclBackend kSymmetricKernel = {"symk.kernel", RCCL_BACKEND_SYMMETRIC, nullptr, kernelChoose, nullptr, true};
