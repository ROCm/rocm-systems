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

// GIN: GPU-initiated networking (Anvil SDMA over LSA, rocSHMEM GDA).

#include "debug.h"
#include "rccl_scheduler/rccl_scheduler.h"
#if defined(ENABLE_ROCSHMEM_GIN)
#include "algorithms/gin/gin_all_gather.h"
#include "algorithms/gin/gin_all_reduce.h"
#endif
#include "algorithms/gin/gin_alltoall.h"

bool rcclUseAlltoAllGda(struct ncclComm* comm) {
#ifdef ENABLE_ROCSHMEM
  if (comm->enableRocshmem && comm->nNodes > 1 && (comm->nRanks / comm->nNodes == 8) &&
      comm->rocshmemThreshold <= 1048576) {
    INFO(NCCL_INIT, "Enabling GDA alltoall for RCCL");
    return true;
  }
#endif
  return false;
}
// GIN AllReduce claims only part of the symmetric range; below it, DDA may take the call.
bool rcclGinArYieldsToDda(rcclSchedCtx& ctx) {
#if defined(ENABLE_ROCSHMEM_GIN)
  return ncclAllReduceGinSdmaYieldToDda(ctx.comm, ctx.sendbuff, ctx.recvbuff, ctx.count, ctx.datatype, ctx.op);
#else
  (void)ctx;
  return false;
#endif
}

namespace {

bool sdmaEligible(rcclSchedCtx& c) {
#if defined(ENABLE_ROCSHMEM_GIN)
  switch (c.func) {
  case ncclFuncAllReduce:
    return ncclAllReduceGinSdmaEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype, c.op);
  case ncclFuncAllGather: return ncclAllGatherGinSdmaEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype);
  case ncclFuncAlltoAll: return ncclAllToAllGinSdmaEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype);
  default: return false;
  }
#else
  (void)c;
  return false;
#endif
}

ncclResult_t sdmaChoose(rcclSchedCtx& c, rcclCollDecision* d, bool* ok) {
#if defined(ENABLE_ROCSHMEM_GIN)
  switch (c.func) {
  case ncclFuncAllReduce:
    d->algo = RCCL_GIN_SDMA;
    d->nMaxChannels = kGinAllReduceLsaCtas;
    break;
  case ncclFuncAllGather:
    d->algo = RCCL_AG_GIN_SDMA;
    d->nMaxChannels = ncclAllGatherGinSdmaCtaCount(c.comm->nRanks, c.countBytes);
    break;
  default: d->algo = RCCL_A2A_GIN_SDMA; break;
  }
  *ok = true;
#else
  (void)c; (void)d;
  *ok = false;
#endif
  return ncclSuccess;
}

bool gdaEligible(rcclSchedCtx& c) {
#ifdef ENABLE_ROCSHMEM
  return rcclUseAlltoAllGda(c.comm) && c.totalBytes <= c.comm->rocshmemThreshold;
#else
  (void)c;
  return false;
#endif
}

ncclResult_t gdaChoose(rcclSchedCtx&, rcclCollDecision* d, bool* ok) {
  d->algo = RCCL_A2A_GDA;
  d->nMaxChannels = 1;  // getAlgoInfo() pins the GDA kernels to one channel
  *ok = true;
  return ncclSuccess;
}

}  // namespace

const rcclBackend kGinSdma = {"gin.sdma", RCCL_BACKEND_GIN_SDMA, sdmaEligible, sdmaChoose, nullptr, false, true};
const rcclBackend kGinGda = {"gin.gda", RCCL_BACKEND_GIN_SDMA, gdaEligible, gdaChoose, nullptr, false, true};
