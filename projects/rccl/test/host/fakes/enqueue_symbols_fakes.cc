/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See enqueue_symbols_fakes.h.

#include <functional>

#include "comm.h"
#include "device.h"
#include "enqueue.h"
#include "nccl.h"
#include "sym_kernels.h"
#include "tuning.h"

#include "signature-drift.h"

#include "enqueue_symbols_fakes.h"
#include "sym_kernels_fakes.h"  // g_symkAvailable and the sym_kernels.cc seams below's canonical home
#include "tuning_fakes.h"       // g_tuningCompute's canonical home

ASSERT_HOOK_MATCHES_PROD(g_testBudget, ncclTestBudget);
ASSERT_HOOK_MATCHES_PROD(g_ncclGetAlgoInfo, ncclGetAlgoInfo);
ASSERT_HOOK_MATCHES_PROD(g_planSetDefaultKernel, ncclPlanSetDefaultKernel);
ASSERT_HOOK_MATCHES_PROD(g_addWorkBatchToPlan, ncclAddWorkBatchToPlan);
ASSERT_HOOK_MATCHES_PROD(g_addProxyOpIfNeeded, ncclAddProxyOpIfNeeded);
ASSERT_HOOK_MATCHES_PROD(g_getCollNetSupport, ncclGetCollNetSupport);
ASSERT_HOOK_MATCHES_PROD(g_getRegBuff, ncclGetRegBuff);
#undef ASSERT_HOOK_MATCHES_PROD

// Generous default: a deny-everything default would make even a single small task look over budget.
static bool DefaultTestBudget(struct ncclKernelPlanBudget*, int, ssize_t) { return true; }
std::function<bool(struct ncclKernelPlanBudget*, int, ssize_t)> g_testBudget = DefaultTestBudget;
int g_testBudgetCalls = 0;

bool ncclTestBudget(struct ncclKernelPlanBudget* budget, int nWorkBatches, ssize_t nWorkBytes) {
  ++g_testBudgetCalls;
  return g_testBudget(budget, nWorkBatches, nWorkBytes);
}

// Generous default: a usable (proto, channel, warp) triple lets an uninterested caller proceed.
static ncclResult_t DefaultGetAlgoInfo(struct ncclComm*, struct ncclTaskColl* task, int, int, int, ncclSimInfo_t*) {
  task->protocol = NCCL_PROTO_SIMPLE;
  task->nMaxChannels = 1;
  task->nWarps = 1;
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, int, int, int, ncclSimInfo_t*)> g_ncclGetAlgoInfo =
    DefaultGetAlgoInfo;

// No-op default: this binary has no real kernel table to select into.
static void DefaultPlanSetDefaultKernel(struct ncclComm*, struct ncclKernelPlan*) {}
std::function<void(struct ncclComm*, struct ncclKernelPlan*)> g_planSetDefaultKernel = DefaultPlanSetDefaultKernel;

// No-op default: nothing here inspects the plan's work-batch fifo directly.
static void DefaultAddWorkBatchToPlan(struct ncclComm*, struct ncclKernelPlan*, int, enum ncclDevWorkType, int,
                                     uint32_t, int, int, bool) {}
std::function<void(struct ncclComm*, struct ncclKernelPlan*, int, enum ncclDevWorkType, int, uint32_t, int, int,
                   bool)>
    g_addWorkBatchToPlan = DefaultAddWorkBatchToPlan;

// Generous default: accepts every proxy op, matching an always-available proxy thread.
static ncclResult_t DefaultAddProxyOpIfNeeded(struct ncclComm*, struct ncclKernelPlan*, struct ncclProxyOp*) {
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, struct ncclKernelPlan*, struct ncclProxyOp*)> g_addProxyOpIfNeeded =
    DefaultAddProxyOpIfNeeded;

// Generous default: no real collnet/registration to report, matching a plain host-only comm.
static ncclResult_t DefaultGetCollNetSupport(struct ncclComm*, struct ncclTaskColl*, int* out) {
  if (out) *out = 0;
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, int*)> g_getCollNetSupport =
    DefaultGetCollNetSupport;
static ncclResult_t DefaultGetRegBuff(struct ncclComm*, struct ncclTaskColl*, int* out) {
  if (out) *out = 0;
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, int*)> g_getRegBuff = DefaultGetRegBuff;

void ResetEnqueueSymbolsFakes() {
  g_testBudget = DefaultTestBudget;
  g_testBudgetCalls = 0;
  g_ncclGetAlgoInfo = DefaultGetAlgoInfo;
  g_planSetDefaultKernel = DefaultPlanSetDefaultKernel;
  g_addWorkBatchToPlan = DefaultAddWorkBatchToPlan;
  g_addProxyOpIfNeeded = DefaultAddProxyOpIfNeeded;
  g_getCollNetSupport = DefaultGetCollNetSupport;
  g_getRegBuff = DefaultGetRegBuff;
}

// src/enqueue/enqueue.cc
ncclResult_t ncclGetAlgoInfo(struct ncclComm* comm, struct ncclTaskColl* task, int collNetSupport, int nvlsSupport,
                             int nTasksPerChannel, ncclSimInfo_t* simInfo) {
  return g_ncclGetAlgoInfo(comm, task, collNetSupport, nvlsSupport, nTasksPerChannel, simInfo);
}
void ncclAddWorkBatchToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan, int channelId,
                            enum ncclDevWorkType workType, int devFuncId, uint32_t workOffset, int p2pEpoch,
                            int p2pRound, bool newBatch) {
  g_addWorkBatchToPlan(comm, plan, channelId, workType, devFuncId, workOffset, p2pEpoch, p2pRound, newBatch);
}
void ncclPlanSetDefaultKernel(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  g_planSetDefaultKernel(comm, plan);
}
ncclResult_t ncclAddProxyOpIfNeeded(struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclProxyOp* op) {
  return g_addProxyOpIfNeeded(comm, plan, op);
}
ncclResult_t ncclGetCollNetSupport(struct ncclComm* comm, struct ncclTaskColl* task, int* out) {
  return g_getCollNetSupport(comm, task, out);
}
ncclResult_t ncclGetRegBuff(struct ncclComm* comm, struct ncclTaskColl* task, int* out) {
  return g_getRegBuff(comm, task, out);
}

// src/tuning/tuning.cc's ncclTuningCompute is defined in tuning_fakes.cc.
