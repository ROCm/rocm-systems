/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See scheduler_fakes.h.

#include <functional>
#include <unordered_map>

#include "comm.h"
#include "device.h"
#include "enqueue.h"
#include "nccl.h"
#include "sym_kernels.h"
#include "tuning.h"
#include "config/algorithm_registry.h"

#include "scheduler_fakes.h"
#include "sym_kernels_fakes.h"  // g_symkAvailable's canonical home
#include "tuning_fakes.h"       // g_tuningCompute's canonical home

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

// No bits set by default, so the LL-kernel-init-once check never fires regardless of which kernelId g_tuningCompute reports.
static int DefaultSymkLLKernelMask() { return 0; }
std::function<int()> g_symkLLKernelMask = DefaultSymkLLKernelMask;

// Matches nccl_stubs.cc's own hardcoded false: no profiler plugin loaded in this binary by default.
static bool DefaultProfilerPluginLoaded() { return false; }
std::function<bool()> g_profilerPluginLoaded = DefaultProfilerPluginLoaded;

// Index 0 always: the kernel-table arrays below are 1-element placeholders, so any other index overruns them.
static int DefaultSymkGetKernelIndex(ncclSymkKernelId, int, ncclDataType_t) { return 0; }
std::function<int(ncclSymkKernelId, int, ncclDataType_t)> g_symkGetKernelIndex = DefaultSymkGetKernelIndex;

static const char* DefaultSymkKernelIdToString(int) { return "fake-sym-kernel"; }
std::function<const char*(int)> g_symkKernelIdToString = DefaultSymkKernelIdToString;

static int DefaultSymkDynamicSmemKernelMask() { return 0; }
std::function<int()> g_symkDynamicSmemKernelMask = DefaultSymkDynamicSmemKernelMask;

// Trivial success, doesn't touch *outDevWork: the channel-packing loop that reads it is Block 12's territory.
static ncclResult_t DefaultSymkMakeDevWork(struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork*) {
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork*)> g_symkMakeDevWork =
    DefaultSymkMakeDevWork;

void ResetSchedulerFakes() {
  g_testBudget = DefaultTestBudget;
  g_testBudgetCalls = 0;
  g_ncclGetAlgoInfo = DefaultGetAlgoInfo;
  g_planSetDefaultKernel = DefaultPlanSetDefaultKernel;
  g_addWorkBatchToPlan = DefaultAddWorkBatchToPlan;
  g_addProxyOpIfNeeded = DefaultAddProxyOpIfNeeded;
  g_getCollNetSupport = DefaultGetCollNetSupport;
  g_getRegBuff = DefaultGetRegBuff;
  g_symkLLKernelMask = DefaultSymkLLKernelMask;
  g_profilerPluginLoaded = DefaultProfilerPluginLoaded;
  g_symkGetKernelIndex = DefaultSymkGetKernelIndex;
  g_symkKernelIdToString = DefaultSymkKernelIdToString;
  g_symkDynamicSmemKernelMask = DefaultSymkDynamicSmemKernelMask;
  g_symkMakeDevWork = DefaultSymkMakeDevWork;
  ncclSymkKernelList[0] = nullptr;  // raw globals, not std::function seams: reset here to avoid cross-test leaks
  ncclSymkKernelListProfile[0] = nullptr;
  ncclSymkKernelMaxDynamicSmem[0] = 0;
  ncclDevFuncNameToId.clear();
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

// Generated device-function table; empty default matches nccl_stubs.cc's own (a miss returns -1 with a WARN).
std::unordered_map<uint64_t, int> ncclDevFuncNameToId;

// src/sym_kernels.cc (ncclSymkAvailable itself is defined in sym_kernels_fakes.cc)
int ncclSymkLLKernelMask() { return g_symkLLKernelMask(); }
int ncclSymkDynamicSmemKernelMask() { return g_symkDynamicSmemKernelMask(); }
int ncclSymkGetKernelIndex(ncclSymkKernelId kernelId, int red, ncclDataType_t ty) {
  return g_symkGetKernelIndex(kernelId, red, ty);
}
const char* ncclSymkKernelIdToString(int kernelId) { return g_symkKernelIdToString(kernelId); }
ncclResult_t ncclSymkMakeDevWork(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclSymkDevWork* outDevWork) {
  return g_symkMakeDevWork(comm, task, outDevWork);
}

// Generated kernel tables; one-element placeholders since g_symkGetKernelIndex's default always selects index 0.
void* ncclSymkKernelList[1] = {nullptr};
void* ncclSymkKernelListProfile[1] = {nullptr};
int ncclSymkKernelMaxDynamicSmem[1] = {0};

// src/config/algorithm_registry.cc: a fixed name is fine since INFO()'s macro guard only evaluates this when logging is on.
const char* ncclAlgNameForSymk(int) { return "sym-kernel"; }

// src/tuning/tuning.cc's ncclTuningCompute is defined in tuning_fakes.cc.

// src/plugin/profiler.cc
bool ncclProfilerPluginLoaded(void) { return g_profilerPluginLoaded(); }
