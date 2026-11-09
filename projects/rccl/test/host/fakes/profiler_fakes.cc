/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See profiler_fakes.h.

#include "profiler_fakes.h"

#include <mutex>

#include "comm.h"  // profiler.h is not self-contained
#include "fail_loud.h"
#include "profiler.h"
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_ncclProfilerStartGroupEvent, ncclProfilerStartGroupEvent);
ASSERT_HOOK_MATCHES_PROD(g_ncclProfilerStopGroupEvent, ncclProfilerStopGroupEvent);
ASSERT_HOOK_MATCHES_PROD(g_ncclProfilerStartTaskEvents, ncclProfilerStartTaskEvents);
ASSERT_HOOK_MATCHES_PROD(g_ncclProfilerStopTaskEvents, ncclProfilerStopTaskEvents);
ASSERT_HOOK_MATCHES_PROD(g_ncclProfilerPostPlanWork, ncclProfilerPostPlanWork);
ASSERT_HOOK_MATCHES_PROD(g_ncclProfilerAddPidToProxyOp, ncclProfilerAddPidToProxyOp);
#undef ASSERT_HOOK_MATCHES_PROD

namespace {
std::mutex g_profilerMutex;  // comm teardown can run on any thread
std::vector<void*> g_proxyTraceDumps;
}  // namespace

std::vector<void*> ProfilerProxyTraceDumps() {
  std::lock_guard<std::mutex> lock(g_profilerMutex);
  return g_proxyTraceDumps;
}

void ncclProfilerProxyTraceDumpIfAny(void* profilerContext) {
  std::lock_guard<std::mutex> lock(g_profilerMutex);
  g_proxyTraceDumps.push_back(profilerContext);
}

static ncclResult_t DefaultStartGroupEvent(struct ncclKernelPlan*) {
  FailLoudUnfaked("profiler_fakes", "ncclProfilerStartGroupEvent");
}
std::function<ncclResult_t(struct ncclKernelPlan*)> g_ncclProfilerStartGroupEvent = DefaultStartGroupEvent;
ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan) {
  return g_ncclProfilerStartGroupEvent(plan);
}

static ncclResult_t DefaultStopGroupEvent(struct ncclKernelPlan*) {
  FailLoudUnfaked("profiler_fakes", "ncclProfilerStopGroupEvent");
}
std::function<ncclResult_t(struct ncclKernelPlan*)> g_ncclProfilerStopGroupEvent = DefaultStopGroupEvent;
ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan) {
  return g_ncclProfilerStopGroupEvent(plan);
}

static ncclResult_t DefaultStartTaskEvents(struct ncclKernelPlan*) {
  FailLoudUnfaked("profiler_fakes", "ncclProfilerStartTaskEvents");
}
std::function<ncclResult_t(struct ncclKernelPlan*)> g_ncclProfilerStartTaskEvents = DefaultStartTaskEvents;
ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan) {
  return g_ncclProfilerStartTaskEvents(plan);
}

static ncclResult_t DefaultStopTaskEvents(struct ncclKernelPlan*) {
  FailLoudUnfaked("profiler_fakes", "ncclProfilerStopTaskEvents");
}
std::function<ncclResult_t(struct ncclKernelPlan*)> g_ncclProfilerStopTaskEvents = DefaultStopTaskEvents;
ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan) {
  return g_ncclProfilerStopTaskEvents(plan);
}

static ncclResult_t DefaultPostPlanWork(struct ncclComm*, struct ncclKernelPlan*) {
  FailLoudUnfaked("profiler_fakes", "ncclProfilerPostPlanWork");
}
std::function<ncclResult_t(struct ncclComm*, struct ncclKernelPlan*)> g_ncclProfilerPostPlanWork =
    DefaultPostPlanWork;
ncclResult_t ncclProfilerPostPlanWork(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  return g_ncclProfilerPostPlanWork(comm, plan);
}

static ncclResult_t DefaultAddPidToProxyOp(struct ncclProxyOp*) {
  FailLoudUnfaked("profiler_fakes", "ncclProfilerAddPidToProxyOp");
}
std::function<ncclResult_t(struct ncclProxyOp*)> g_ncclProfilerAddPidToProxyOp = DefaultAddPidToProxyOp;
ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp* op) {
  return g_ncclProfilerAddPidToProxyOp(op);
}

void ResetProfilerFakes() {
  {
    std::lock_guard<std::mutex> lock(g_profilerMutex);
    g_proxyTraceDumps.clear();
  }
  g_ncclProfilerStartGroupEvent = DefaultStartGroupEvent;
  g_ncclProfilerStopGroupEvent = DefaultStopGroupEvent;
  g_ncclProfilerStartTaskEvents = DefaultStartTaskEvents;
  g_ncclProfilerStopTaskEvents = DefaultStopTaskEvents;
  g_ncclProfilerPostPlanWork = DefaultPostPlanWork;
  g_ncclProfilerAddPidToProxyOp = DefaultAddPidToProxyOp;
}
