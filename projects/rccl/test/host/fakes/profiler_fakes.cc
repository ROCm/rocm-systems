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

namespace {
std::mutex g_profilerMutex;  // comm teardown can run on any thread
std::vector<void*> g_proxyTraceDumps;
}  // namespace

std::vector<void*> ProfilerProxyTraceDumps() {
  std::lock_guard<std::mutex> lock(g_profilerMutex);
  return g_proxyTraceDumps;
}

std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStartGroupEvent;
std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStopGroupEvent;
std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStartTaskEvents;
std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStopTaskEvents;
std::function<ncclResult_t(struct ncclComm*, struct ncclKernelPlan*)> g_profilerPostPlanWork;

void ResetProfilerFakes() {
  std::lock_guard<std::mutex> lock(g_profilerMutex);
  g_proxyTraceDumps.clear();
  g_profilerStartGroupEvent = nullptr;
  g_profilerStopGroupEvent = nullptr;
  g_profilerStartTaskEvents = nullptr;
  g_profilerStopTaskEvents = nullptr;
  g_profilerPostPlanWork = nullptr;
}

void ncclProfilerProxyTraceDumpIfAny(void* profilerContext) {
  std::lock_guard<std::mutex> lock(g_profilerMutex);
  g_proxyTraceDumps.push_back(profilerContext);
}

ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan) {
  if (!g_profilerStartGroupEvent) FailLoudUnfaked("profiler_fakes", "ncclProfilerStartGroupEvent");
  return g_profilerStartGroupEvent(plan);
}
ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan) {
  if (!g_profilerStopGroupEvent) FailLoudUnfaked("profiler_fakes", "ncclProfilerStopGroupEvent");
  return g_profilerStopGroupEvent(plan);
}
ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan) {
  if (!g_profilerStartTaskEvents) FailLoudUnfaked("profiler_fakes", "ncclProfilerStartTaskEvents");
  return g_profilerStartTaskEvents(plan);
}
ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan) {
  if (!g_profilerStopTaskEvents) FailLoudUnfaked("profiler_fakes", "ncclProfilerStopTaskEvents");
  return g_profilerStopTaskEvents(plan);
}
ncclResult_t ncclProfilerPostPlanWork(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  if (!g_profilerPostPlanWork) FailLoudUnfaked("profiler_fakes", "ncclProfilerPostPlanWork");
  return g_profilerPostPlanWork(comm, plan);
}
ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp*) {
  FailLoudUnfaked("profiler_fakes", "ncclProfilerAddPidToProxyOp");
}
