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

void ResetProfilerFakes() {
  std::lock_guard<std::mutex> lock(g_profilerMutex);
  g_proxyTraceDumps.clear();
}

void ncclProfilerProxyTraceDumpIfAny(void* profilerContext) {
  std::lock_guard<std::mutex> lock(g_profilerMutex);
  g_proxyTraceDumps.push_back(profilerContext);
}

ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan*) {
  FailLoudUnfaked("profiler_fakes", "ncclProfilerStartGroupEvent");
}
ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan*) {
  FailLoudUnfaked("profiler_fakes", "ncclProfilerStopGroupEvent");
}
ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan*) {
  FailLoudUnfaked("profiler_fakes", "ncclProfilerStartTaskEvents");
}
ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan*) {
  FailLoudUnfaked("profiler_fakes", "ncclProfilerStopTaskEvents");
}
ncclResult_t ncclProfilerPostPlanWork(struct ncclComm*, struct ncclKernelPlan*) {
  FailLoudUnfaked("profiler_fakes", "ncclProfilerPostPlanWork");
}
ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp*) {
  FailLoudUnfaked("profiler_fakes", "ncclProfilerAddPidToProxyOp");
}
