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

ASSERT_HOOK_MATCHES_PROD(g_profilerStartGroupApiEvent, ncclProfilerStartGroupApiEvent);
ASSERT_HOOK_MATCHES_PROD(g_profilerRecordGroupApiEventState, ncclProfilerRecordGroupApiEventState);
ASSERT_HOOK_MATCHES_PROD(g_profilerStopGroupApiEvent, ncclProfilerStopGroupApiEvent);
ASSERT_HOOK_MATCHES_PROD(g_profilerStartCollApiEvent, ncclProfilerStartCollApiEvent);
ASSERT_HOOK_MATCHES_PROD(g_profilerStopCollApiEvent, ncclProfilerStopCollApiEvent);
ASSERT_HOOK_MATCHES_PROD(g_profilerStartP2pApiEvent, ncclProfilerStartP2pApiEvent);
ASSERT_HOOK_MATCHES_PROD(g_profilerStopP2pApiEvent, ncclProfilerStopP2pApiEvent);
ASSERT_HOOK_MATCHES_PROD(g_profilerStartGroupEvent, ncclProfilerStartGroupEvent);
ASSERT_HOOK_MATCHES_PROD(g_profilerStopGroupEvent, ncclProfilerStopGroupEvent);
ASSERT_HOOK_MATCHES_PROD(g_profilerStartTaskEvents, ncclProfilerStartTaskEvents);
ASSERT_HOOK_MATCHES_PROD(g_profilerStopTaskEvents, ncclProfilerStopTaskEvents);
ASSERT_HOOK_MATCHES_PROD(g_profilerPostPlanWork, ncclProfilerPostPlanWork);
ASSERT_HOOK_MATCHES_PROD(g_profilerStartAddonEvent, ncclProfilerStartAddonEvent);
ASSERT_HOOK_MATCHES_PROD(g_profilerStopAddonEvent, ncclProfilerStopAddonEvent);
ASSERT_HOOK_MATCHES_PROD(g_profilerStartCeCollEvent, ncclProfilerStartCeCollEvent);
ASSERT_HOOK_MATCHES_PROD(g_profilerStopCeCollEvent, ncclProfilerStopCeCollEvent);
#undef ASSERT_HOOK_MATCHES_PROD

namespace {
std::mutex g_profilerMutex;  // comm teardown can run on any thread
std::vector<void*> g_proxyTraceDumps;

// Generic lambda can be assigned to all the std::function hooks below
// Unlike a template function, does not need explicit template arguments.
constexpr auto DefaultProfilerLambda = [](auto&&...) -> ncclResult_t { return ncclSuccess; };
}  // namespace

std::vector<void*> ProfilerProxyTraceDumps() {
  std::lock_guard<std::mutex> lock(g_profilerMutex);
  return g_proxyTraceDumps;
}

thread_local ncclProfilerApiState_t ncclProfilerApiState = {};

std::function<ncclResult_t(struct ncclInfo*, bool)> g_profilerStartGroupApiEvent = DefaultProfilerLambda;
std::function<ncclResult_t(ncclProfilerEventState_t)> g_profilerRecordGroupApiEventState = DefaultProfilerLambda;
std::function<ncclResult_t()> g_profilerStopGroupApiEvent = DefaultProfilerLambda;
std::function<ncclResult_t(struct ncclInfo*, bool)> g_profilerStartCollApiEvent = DefaultProfilerLambda;
std::function<ncclResult_t()> g_profilerStopCollApiEvent = DefaultProfilerLambda;
std::function<ncclResult_t(struct ncclInfo*, bool)> g_profilerStartP2pApiEvent = DefaultProfilerLambda;
std::function<ncclResult_t()> g_profilerStopP2pApiEvent = DefaultProfilerLambda;

std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStartGroupEvent = DefaultProfilerLambda;
std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStopGroupEvent = DefaultProfilerLambda;
std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStartTaskEvents = DefaultProfilerLambda;
std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStopTaskEvents = DefaultProfilerLambda;
std::function<ncclResult_t(struct ncclComm*, struct ncclKernelPlan*)> g_profilerPostPlanWork = DefaultProfilerLambda;

std::function<ncclResult_t(struct ncclInfo*, bool, struct ncclProfilerEventRef*, struct ncclProfilerEventRef*)>
    g_profilerStartAddonEvent = DefaultProfilerLambda;
std::function<ncclResult_t(struct ncclProfilerEventRef*, struct ncclProfilerEventRef*)>
    g_profilerStopAddonEvent = DefaultProfilerLambda;

std::function<ncclResult_t(struct ncclComm*, struct ncclCeCollArgs*, hipStream_t)>
    g_profilerStartCeCollEvent = DefaultProfilerLambda;
std::function<ncclResult_t(struct ncclComm*, struct ncclCeCollArgs*, hipStream_t)>
    g_profilerStopCeCollEvent = DefaultProfilerLambda;

void ResetProfilerFakes() {
  std::lock_guard<std::mutex> lock(g_profilerMutex);
  g_proxyTraceDumps.clear();
  g_profilerStartGroupApiEvent = DefaultProfilerLambda;
  g_profilerRecordGroupApiEventState = DefaultProfilerLambda;
  g_profilerStopGroupApiEvent = DefaultProfilerLambda;
  g_profilerStartCollApiEvent = DefaultProfilerLambda;
  g_profilerStopCollApiEvent = DefaultProfilerLambda;
  g_profilerStartP2pApiEvent = DefaultProfilerLambda;
  g_profilerStopP2pApiEvent = DefaultProfilerLambda;
  g_profilerStartGroupEvent = DefaultProfilerLambda;
  g_profilerStopGroupEvent = DefaultProfilerLambda;
  g_profilerStartTaskEvents = DefaultProfilerLambda;
  g_profilerStopTaskEvents = DefaultProfilerLambda;
  g_profilerPostPlanWork = DefaultProfilerLambda;
  g_profilerStartAddonEvent = DefaultProfilerLambda;
  g_profilerStopAddonEvent = DefaultProfilerLambda;
  g_profilerStartCeCollEvent = DefaultProfilerLambda;
  g_profilerStopCeCollEvent = DefaultProfilerLambda;
}

void ncclProfilerProxyTraceDumpIfAny(void* profilerContext) {
  std::lock_guard<std::mutex> lock(g_profilerMutex);
  g_proxyTraceDumps.push_back(profilerContext);
}

ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  return g_profilerStartGroupApiEvent(info, isGraphCaptured);
}
ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t eState) {
  return g_profilerRecordGroupApiEventState(eState);
}
ncclResult_t ncclProfilerStopGroupApiEvent() {
  return g_profilerStopGroupApiEvent();
}
ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  return g_profilerStartCollApiEvent(info, isGraphCaptured);
}
ncclResult_t ncclProfilerStopCollApiEvent() {
  return g_profilerStopCollApiEvent();
}
ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  return g_profilerStartP2pApiEvent(info, isGraphCaptured);
}
ncclResult_t ncclProfilerStopP2pApiEvent() {
  return g_profilerStopP2pApiEvent();
}

ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan) {
  return g_profilerStartGroupEvent(plan);
}
ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan) {
  return g_profilerStopGroupEvent(plan);
}
ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan) {
  return g_profilerStartTaskEvents(plan);
}
ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan) {
  return g_profilerStopTaskEvents(plan);
}
ncclResult_t ncclProfilerPostPlanWork(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  return g_profilerPostPlanWork(comm, plan);
}

ncclResult_t ncclProfilerStartAddonEvent(struct ncclInfo* info, bool isGraphCaptured,
                                         struct ncclProfilerEventRef* groupEventHandle,
                                         struct ncclProfilerEventRef* collEventHandle) {
  return g_profilerStartAddonEvent(info, isGraphCaptured, groupEventHandle, collEventHandle);
}
ncclResult_t ncclProfilerStopAddonEvent(struct ncclProfilerEventRef* groupEventHandle,
                                        struct ncclProfilerEventRef* collEventHandle) {
  return g_profilerStopAddonEvent(groupEventHandle, collEventHandle);
}

ncclResult_t ncclProfilerStartCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, hipStream_t stream) {
  return g_profilerStartCeCollEvent(comm, args, stream);
}
ncclResult_t ncclProfilerStopCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, hipStream_t stream) {
  return g_profilerStopCeCollEvent(comm, args, stream);
}
ncclResult_t ncclProfilerStartCeSyncEvent(struct ncclComm*, struct ncclCeCollArgs*, hipStream_t,
                                          struct ncclProfilerEventRef* handle) {
  *handle = ncclProfilerEventRef{};
  return ncclSuccess;
}
ncclResult_t ncclProfilerStopCeSyncEvent(struct ncclComm*, struct ncclProfilerEventRef*, hipStream_t) {
  return ncclSuccess;
}
ncclResult_t ncclProfilerStartCeBatchEvent(struct ncclComm*, struct ncclCeCollArgs*, struct ncclCeBatchOpsParams*,
                                           hipStream_t, struct ncclProfilerEventRef* handle) {
  *handle = ncclProfilerEventRef{};
  return ncclSuccess;
}
ncclResult_t ncclProfilerStopCeBatchEvent(struct ncclComm*, struct ncclProfilerEventRef*, hipStream_t) {
  return ncclSuccess;
}

ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp*) {
  FailLoudUnfaked("profiler_fakes", "ncclProfilerAddPidToProxyOp");
}
