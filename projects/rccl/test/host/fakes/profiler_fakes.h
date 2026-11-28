/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// src/plugin/profiler.cc's ncclProfilerProxyTraceDumpIfAny, recorded, and the events raised by
// ncclEnqueueCheck, {p2p,coll,ceColl}TaskAppend, hostStreamPlanTask, & rcclAddonLaunch, hookable.
// The rest of that file's floor is still in nccl_stubs.cc.

#ifndef RCCL_TEST_HOST_PROFILER_FAKES_H_
#define RCCL_TEST_HOST_PROFILER_FAKES_H_

#include <functional>
#include <vector>

#include "nccl.h"
#include "nccl_common.h"    // nccl_profiler.h is not self-contained
#include "nccl_profiler.h"  // ncclProfilerEventState_t

struct ncclComm;
struct ncclInfo;
struct ncclKernelPlan;
struct ncclCeCollArgs;
struct ncclProfilerEventRef;

// Every profilerContext passed to ncclProfilerProxyTraceDumpIfAny since the last reset, in order.
std::vector<void*> ProfilerProxyTraceDumps();

// ncclProfilerStart/Stop{Group,Coll,P2p}ApiEvent and ncclProfilerRecordGroupApiEventState. Default succeed.
extern std::function<ncclResult_t(struct ncclInfo*, bool)> g_profilerStartGroupApiEvent;
extern std::function<ncclResult_t(ncclProfilerEventState_t)> g_profilerRecordGroupApiEventState;
extern std::function<ncclResult_t()> g_profilerStopGroupApiEvent;
extern std::function<ncclResult_t(struct ncclInfo*, bool)> g_profilerStartCollApiEvent;
extern std::function<ncclResult_t()> g_profilerStopCollApiEvent;
extern std::function<ncclResult_t(struct ncclInfo*, bool)> g_profilerStartP2pApiEvent;
extern std::function<ncclResult_t()> g_profilerStopP2pApiEvent;

// ncclProfilerStart/Stop{Group,Task}Event(s). Default succeed.
extern std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStartGroupEvent;
extern std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStopGroupEvent;
extern std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStartTaskEvents;
extern std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStopTaskEvents;
extern std::function<ncclResult_t(struct ncclComm*, struct ncclKernelPlan*)> g_profilerPostPlanWork;

// ncclProfilerStart/StopAddonEvent. Default succeed.
extern std::function<ncclResult_t(struct ncclInfo*, bool, struct ncclProfilerEventRef*, struct ncclProfilerEventRef*)>
    g_profilerStartAddonEvent;
extern std::function<ncclResult_t(struct ncclProfilerEventRef*, struct ncclProfilerEventRef*)>
    g_profilerStopAddonEvent;

// ncclProfilerStart/StopCeCollEvent. Default succeed.
extern std::function<ncclResult_t(struct ncclComm*, struct ncclCeCollArgs*, hipStream_t)>
    g_profilerStartCeCollEvent;
extern std::function<ncclResult_t(struct ncclComm*, struct ncclCeCollArgs*, hipStream_t)>
    g_profilerStopCeCollEvent;

void ResetProfilerFakes();

#endif  // RCCL_TEST_HOST_PROFILER_FAKES_H_
