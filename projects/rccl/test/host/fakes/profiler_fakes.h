/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// src/plugin/profiler.cc's ncclProfilerProxyTraceDumpIfAny, recorded. The rest of that file's floor is still in
// nccl_stubs.cc and comm_fakes.cc.

#ifndef RCCL_TEST_HOST_PROFILER_FAKES_H_
#define RCCL_TEST_HOST_PROFILER_FAKES_H_

#include <functional>
#include <vector>

#include "nccl.h"

struct ncclComm;
struct ncclKernelPlan;
struct ncclProxyOp;

// Every profilerContext passed to ncclProfilerProxyTraceDumpIfAny since the last reset, in order.
std::vector<void*> ProfilerProxyTraceDumps();

// The plan-event entry points src/plugin/profiler.cc owns. Each fails loud until a test installs
// one, so a binary that never drives a plan through the profiler keeps the unfaked diagnostic.
extern std::function<ncclResult_t(struct ncclKernelPlan*)> g_ncclProfilerStartGroupEvent;
extern std::function<ncclResult_t(struct ncclKernelPlan*)> g_ncclProfilerStopGroupEvent;
extern std::function<ncclResult_t(struct ncclKernelPlan*)> g_ncclProfilerStartTaskEvents;
extern std::function<ncclResult_t(struct ncclKernelPlan*)> g_ncclProfilerStopTaskEvents;
extern std::function<ncclResult_t(struct ncclComm*, struct ncclKernelPlan*)> g_ncclProfilerPostPlanWork;
extern std::function<ncclResult_t(struct ncclProxyOp*)> g_ncclProfilerAddPidToProxyOp;

void ResetProfilerFakes();

#endif  // RCCL_TEST_HOST_PROFILER_FAKES_H_
