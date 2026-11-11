/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// src/plugin/profiler.cc's ncclProfilerProxyTraceDumpIfAny, recorded, and the plan events
// hostStreamPlanTask raises, hookable. The rest of that file's floor is still in nccl_stubs.cc and
// comm_fakes.cc.

#ifndef RCCL_TEST_HOST_PROFILER_FAKES_H_
#define RCCL_TEST_HOST_PROFILER_FAKES_H_

#include <functional>
#include <vector>

#include "nccl.h"

struct ncclComm;
struct ncclKernelPlan;

// Every profilerContext passed to ncclProfilerProxyTraceDumpIfAny since the last reset, in order.
std::vector<void*> ProfilerProxyTraceDumps();

// Unset (the default) fails loud.
extern std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStartGroupEvent;
extern std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStopGroupEvent;
extern std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStartTaskEvents;
extern std::function<ncclResult_t(struct ncclKernelPlan*)> g_profilerStopTaskEvents;
extern std::function<ncclResult_t(struct ncclComm*, struct ncclKernelPlan*)> g_profilerPostPlanWork;

void ResetProfilerFakes();

#endif  // RCCL_TEST_HOST_PROFILER_FAKES_H_
