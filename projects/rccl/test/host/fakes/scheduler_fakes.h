/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for externals allgatherv_sched.cc/symmetric_sched.cc reach that nothing else fakes yet.

#ifndef RCCL_TEST_HOST_FAKES_SCHEDULER_FAKES_H_
#define RCCL_TEST_HOST_FAKES_SCHEDULER_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclComm;
struct ncclTaskColl;
struct ncclKernelPlan;
struct ncclKernelPlanBudget;

// enqueue.cc's ncclTestBudget (real seam: tests drive the batch-size stopping condition directly).
extern std::function<bool(struct ncclKernelPlanBudget*, int, ssize_t)> g_testBudget;
extern int g_testBudgetCalls;  // UNDRIVEN

// enqueue.cc's ncclGetAlgoInfo: default fills in tcoll's protocol/channel/warp fields with usable values.
extern std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, int, int, int, ncclSimInfo_t*)>
    g_getAlgoInfo;

// enqueue.cc's ncclPlanSetDefaultKernel: default is a no-op (no real kernel table exists in this binary).
extern std::function<void(struct ncclComm*, struct ncclKernelPlan*)> g_planSetDefaultKernel;

void ResetSchedulerFakes();

#endif  // RCCL_TEST_HOST_FAKES_SCHEDULER_FAKES_H_
