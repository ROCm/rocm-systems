/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for externals allgatherv_sched.cc/symmetric_sched.cc reach that nothing else fakes yet.

#ifndef RCCL_TEST_HOST_FAKES_SCHEDULER_FAKES_H_
#define RCCL_TEST_HOST_FAKES_SCHEDULER_FAKES_H_

#include <cstdint>
#include <functional>

#include "nccl.h"
#include "nccl_common.h"  // ncclFunc_t: an internal type, not part of the public nccl.h API surface

struct ncclComm;
struct ncclTaskColl;
struct ncclKernelPlan;
struct ncclKernelPlanBudget;
struct ncclProxyOp;
enum ncclDevWorkType : uint8_t;

// enqueue.cc's ncclTestBudget (real seam: tests drive the batch-size stopping condition directly).
extern std::function<bool(struct ncclKernelPlanBudget*, int, ssize_t)> g_testBudget;
extern int g_testBudgetCalls;  // UNDRIVEN

// enqueue.cc's ncclGetAlgoInfo: default fills in tcoll's protocol/channel/warp fields with usable values.
extern std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, int, int, int, ncclSimInfo_t*)>
    g_getAlgoInfo;

// enqueue.cc's ncclPlanSetDefaultKernel: default is a no-op (no real kernel table exists in this binary).
extern std::function<void(struct ncclComm*, struct ncclKernelPlan*)> g_planSetDefaultKernel;

// enqueue.cc's ncclAddWorkBatchToPlan: default is a no-op observer.
extern std::function<void(struct ncclComm*, struct ncclKernelPlan*, int, enum ncclDevWorkType, int, uint32_t, int,
                          int, bool)>
    g_addWorkBatchToPlan;

// enqueue.cc's ncclAddProxyOpIfNeeded: default accepts every proxy op.
extern std::function<ncclResult_t(struct ncclComm*, struct ncclKernelPlan*, struct ncclProxyOp*)>
    g_addProxyOpIfNeeded;

// sym_kernels.cc's ncclSymkAvailable: default reports every (func, redOp, dtype, count) as symmetric-eligible.
extern std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t)> g_symkAvailable;

void ResetSchedulerFakes();

#endif  // RCCL_TEST_HOST_FAKES_SCHEDULER_FAKES_H_
