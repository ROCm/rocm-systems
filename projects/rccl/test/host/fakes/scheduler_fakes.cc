/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See scheduler_fakes.h.

#include <unordered_map>

#include "comm.h"
#include "device.h"
#include "enqueue.h"
#include "nccl.h"
#include "sym_kernels.h"
#include "tuning.h"
#include "config/algorithm_registry.h"

#include "fail_loud.h"

// src/enqueue/enqueue.cc
bool ncclTestBudget(struct ncclKernelPlanBudget*, int, ssize_t) {
  FailLoudUnfaked("scheduler_fakes", "ncclTestBudget");
}
ncclResult_t ncclGetAlgoInfo(struct ncclComm*, struct ncclTaskColl*, int, int, int, ncclSimInfo_t*) {
  FailLoudUnfaked("scheduler_fakes", "ncclGetAlgoInfo");
}
void ncclAddWorkBatchToPlan(struct ncclComm*, struct ncclKernelPlan*, int, enum ncclDevWorkType, int, uint32_t, int,
                            int, bool) {
  FailLoudUnfaked("scheduler_fakes", "ncclAddWorkBatchToPlan");
}
void ncclPlanSetDefaultKernel(struct ncclComm*, struct ncclKernelPlan*) {
  FailLoudUnfaked("scheduler_fakes", "ncclPlanSetDefaultKernel");
}
ncclResult_t ncclAddProxyOpIfNeeded(struct ncclComm*, struct ncclKernelPlan*, struct ncclProxyOp*) {
  FailLoudUnfaked("scheduler_fakes", "ncclAddProxyOpIfNeeded");
}
ncclResult_t ncclGetCollNetSupport(struct ncclComm*, struct ncclTaskColl*, int*) {
  FailLoudUnfaked("scheduler_fakes", "ncclGetCollNetSupport");
}
ncclResult_t ncclGetRegBuff(struct ncclComm*, struct ncclTaskColl*, int*) {
  FailLoudUnfaked("scheduler_fakes", "ncclGetRegBuff");
}

// Generated device-function table; empty default matches nccl_stubs.cc's own (a miss returns -1 with a WARN).
std::unordered_map<uint64_t, int> ncclDevFuncNameToId;

// src/sym_kernels.cc
bool ncclSymkAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t) {
  FailLoudUnfaked("scheduler_fakes", "ncclSymkAvailable");
}
int ncclSymkLLKernelMask() { FailLoudUnfaked("scheduler_fakes", "ncclSymkLLKernelMask"); }
int ncclSymkDynamicSmemKernelMask() { FailLoudUnfaked("scheduler_fakes", "ncclSymkDynamicSmemKernelMask"); }
int ncclSymkGetKernelIndex(ncclSymkKernelId, int, ncclDataType_t) {
  FailLoudUnfaked("scheduler_fakes", "ncclSymkGetKernelIndex");
}
const char* ncclSymkKernelIdToString(int) { FailLoudUnfaked("scheduler_fakes", "ncclSymkKernelIdToString"); }
ncclResult_t ncclSymkMakeDevWork(struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork*) {
  FailLoudUnfaked("scheduler_fakes", "ncclSymkMakeDevWork");
}

// Generated kernel tables; one-element placeholders since nothing here calls ncclSymmetricTaskScheduler.
void* ncclSymkKernelList[1] = {nullptr};
void* ncclSymkKernelListProfile[1] = {nullptr};
int ncclSymkKernelMaxDynamicSmem[1] = {0};

// src/config/algorithm_registry.cc
const char* ncclAlgNameForSymk(int) { FailLoudUnfaked("scheduler_fakes", "ncclAlgNameForSymk"); }

// src/tuning/tuning.cc
ncclResult_t ncclTuningCompute(struct ncclTuningInput_t*, struct ncclTuningResult_t*) {
  FailLoudUnfaked("scheduler_fakes", "ncclTuningCompute");
}

// src/plugin/profiler.cc; not fail-loud since "no plugin loaded" is simply true here (matches nccl_stubs.cc).
bool ncclProfilerPluginLoaded(void) { return false; }
