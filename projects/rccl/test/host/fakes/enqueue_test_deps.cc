/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See enqueue_test_deps.h. This file holds only the reset chain for the
// `rccl-UnitTestsMicroEnqueue` binary; every seam it resets is defined in a
// fakes file named after the production TU that owns the symbol.

#include "enqueue_test_deps.h"

#include "fail_loud.h"
#include "nccl_fakes.h"
#include "profiler.h"
#include "rma/rma.h"

// init.cc owns NCCL_PARAM(P2pDisable). enqueue.cc's addP2pToPlan calls it via
// rcclP2pPolicyChannels; wrap_fakes.cc supplies the body for binaries that
// compile rccl_wrap.cc, which this target does not.
int64_t ncclParamP2pDisable() { return g_loadParam("P2P_DISABLE", 0); }

// ncclLaunchPrepare links these in, but the tests that call it return before RMA scheduling and
// before any plan is launched. Defined here rather than in profiler_fakes.cc / rma_fakes.cc, which
// share binaries with the real profiler.cc and rma code.
ncclResult_t scheduleRmaTasksToPlan(struct ncclComm*, struct ncclKernelPlan*) {
  FailLoudUnfaked("enqueue_test_deps", "scheduleRmaTasksToPlan");
}
ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan*) {
  FailLoudUnfaked("enqueue_test_deps", "ncclProfilerStartGroupEvent");
}
ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan*) {
  FailLoudUnfaked("enqueue_test_deps", "ncclProfilerStopGroupEvent");
}
ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan*) {
  FailLoudUnfaked("enqueue_test_deps", "ncclProfilerStartTaskEvents");
}
ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan*) {
  FailLoudUnfaked("enqueue_test_deps", "ncclProfilerStopTaskEvents");
}
ncclResult_t ncclProfilerPostPlanWork(struct ncclComm*, struct ncclKernelPlan*) {
  FailLoudUnfaked("enqueue_test_deps", "ncclProfilerPostPlanWork");
}
ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp*) {
  FailLoudUnfaked("enqueue_test_deps", "ncclProfilerAddPidToProxyOp");
}
const char* ncclGetErrorString(ncclResult_t) { FailLoudUnfaked("enqueue_test_deps", "ncclGetErrorString"); }
hipError_t hipLaunchHostFunc(hipStream_t, hipHostFn_t, void*) {
  FailLoudUnfaked("enqueue_test_deps", "hipLaunchHostFunc");
}

void ResetEnqueueTestDeps() {
  ResetAllocatorFakes();
  ResetHipFakes();
  ResetNcclFakes();
  ResetNcclStubs();
  ResetCeFakes();
  ResetCommFakes();
  ResetDevRuntimeFakes();
  ResetGroupFakes();
  ResetMgmtTaskEnqFakes();
  ResetProxyFakes();
  ResetRcclWrapFakes();
  ResetRecorderFakes();
  ResetRegisterStubs();
  ResetRmaFakes();
  ResetSchedStubs();
  ResetStrongStreamStubs();
  ResetSymKernelsFakes();
  ResetTransportStubs();
  ResetTuningFakes();
  ResetEnvFakes();

  // enqueue.cc has raw libc getenv() call sites, not just ncclGetEnv ones
  // (topoGetAlgoInfo:2687-2688 reads NCCL_PROTO and NCCL_ALGO that way). The
  // interposer in env_fakes.cc only intercepts names the map KNOWS; an unmapped
  // name falls through to the real environment. Mapping them absent is what
  // makes the suite hermetic against the CI machine's ambient environment
  // rather than merely capable of being made so.
  SetMicroEnvAbsent("NCCL_PROTO");
  SetMicroEnvAbsent("NCCL_ALGO");
}
