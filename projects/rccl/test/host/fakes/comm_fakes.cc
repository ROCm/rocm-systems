/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// See comm_fakes.h. Communicator-lifecycle symbols real in the init unit but
// faked for other micro-test units: the ncclCommSetAsyncError seam, a couple of
// comm/util globals, and the pure-instrumentation recorder / nvtx no-ops.

#include "nccl.h"
#include "comm.h"      // also pulls recorder.h (no include guard)
#include "utils.h"
#include "roctx.h"
#include "profiler.h"

#include "fakes/comm_fakes.h"

#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_commSetAsyncError, ncclCommSetAsyncError);

#undef ASSERT_HOOK_MATCHES_PROD

// --- Controllable seam ----------------------------------------------------
static ncclResult_t DefaultCommSetAsyncError(struct ncclComm*, ncclResult_t) {
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, ncclResult_t)> g_commSetAsyncError =
    DefaultCommSetAsyncError;

ncclResult_t ncclCommSetAsyncError(struct ncclComm* comm, ncclResult_t nextState) {
  return g_commSetAsyncError(comm, nextState);
}

void ResetCommFakes() {
  g_commSetAsyncError = DefaultCommSetAsyncError;
}

// --- Comm / util globals (real in init.cc / utils.cc) ---------------------
enum ncclLaunchMode ncclParamLaunchMode = ncclLaunchModeParallel;

// Per-thread wait signal referenced by the inline MPSC-callback drain helpers
// in utils.h.
thread_local struct ncclThreadSignal ncclThreadSignalLocalInstance = {};

// --- Pure instrumentation (no behaviour to assert) ------------------------
namespace rccl {
Recorder::Recorder() {}
Recorder::~Recorder() {}
Recorder& Recorder::instance() {
  static Recorder inst;
  return inst;
}
ncclResult_t Recorder::record(rcclCall_t, int) { return ncclSuccess; }  // group op
void Recorder::record(int, ncclSimInfo_t*) {}                           // SimulatedGroupEnd
}  // namespace rccl

roctx_scoped_range_in::roctx_scoped_range_in(const char*) noexcept {}
roctx_scoped_range_in::~roctx_scoped_range_in() {}

thread_local ncclProfilerApiState_t ncclProfilerApiState = {};
ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t) { return ncclSuccess; }
ncclResult_t ncclProfilerStopGroupApiEvent() { return ncclSuccess; }
