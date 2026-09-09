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

ncclResult_t g_commEnsureReadyResult = ncclSuccess;
ncclResult_t ncclCommEnsureReady(struct ncclComm*) { return g_commEnsureReadyResult; }

void ResetCommFakes() {
  g_commSetAsyncError = DefaultCommSetAsyncError;
  g_commEnsureReadyResult = ncclSuccess;
}

// --- Comm globals (real in init.cc) ---------------------------------------
enum ncclLaunchMode ncclParamLaunchMode = ncclLaunchModeParallel;

// ncclThreadSignalLocalInstance is owned by src/misc/utils.cc, not init.cc, and
// lives in utils_fakes.cc: the init and enqueue targets compile the real
// utils.cc as an oracle TU, so a copy here is a duplicate symbol for them.

// --- Pure instrumentation (no behaviour to assert) ------------------------
// rccl::Recorder lives in recorder_fakes.cc: it was defined identically here, in
// init_fakes.cc and in enqueue_fakes.cc, differing only in which record()
// overloads each target referenced.

roctx_scoped_range_in::roctx_scoped_range_in(const char*) noexcept {}
roctx_scoped_range_in::~roctx_scoped_range_in() {}

thread_local ncclProfilerApiState_t ncclProfilerApiState = {};
ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t) { return ncclSuccess; }
ncclResult_t ncclProfilerStopGroupApiEvent() { return ncclSuccess; }
