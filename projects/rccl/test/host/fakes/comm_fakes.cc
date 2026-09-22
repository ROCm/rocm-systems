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
#include "nccl_fakes.h"  // g_loadParam, for the NCCL_PARAM defaults this stands in for
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

// The per-launch resource params init.cc declares. collTaskAppend resolves the
// task's CTA counts through them (env > per-call > comm), so they are referenced
// from outside init.cc and the redirected NCCL_PARAM does not cover them. Same
// env names and defaults as init.cc, routed through g_loadParam so a fixture can
// script the env-override arm of NCCL_CONFIG_SET.
int64_t ncclParamMinCTAs() { return g_loadParam("MIN_CTAS", NCCL_CONFIG_UNDEF_INT); }          // init.cc:2667
int64_t ncclParamMaxCTAs() { return g_loadParam("MAX_CTAS", NCCL_CONFIG_UNDEF_INT); }          // init.cc:2666
int64_t ncclParamNvlsChannels() { return g_loadParam("NVLS_NCHANNELS", NCCL_CONFIG_UNDEF_INT); }  // init.cc:135
int64_t ncclParamCGAClusterSize() { return g_loadParam("CGA_CLUSTER_SIZE", NCCL_CONFIG_UNDEF_INT); }  // init.cc:2664

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
ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo*, bool) { return ncclSuccess; }
ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t) { return ncclSuccess; }
ncclResult_t ncclProfilerStopGroupApiEvent() { return ncclSuccess; }
ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo*, bool) { return ncclSuccess; }
ncclResult_t ncclProfilerStopCollApiEvent() { return ncclSuccess; }
