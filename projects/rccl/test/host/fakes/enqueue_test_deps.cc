/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See enqueue_test_deps.h. This file holds the reset chain for the
// `rccl-UnitTestsMicroEnqueue` binary and the few stubs that cannot live in a
// shared fakes file; every seam it resets is defined in a fakes file named
// after the production TU that owns the symbol.

#include "enqueue_test_deps.h"

#include "fail_loud.h"
#include "nccl_fakes.h"
#include "rma/rma.h"

// init.cc owns NCCL_PARAM(P2pDisable). enqueue.cc's addP2pToPlan calls it via
// rcclP2pPolicyChannels; wrap_fakes.cc supplies the body for binaries that
// compile rccl_wrap.cc, which this target does not.
int64_t ncclParamP2pDisable() { return g_loadParam("P2P_DISABLE", 0); }
bool ncclP2pUsesMemcpy() {
  const char* value = getenv("NCCL_P2P_USE_CUDA_MEMCPY");
  return value != nullptr && strtoll(value, nullptr, 0) != 0;
}
int64_t rcclParamP2pCeChannelsPerPeer() { return g_loadParam("RCCL_P2P_CE_CHANNELS_PER_PEER", 16); }
int rcclP2pCeChannelsPerPeerCap(int current, int pool, bool isGfx1250, bool useMemcpy, int64_t configuredCap) {
  if (!isGfx1250 || !useMemcpy || configuredCap <= 0 || pool <= 0) return current;
  int cap = pow2Down((int)(configuredCap < pool ? configuredCap : pool));
  if (cap < 1) cap = 1;
  return current < cap ? current : cap;
}

// rma.cc owns scheduleRmaTasksToPlan, but rma_fakes.cc also links into binaries that compile the
// real rma.cc. api_trace.cc owns ncclGetErrorString, and api_trace_fakes.cc does not link here.
#ifndef RCCL_ENQUEUE_TEST_DEPS_OMIT_scheduleRmaTasksToPlan
ncclResult_t scheduleRmaTasksToPlan(struct ncclComm*, struct ncclKernelPlan*) {
  FailLoudUnfaked("enqueue_test_deps", "scheduleRmaTasksToPlan");
}
#endif
const char* ncclGetErrorString(ncclResult_t) { FailLoudUnfaked("enqueue_test_deps", "ncclGetErrorString"); }

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
  ResetProfilerFakes();
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
