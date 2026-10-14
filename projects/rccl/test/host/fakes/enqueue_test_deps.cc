/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See enqueue_test_deps.h. This file holds only the reset chain for the
// `rccl-UnitTestsMicroEnqueue` binary; every seam it resets is defined in a
// fakes file named after the production TU that owns the symbol.

#include "enqueue_test_deps.h"

#include "nccl_fakes.h"

#include "allocator.h"
#include "dev_runtime.h"
#include "device.h"
#include "fail_loud.h"
#include "group.h"
#include "profiler.h"
#include "rccl_common.h"
#include "rma/rma.h"

// init.cc owns NCCL_PARAM(P2pDisable). enqueue.cc's addP2pToPlan calls it via
// rcclP2pPolicyChannels; wrap_fakes.cc supplies the body for binaries that
// compile rccl_wrap.cc, which this target does not.
int64_t ncclParamP2pDisable() { return g_loadParam("P2P_DISABLE", 0); }

// ---- rccl_wrap.cc entry points taskAppend reaches (see enqueue_test_deps.h) ----
static bool DefaultA2aCeScratchInPlace(const ncclComm*, const void*, const void*) { return false; }
static ncclResult_t DefaultA2aInPlaceGuard(const ncclComm*, const void*, const void*, size_t) { return ncclSuccess; }
std::function<bool(const ncclComm*, const void*, const void*)> g_rcclAlltoAllCeScratchInPlace =
    DefaultA2aCeScratchInPlace;
std::function<ncclResult_t(const ncclComm*, const void*, const void*, size_t)> g_rcclAlltoAllInPlaceGuard =
    DefaultA2aInPlaceGuard;
int g_rcclAlltoAllInPlaceGuardCalls = 0;
rcclA2aInPlaceSite g_rcclAlltoAllInPlaceGuardSite = rcclA2aSiteAlltoAll;
bool g_rcclAlltoAllInPlaceGuardCapturing = false;
bool g_rcclAlltoAllInPlaceGuardRegisteredCe = false;
bool g_rcclAlltoAllRegisteredCeRefused = false;
bool g_rcclForceCeScratchEnabled = false;
bool g_rcclNcclAlgoEnvIsSet = false;
bool rcclAlltoAllCeScratchInPlace(const ncclComm* comm, const void* sendbuff, const void* recvbuff) {
  return g_rcclAlltoAllCeScratchInPlace(comm, sendbuff, recvbuff);
}
bool rcclAlltoAllRegisteredCeRefused(const ncclComm*, const void* sendbuff, const void* recvbuff) {
  return g_rcclAlltoAllRegisteredCeRefused && sendbuff == recvbuff;
}
ncclResult_t rcclAlltoAllInPlaceGuard(const ncclComm* comm, const void* sendbuff, const void* recvbuff,
                                      size_t totalBytes, rcclA2aInPlaceSite site, bool capturing, bool registeredCe) {
  ++g_rcclAlltoAllInPlaceGuardCalls;
  g_rcclAlltoAllInPlaceGuardSite = site;
  g_rcclAlltoAllInPlaceGuardCapturing = capturing;
  g_rcclAlltoAllInPlaceGuardRegisteredCe = registeredCe;
  return g_rcclAlltoAllInPlaceGuard(comm, sendbuff, recvbuff, totalBytes);
}
// Arch-default answers with no table: CE AllReduce off; forced CE-Scratch and NCCL_ALGO are seams (default off).
bool rcclForceCeScratchEnabled(const ncclComm*) { return g_rcclForceCeScratchEnabled; }
bool rcclCeAllReduceEnabled(const ncclComm*) { return false; }
bool rcclForceCeAllReduceEnabled(const ncclComm*) { return false; }
size_t rcclCeAr2ShotMax(const ncclComm*) { return 0; }
bool rcclNcclAlgoEnvIsSet() { return g_rcclNcclAlgoEnvIsSet; }
bool rcclAllGatherCeRegisteredWindow(const ncclComm*, size_t, ncclSymRegType_t, bool) { return false; }
// Reached only by the 1-rank, Send/Recv and RMA arms, which no taskAppend test drives.
ncclResult_t ncclLaunchOneRank(void*, void const*, size_t, struct ncclDevRedOpFull, ncclDataType_t, hipStream_t,
                               hipEvent_t, void const*) {
  FailLoudUnfaked("enqueue_test_deps", "ncclLaunchOneRank");
}
ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo*, bool) {
  FailLoudUnfaked("enqueue_test_deps", "ncclProfilerStartP2pApiEvent");
}
ncclResult_t ncclProfilerStopP2pApiEvent() { FailLoudUnfaked("enqueue_test_deps", "ncclProfilerStopP2pApiEvent"); }
bool ncclRmaInitialized(struct ncclComm*) { FailLoudUnfaked("enqueue_test_deps", "ncclRmaInitialized"); }
ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool*, void*, void**) {
  FailLoudUnfaked("enqueue_test_deps", "ncclShadowPoolToHost");
}
bool ncclDevrWinRegEnabled(int, enum ncclDevrRegisterCapability) {
  FailLoudUnfaked("enqueue_test_deps", "ncclDevrWinRegEnabled");
}
thread_local struct ncclComm* ncclGroupCommPreconnectHead = nullptr;

void ResetEnqueueTestDeps() {
  g_rcclAlltoAllCeScratchInPlace = DefaultA2aCeScratchInPlace;
  g_rcclAlltoAllInPlaceGuard = DefaultA2aInPlaceGuard;
  g_rcclAlltoAllInPlaceGuardCalls = 0;
  g_rcclAlltoAllInPlaceGuardSite = rcclA2aSiteAlltoAll;
  g_rcclAlltoAllInPlaceGuardCapturing = false;
  g_rcclAlltoAllInPlaceGuardRegisteredCe = false;
  g_rcclAlltoAllRegisteredCeRefused = false;
  g_rcclForceCeScratchEnabled = false;
  g_rcclNcclAlgoEnvIsSet = false;
  ResetHipFakes();
  ResetNcclFakes();
  ResetNcclStubs();
  ResetCeFakes();
  ResetCommFakes();
  ResetDevRuntimeFakes();
  ResetGroupFakes();
  ResetProxyFakes();
  ResetRcclWrapFakes();
  ResetRecorderFakes();
  ResetRegisterStubs();
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
