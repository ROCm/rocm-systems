/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for src/rccl_wrap.cc. Controllable seams first, then the fail-loud floor
// for the entry points no host-only microtest executes.

#include "rccl_wrap_fakes.h"

#include "comm.h"
#include "fail_loud.h"
#include "info.h"

// ===========================================================================
// Controllable seams
// ===========================================================================

// These tuning hooks default to no-ops so a test sees the UNMODIFIED selection.
// Each carries a call counter: a no-op that is never called and a no-op that is
// called look identical otherwise, so a dropped call site would be invisible to
// any future test that cares.
static void DefaultNoOpTune(struct ncclComm*, size_t const&, struct ncclTaskColl*) {}
std::function<void(struct ncclComm*, size_t const&, struct ncclTaskColl*)>
    g_rcclUpdateCollectiveProtocol = DefaultNoOpTune;
std::function<void(struct ncclComm*, size_t const&, struct ncclTaskColl*)>
    g_rcclSetPipelining = DefaultNoOpTune;
int g_rcclUpdateCollectiveProtocolCalls = 0;
int g_rcclSetPipeliningCalls = 0;
int g_rcclUpdateThreadThresholdCalls = 0;
int g_rcclOptThreadBlockSizeCalls = 0;

void rcclUpdateCollectiveProtocol(struct ncclComm* comm, size_t const& nBytes,
                                  struct ncclTaskColl* info) {
  ++g_rcclUpdateCollectiveProtocolCalls;
  g_rcclUpdateCollectiveProtocol(comm, nBytes, info);
}
void rcclSetPipelining(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info) {
  ++g_rcclSetPipeliningCalls;
  g_rcclSetPipelining(comm, nBytes, info);
}
void rcclUpdateThreadThreshold(struct ncclComm*, size_t const&, struct ncclTaskColl*, int&) {
  ++g_rcclUpdateThreadThresholdCalls;
}
void rcclOptThreadBlockSize(struct ncclComm*, struct ncclTaskColl*, size_t, int&) {
  ++g_rcclOptThreadBlockSizeCalls;
}

static ncclResult_t DefaultOverrideChannels(struct ncclComm*, ncclFunc_t, size_t, int&) {
  return ncclSuccess;  // no override -- leaves the caller's nc untouched
}
std::function<ncclResult_t(struct ncclComm*, ncclFunc_t, size_t, int&)>
    g_rcclOverrideChannels = DefaultOverrideChannels;
int g_rcclOverrideChannelsCalls = 0;
ncclResult_t rcclOverrideChannels(struct ncclComm* comm, ncclFunc_t coll, size_t nBytes, int& nc) {
  ++g_rcclOverrideChannelsCalls;
  return g_rcclOverrideChannels(comm, coll, nBytes, nc);
}

// "No override applied" is the honest default (production reads env vars that are
// absent here), but the result is configurable so a test can exercise the
// failure propagation, and counted so "was it consulted?" is answerable.
ncclResult_t g_rcclOverrideAlgorithmResult = ncclSuccess;
ncclResult_t g_rcclOverrideProtocolResult = ncclSuccess;
int g_rcclOverrideAlgorithmCalls = 0;
int g_rcclOverrideProtocolCalls = 0;
ncclResult_t rcclOverrideAlgorithm(const char*[], float (*)[NCCL_NUM_PROTOCOLS],
                                   struct ncclTaskColl*) {
  ++g_rcclOverrideAlgorithmCalls;
  return g_rcclOverrideAlgorithmResult;
}
ncclResult_t rcclOverrideProtocol(const char*[], float (*)[NCCL_NUM_PROTOCOLS],
                                  struct ncclTaskColl*) {
  ++g_rcclOverrideProtocolCalls;
  return g_rcclOverrideProtocolResult;
}

bool g_rcclIsArchSupportedForFunc = true;
bool rcclIsArchSupportedForFunc(struct ncclTaskColl*, char const*) {
  return g_rcclIsArchSupportedForFunc;
}

bool g_rcclCeAllReduceAllowed = false;
bool rcclCeAllReduceAllowed(struct ncclComm*) { return g_rcclCeAllReduceAllowed; }
int g_rcclCeAllReduceGraphLatchTickCalls = 0;
bool g_rcclCeAllReduceGraphLatchTickLastCapturing = false;
void rcclCeAllReduceGraphLatchTick(struct ncclComm*, bool ceCapturing) {
  ++g_rcclCeAllReduceGraphLatchTickCalls;
  g_rcclCeAllReduceGraphLatchTickLastCapturing = ceCapturing;
}

// WARP_SPEED. Defined UNCONDITIONALLY, not behind `#ifdef ENABLE_WARP_SPEED`:
// the fake set must not depend on the same build option as the code it fakes,
// or it goes missing in exactly the configuration that needs it. Unreferenced
// definitions are dropped by --gc-sections when the option is off.
bool g_rcclWarpSpeedSupported = false;
int g_rcclWarpSpeedSupportedCalls = 0;
bool rcclWarpSpeedSupported(struct ncclComm*, struct ncclKernelPlan*) {
  ++g_rcclWarpSpeedSupportedCalls;
  return g_rcclWarpSpeedSupported;
}

// Identity by default, so an ENABLE_WARP_SPEED=ON build computes the same
// channel count as an OFF one and the existing topoGetAlgoInfo expectations hold.
static int DefaultWarpSpeedAdjustChannels(struct ncclComm*, struct ncclTaskColl*, int nc) {
  return nc;
}
std::function<int(struct ncclComm*, struct ncclTaskColl*, int)>
    g_rcclWarpSpeedAdjustChannels = DefaultWarpSpeedAdjustChannels;
int rcclWarpSpeedAdjustChannels(struct ncclComm* comm, struct ncclTaskColl* info, int nc) {
  return g_rcclWarpSpeedAdjustChannels(comm, info, nc);
}

ncclResult_t g_rcclSetWarpSpeedAutoResult = ncclSuccess;
int g_rcclSetWarpSpeedAutoCalls = 0;
ncclResult_t rcclSetWarpSpeedAuto(struct ncclComm*, struct ncclTaskColl*, size_t) {
  ++g_rcclSetWarpSpeedAutoCalls;
  return g_rcclSetWarpSpeedAutoResult;
}

// Leaves the caller's channel count untouched, matching the OFF configuration.
int g_rcclSetWarpSpeedCUsCalls = 0;
void rcclSetWarpSpeedCUs(struct ncclComm*, int, int, int&) { ++g_rcclSetWarpSpeedCUsCalls; }

void ResetRcclWrapFakes() {
  g_rcclUpdateCollectiveProtocol = DefaultNoOpTune;
  g_rcclSetPipelining = DefaultNoOpTune;
  g_rcclOverrideChannels = DefaultOverrideChannels;
  g_rcclOverrideChannelsCalls = 0;
  g_rcclIsArchSupportedForFunc = true;
  g_rcclUpdateCollectiveProtocolCalls = 0;
  g_rcclSetPipeliningCalls = 0;
  g_rcclUpdateThreadThresholdCalls = 0;
  g_rcclOptThreadBlockSizeCalls = 0;
  g_rcclOverrideAlgorithmResult = ncclSuccess;
  g_rcclOverrideProtocolResult = ncclSuccess;
  g_rcclOverrideAlgorithmCalls = 0;
  g_rcclOverrideProtocolCalls = 0;
  g_rcclCeAllReduceAllowed = false;
  g_rcclCeAllReduceGraphLatchTickCalls = 0;
  g_rcclCeAllReduceGraphLatchTickLastCapturing = false;
  g_rcclWarpSpeedSupported = false;
  g_rcclWarpSpeedSupportedCalls = 0;
  g_rcclWarpSpeedAdjustChannels = DefaultWarpSpeedAdjustChannels;
  g_rcclSetWarpSpeedAutoResult = ncclSuccess;
  g_rcclSetWarpSpeedAutoCalls = 0;
  g_rcclSetWarpSpeedCUsCalls = 0;
}

// ===========================================================================
// Fail-loud floor -- rccl_wrap.cc entry points no host-only microtest executes.
// Reaching one aborts AND names itself on stderr, which is the point: an unfaked
// path must be loud. A bare ::abort() here would print nothing.
// ===========================================================================

size_t rcclHierarchicalTempBufferSize(int, bool, bool) {
  FailLoudUnfaked("rccl_wrap_fakes", "rcclHierarchicalTempBufferSize");
}
bool rcclCanUseWarpSpeedAuto(struct ncclComm*, int) {
  FailLoudUnfaked("rccl_wrap_fakes", "rcclCanUseWarpSpeedAuto");
}
ncclResult_t rcclCommSetP2pShiftSize(struct ncclComm*) {
  FailLoudUnfaked("rccl_wrap_fakes", "rcclCommSetP2pShiftSize");
}
int64_t rcclParamWarpSpeedForceEnable() {                      // rccl_wrap.cc:78
  FailLoudUnfaked("rccl_wrap_fakes", "rcclParamWarpSpeedForceEnable");
}
int64_t rcclParamHierarchicalAllGather() {                     // rccl_wrap.cc:704
  FailLoudUnfaked("rccl_wrap_fakes", "rcclParamHierarchicalAllGather");
}
int64_t rcclParamHierarchicalReduceScatter() {                 // rccl_wrap.cc:1357
  FailLoudUnfaked("rccl_wrap_fakes", "rcclParamHierarchicalReduceScatter");
}
