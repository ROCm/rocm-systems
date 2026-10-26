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
#include "rccl_common.h"
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_rcclSelectAllReduce, rcclSelectAllReduce);
ASSERT_HOOK_MATCHES_PROD(g_rcclSelectAllGather, rcclSelectAllGather);
ASSERT_HOOK_MATCHES_PROD(g_rcclSelectReduceScatter, rcclSelectReduceScatter);
ASSERT_HOOK_MATCHES_PROD(g_rcclSelectAlltoAll, rcclSelectAlltoAll);
ASSERT_HOOK_MATCHES_PROD(g_rcclDdaEnabled, rcclDdaEnabled);
ASSERT_HOOK_MATCHES_PROD(g_rcclDdaEntryThreshold, rcclDdaEntryThreshold);
ASSERT_HOOK_MATCHES_PROD(g_rcclGetAlgoName, rcclGetAlgoName);
ASSERT_HOOK_MATCHES_PROD(g_rcclUseAllGatherDirect, rcclUseAllGatherDirect);
ASSERT_HOOK_MATCHES_PROD(g_rcclUseReduceScatterDirect, rcclUseReduceScatterDirect);
#undef ASSERT_HOOK_MATCHES_PROD

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
int g_rcclUpdateCollectiveAlgorithmCalls = 0;
bool g_rcclUpdateCollectiveAlgorithmBeforeProtocol = false;
int g_rcclUpdateCollectiveChannelsCalls = 0;
void rcclUpdateCollectiveAlgorithm(struct ncclComm*, size_t, struct ncclTaskColl*, float (*)[NCCL_NUM_PROTOCOLS]) {
  ++g_rcclUpdateCollectiveAlgorithmCalls;
  // Production requires the algorithm hook to run before the protocol hook.
  g_rcclUpdateCollectiveAlgorithmBeforeProtocol = g_rcclUpdateCollectiveProtocolCalls == 0;
}
void rcclUpdateCollectiveChannels(struct ncclComm*, size_t, const struct ncclTaskColl*, int&) {
  ++g_rcclUpdateCollectiveChannelsCalls;
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

bool g_rcclCeArGraphSafe = false;
bool rcclCeArGraphSafe(struct ncclComm*) { return g_rcclCeArGraphSafe; }
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

int64_t g_rcclParamWarpSpeedForceEnable = 0;  // rccl_wrap.cc:81 RCCL_PARAM default
int g_rcclParamWarpSpeedForceEnableCalls = 0;
int64_t rcclParamWarpSpeedForceEnable() {                      // rccl_wrap.cc:78
  ++g_rcclParamWarpSpeedForceEnableCalls;
  return g_rcclParamWarpSpeedForceEnable;
}

bool g_rcclCanUseWarpSpeedAutoResult = false;
int g_rcclCanUseWarpSpeedAutoCalls = 0;
bool rcclCanUseWarpSpeedAuto(struct ncclComm*, int) {
  ++g_rcclCanUseWarpSpeedAutoCalls;
  return g_rcclCanUseWarpSpeedAutoResult;
}

static size_t DefaultHierarchicalTempBufferSize(int, bool, bool) { return 32 * 1024 * 1024; }
std::function<size_t(int, bool, bool)> g_rcclHierarchicalTempBufferSize = DefaultHierarchicalTempBufferSize;
size_t rcclHierarchicalTempBufferSize(int nNodes, bool allGather, bool reduceScatter) {
  return g_rcclHierarchicalTempBufferSize(nNodes, allGather, reduceScatter);
}
int64_t g_rcclParamHierarchicalAllGather = 1;      // RCCL_PARAM(HierarchicalAllGather) default
int64_t rcclParamHierarchicalAllGather() { return g_rcclParamHierarchicalAllGather; }
int64_t g_rcclParamHierarchicalReduceScatter = 0;  // RCCL_PARAM(HierarchicalReduceScatter) default
int64_t rcclParamHierarchicalReduceScatter() { return g_rcclParamHierarchicalReduceScatter; }
int64_t g_rcclParamCeAllReduce = -1;               // rccl_wrap.cc:65 RCCL_PARAM default
int64_t rcclParamCeAllReduce() { return g_rcclParamCeAllReduce; }
int64_t g_rcclParamCeArStagingBytes = -1;          // rccl_wrap.cc:70 RCCL_PARAM default
int64_t rcclParamCeArStagingBytes() { return g_rcclParamCeArStagingBytes; }

static ncclResult_t DefaultSelectAllReduce(struct ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t,
                                           hipStream_t, bool, bool, struct rcclCollDecision*) {
  FailLoudUnfaked("rccl_wrap_fakes", "rcclSelectAllReduce");
}
std::function<ncclResult_t(struct ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, hipStream_t,
                           bool, bool, struct rcclCollDecision*)>
    g_rcclSelectAllReduce = DefaultSelectAllReduce;
ncclResult_t rcclSelectAllReduce(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype, ncclRedOp_t op, hipStream_t stream, bool query,
                                 bool graphCapturingHint, struct rcclCollDecision* decision) {
  return g_rcclSelectAllReduce(comm, sendbuff, recvbuff, count, datatype, op, stream, query, graphCapturingHint,
                               decision);
}

static ncclResult_t DefaultSelectAllGather(struct ncclComm*, const void*, void*, size_t, ncclDataType_t, hipStream_t,
                                           bool, bool, struct rcclCollDecision*) {
  FailLoudUnfaked("rccl_wrap_fakes", "rcclSelectAllGather");
}
std::function<ncclResult_t(struct ncclComm*, const void*, void*, size_t, ncclDataType_t, hipStream_t, bool, bool,
                           struct rcclCollDecision*)>
    g_rcclSelectAllGather = DefaultSelectAllGather;
ncclResult_t rcclSelectAllGather(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                 ncclDataType_t datatype, hipStream_t stream, bool query, bool graphCapturingHint,
                                 struct rcclCollDecision* decision) {
  return g_rcclSelectAllGather(comm, sendbuff, recvbuff, sendcount, datatype, stream, query, graphCapturingHint,
                               decision);
}

static ncclResult_t DefaultSelectReduceScatter(struct ncclComm*, const void*, void*, size_t, ncclDataType_t,
                                               ncclRedOp_t, bool, struct rcclCollDecision*) {
  FailLoudUnfaked("rccl_wrap_fakes", "rcclSelectReduceScatter");
}
std::function<ncclResult_t(struct ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, bool,
                           struct rcclCollDecision*)>
    g_rcclSelectReduceScatter = DefaultSelectReduceScatter;
ncclResult_t rcclSelectReduceScatter(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                     ncclDataType_t datatype, ncclRedOp_t op, bool query,
                                     struct rcclCollDecision* decision) {
  return g_rcclSelectReduceScatter(comm, sendbuff, recvbuff, recvcount, datatype, op, query, decision);
}

static ncclResult_t DefaultSelectAlltoAll(struct ncclComm*, const void*, void*, size_t, ncclDataType_t, hipStream_t,
                                          bool, bool, struct rcclCollDecision*) {
  FailLoudUnfaked("rccl_wrap_fakes", "rcclSelectAlltoAll");
}
std::function<ncclResult_t(struct ncclComm*, const void*, void*, size_t, ncclDataType_t, hipStream_t, bool, bool,
                           struct rcclCollDecision*)>
    g_rcclSelectAlltoAll = DefaultSelectAlltoAll;
ncclResult_t rcclSelectAlltoAll(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                ncclDataType_t datatype, hipStream_t stream, bool query, bool graphCapturingHint,
                                struct rcclCollDecision* decision) {
  return g_rcclSelectAlltoAll(comm, sendbuff, recvbuff, count, datatype, stream, query, graphCapturingHint, decision);
}

static bool DefaultDdaEnabled(const struct ncclComm*, size_t, size_t, bool, const char*) {
  FailLoudUnfaked("rccl_wrap_fakes", "rcclDdaEnabled");
}
std::function<bool(const struct ncclComm*, size_t, size_t, bool, const char*)> g_rcclDdaEnabled = DefaultDdaEnabled;
bool rcclDdaEnabled(const struct ncclComm* comm, size_t totalBytes, size_t threshold, bool query, const char* prefix) {
  return g_rcclDdaEnabled(comm, totalBytes, threshold, query, prefix);
}

static size_t DefaultDdaEntryThreshold(const struct ncclComm*, ncclFunc_t) {
  FailLoudUnfaked("rccl_wrap_fakes", "rcclDdaEntryThreshold");
}
std::function<size_t(const struct ncclComm*, ncclFunc_t)> g_rcclDdaEntryThreshold = DefaultDdaEntryThreshold;
size_t rcclDdaEntryThreshold(const struct ncclComm* comm, ncclFunc_t func) {
  return g_rcclDdaEntryThreshold(comm, func);
}

static ncclResult_t DefaultGetAlgoName(int, const char**) { FailLoudUnfaked("rccl_wrap_fakes", "rcclGetAlgoName"); }
std::function<ncclResult_t(int, const char**)> g_rcclGetAlgoName = DefaultGetAlgoName;
ncclResult_t rcclGetAlgoName(int algo, const char** algoName) { return g_rcclGetAlgoName(algo, algoName); }

static bool DefaultUseAllGatherDirect(struct ncclComm*, size_t&) {
  FailLoudUnfaked("rccl_wrap_fakes", "rcclUseAllGatherDirect");
}
std::function<bool(struct ncclComm*, size_t&)> g_rcclUseAllGatherDirect = DefaultUseAllGatherDirect;
bool rcclUseAllGatherDirect(struct ncclComm* comm, size_t& msgSize) { return g_rcclUseAllGatherDirect(comm, msgSize); }

static bool DefaultUseReduceScatterDirect(struct ncclComm*, size_t&) {
  FailLoudUnfaked("rccl_wrap_fakes", "rcclUseReduceScatterDirect");
}
std::function<bool(struct ncclComm*, size_t&)> g_rcclUseReduceScatterDirect = DefaultUseReduceScatterDirect;
bool rcclUseReduceScatterDirect(struct ncclComm* comm, size_t& msgSize) {
  return g_rcclUseReduceScatterDirect(comm, msgSize);
}

int64_t g_rcclParamDdaEnable = 1;  // rccl_wrap.cc:98 RCCL_PARAM default
int64_t rcclParamDdaEnable() { return g_rcclParamDdaEnable; }

void ResetRcclWrapFakes() {
  g_rcclUpdateCollectiveProtocol = DefaultNoOpTune;
  g_rcclSetPipelining = DefaultNoOpTune;
  g_rcclOverrideChannels = DefaultOverrideChannels;
  g_rcclOverrideChannelsCalls = 0;
  g_rcclIsArchSupportedForFunc = true;
  g_rcclParamDirectReduceScatterThreshold = 8388608;
  g_rcclHierarchicalTempBufferSize = DefaultHierarchicalTempBufferSize;
  g_rcclParamHierarchicalAllGather = 1;
  g_rcclParamHierarchicalReduceScatter = 0;
  g_rcclParamCeAllReduce = -1;
  g_rcclParamCeArStagingBytes = -1;
  g_rcclUpdateCollectiveProtocolCalls = 0;
  g_rcclUpdateCollectiveAlgorithmCalls = 0;
  g_rcclUpdateCollectiveAlgorithmBeforeProtocol = false;
  g_rcclUpdateCollectiveChannelsCalls = 0;
  g_rcclSetPipeliningCalls = 0;
  g_rcclUpdateThreadThresholdCalls = 0;
  g_rcclOptThreadBlockSizeCalls = 0;
  g_rcclOverrideAlgorithmResult = ncclSuccess;
  g_rcclOverrideProtocolResult = ncclSuccess;
  g_rcclOverrideAlgorithmCalls = 0;
  g_rcclOverrideProtocolCalls = 0;
  g_rcclCeArGraphSafe = false;
  g_rcclCeAllReduceGraphLatchTickCalls = 0;
  g_rcclCeAllReduceGraphLatchTickLastCapturing = false;
  g_rcclWarpSpeedSupported = false;
  g_rcclWarpSpeedSupportedCalls = 0;
  g_rcclWarpSpeedAdjustChannels = DefaultWarpSpeedAdjustChannels;
  g_rcclSetWarpSpeedAutoResult = ncclSuccess;
  g_rcclSetWarpSpeedAutoCalls = 0;
  g_rcclSetWarpSpeedCUsCalls = 0;
  g_rcclParamWarpSpeedForceEnable = 0;
  g_rcclParamWarpSpeedForceEnableCalls = 0;
  g_rcclCanUseWarpSpeedAutoResult = false;
  g_rcclCanUseWarpSpeedAutoCalls = 0;
  g_validHsaScratch = true;
  g_lastHsaScratchEnv = nullptr;
  g_firmwareVersion = 0;
  g_rcclSelectAllReduce = DefaultSelectAllReduce;
  g_rcclSelectAllGather = DefaultSelectAllGather;
  g_rcclSelectReduceScatter = DefaultSelectReduceScatter;
  g_rcclSelectAlltoAll = DefaultSelectAlltoAll;
  g_rcclDdaEnabled = DefaultDdaEnabled;
  g_rcclDdaEntryThreshold = DefaultDdaEntryThreshold;
  g_rcclGetAlgoName = DefaultGetAlgoName;
  g_rcclUseAllGatherDirect = DefaultUseAllGatherDirect;
  g_rcclUseReduceScatterDirect = DefaultUseReduceScatterDirect;
  g_rcclParamDdaEnable = 1;
}

bool g_validHsaScratch = true;
// Records the argument so a test can observe that checkHsaEnvSetting actually read the environment.
const char* g_lastHsaScratchEnv = nullptr;
bool validHsaScratchEnvSetting(const char* hsaScratchEnv, int /*hipRuntimeVersion*/,
                               int /*firmwareVersion*/, const char* /*gcnArchName*/) {
  g_lastHsaScratchEnv = hsaScratchEnv;
  return g_validHsaScratch;
}

int g_firmwareVersion = 0;
int getFirmwareVersion() { return g_firmwareVersion; }

void rcclSetDefaultBuffSizes(struct ncclComm*, int* defaults) {
  defaults[0] = 1 << 18;  // LL
  defaults[1] = 1 << 18;  // LL128
  defaults[2] = 1 << 22;  // SIMPLE
}
void rcclSetP2pNetChunkSize(struct ncclComm*, int& sz) { sz = 1 << 17; }

// ===========================================================================
// Fail-loud floor -- rccl_wrap.cc entry points no host-only microtest executes.
// Reaching one aborts AND names itself on stderr, which is the point: an unfaked
// path must be loud. A bare ::abort() here would print nothing.
// ===========================================================================

ncclResult_t rcclCommSetP2pShiftSize(struct ncclComm*) {
  FailLoudUnfaked("rccl_wrap_fakes", "rcclCommSetP2pShiftSize");
}
int64_t rcclParamHierarchicalLazyInit() {
  FailLoudUnfaked("rccl_wrap_fakes", "rcclParamHierarchicalLazyInit");
}
void rcclSetPxn(struct ncclComm*, int&) { FailLoudUnfaked("rccl_wrap_fakes", "rcclSetPxn"); }
void rcclSetCeAllReduceFastPath(struct ncclComm*) {
  FailLoudUnfaked("rccl_wrap_fakes", "rcclSetCeAllReduceFastPath");
}
int64_t g_rcclParamDirectReduceScatterThreshold = 8388608;     // rccl_wrap.cc:51 default
int64_t rcclParamDirectReduceScatterThreshold() {              // rccl_wrap.cc:51
  return g_rcclParamDirectReduceScatterThreshold;
}
