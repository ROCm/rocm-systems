/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Enqueue-only fake seams + fail-loud stub floor for `rccl-UnitTestsMicroEnqueue`.
//
// This file holds the CONTROLLABLE seams: a std::function or result variable a
// test drives. The fail-loud ::abort() stub floor that satisfies enqueue.cc's
// remaining link closure lives in enqueue_stubs.cc; the only ::abort()s here are
// null-argument checks on arguments production never passes as null. Benign
// teardown paths return ncclSuccess, because a happy-path test reaches them.

#include "enqueue_fakes.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

#include "comm.h"
#include "device.h"
#include "info.h"
#include "nccl.h"
#include "proxy.h"
// NOT "recorder.h": comm.h already pulls in one copy, and including it again
// here re-declares the whole rcclCall_t enumerator list -> "redefinition of
// enumerator rrBroadcast". init_fakes.h documents the same collision.
// comm.h's copy already gives us rccl::Recorder and rcclCall_t.

// ===========================================================================
// Tier 1 -- controllable seams
// ===========================================================================

// Defaults to ncclSystemError, NOT success. Production wraps this in NCCLCHECK,
// so the effect is that the FIRST eligible lookup aborts updateCollCostTable with
// an error -- not that the table is left fully IGNOREd. That is deliberate: an
// unscripted topology query must fail loudly rather than quietly succeed, because
// a success default would make every (algo, proto) pair look selectable and any
// test that forgot to script a time would silently exercise the wrong path.
static ncclResult_t DefaultTopoGetAlgoTime(struct ncclComm*, int, int, int, size_t, int, float*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ncclComm*, int, int, int, size_t, int, float*)>
    g_topoGetAlgoTime = DefaultTopoGetAlgoTime;
int g_topoGetAlgoTimeCalls = 0;

ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol,
                                 size_t nBytes, int numPipeOps, float* time) {
  ++g_topoGetAlgoTimeCalls;
  return g_topoGetAlgoTime(comm, coll, algorithm, protocol, nBytes, numPipeOps, time);
}

int64_t g_paramMinNchannels = 0;
int64_t g_paramMaxNchannels = MAXCHANNELS;
int64_t ncclParamMinNchannels() { return g_paramMinNchannels; }
int64_t ncclParamMaxNchannels() { return g_paramMaxNchannels; }

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

// rcclEffectiveP2pBatchEnable:1174 reaches this only on the gfx950 arm; the
// shared nccl_stubs.cc copy is fail-loud, so this target omits it via
// RCCL_STUBS_OMIT_rcclUseAinic and supplies the value from
// fakes/enqueue_stub_overrides.cc.
// g_rcclUseAinic / rcclUseAinic live in fakes/enqueue_stub_overrides.cc, next to
// the nccl_stubs.cc omission they replace.

// Registration / window seams.
// NOTE: ncclRegLocalIsValid is NOT defined here -- nccl_fakes.cc:184 already
// provides it for the whole microtest family. Defining it again is a duplicate
// symbol at link time. Same for ncclDevrFindWindow (nccl_stubs.cc:48, fail-loud)
// and ncclStreamWaitStream (nccl_fakes.cc:293). If a test needs to drive one,
// upgrade the shared fake rather than shadowing it here.
bool g_devrWindowIsMultiSegment = false;
bool g_devrWindowHasSysmemSegment = false;
bool ncclDevrWindowIsMultiSegment(struct ncclDevrWindow*) { return g_devrWindowIsMultiSegment; }
bool ncclDevrWindowHasSysmemSegment(struct ncclDevrWindow*) { return g_devrWindowHasSysmemSegment; }

bool g_ceImplemented = false;
bool g_ceAvailable = false;
bool g_ceScratchAvailable = false;
bool g_hierCeAvailable = false;
bool g_rcclCeAllReduceAllowed = false;
bool ncclCeImplemented(ncclFunc_t, int, ncclDataType_t) { return g_ceImplemented; }
bool ncclCeAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) {
  return g_ceAvailable;
}
bool ncclCeScratchAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) {
  return g_ceScratchAvailable;
}
bool ncclHierCeAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) {
  return g_hierCeAvailable;
}
bool rcclCeAllReduceAllowed(struct ncclComm*) { return g_rcclCeAllReduceAllowed; }
int g_rcclCeAllReduceGraphLatchTickCalls = 0;
bool g_rcclCeAllReduceGraphLatchTickLastCapturing = false;
void rcclCeAllReduceGraphLatchTick(struct ncclComm*, bool ceCapturing) {
  ++g_rcclCeAllReduceGraphLatchTickCalls;
  g_rcclCeAllReduceGraphLatchTickLastCapturing = ceCapturing;
}

// updateCollCostTable:2527 reads NCCL_PROTO through this, behind a function-local
// `static bool userProtoInputCached` -- so only the FIRST call in the process can
// observe a value. Tests must not depend on the env, and cannot re-script it;
// see the note on that latch beside updateCollCostTable in enqueue-test.cc.
std::unordered_map<std::string, std::string> g_enqEnv;
const char* ncclGetEnv(const char* name) {
  if (!name) return nullptr;
  auto it = g_enqEnv.find(name);
  return it == g_enqEnv.end() ? nullptr : it->second.c_str();
}

// ncclCommEnsureReady: the RedOp create path joins the init thread through this.
// Controllable so a test can exercise the not-ready rejection.
ncclResult_t g_commEnsureReadyResult = ncclSuccess;
ncclResult_t ncclCommEnsureReady(struct ncclComm*) { return g_commEnsureReadyResult; }

// Recorder is pure instrumentation -> no-op fake, but the RESULT of the
// RedOpCreate record() call is checked by production (NCCLCHECK), so it gets a
// seam. Only the overloads enqueue.cc actually reaches are defined here.
namespace rccl {
Recorder::Recorder() {}
Recorder::~Recorder() {}
Recorder& Recorder::instance() {
  static Recorder inst;
  return inst;
}
// recorder.h:128 declares ONE overload with defaulted trailing args, which both
// RedOpCreate (6 args) and RedOpDestroy (3 args) call. Defining a second 3-arg
// version here would be a redefinition, not an overload.
ncclResult_t Recorder::record(rcclCall_t, ncclRedOp_t, ncclComm_t, ncclDataType_t,
                              ncclScalarResidence_t, void*) {
  return g_recorderResult;
}
}  // namespace rccl
ncclResult_t g_recorderResult = ncclSuccess;

// ncclProxySaveOp uses a "justInquire" protocol: the caller asks whether the op
// matters BEFORE allocating for it, so the out-param -- not the return code --
// is what decides whether addProxyOpIfNeeded enqueues.
ncclResult_t g_proxySaveOpResult = ncclSuccess;
int g_proxySaveOpCalls = 0;
bool g_proxySaveOpJustInquire = false;
struct ncclComm* g_proxySaveOpLastComm = nullptr;
struct ncclProxyOp* g_proxySaveOpLastOp = nullptr;
int g_proxySaveOpLastChannelId = -1;
uint64_t g_proxySaveOpLastOpCount = 0;
bool g_proxySaveOpSawJustInquireIn = false;

// Records its arguments. Without this, addProxyOpIfNeeded could hand the
// transport the wrong communicator or a different op and every proxy test would
// still pass -- the queued copy is made from the CALLER's op, not from whatever
// was inquired about, so asserting on the queue alone proves nothing.
ncclResult_t ncclProxySaveOp(struct ncclComm* comm, struct ncclProxyOp* op, bool* justInquire) {
  ++g_proxySaveOpCalls;
  g_proxySaveOpLastComm = comm;
  g_proxySaveOpLastOp = op;
  // Production always passes a real op and a real out-param; a null here is a
  // defect in the caller, not a case worth tolerating silently.
  if (op == nullptr) {
    std::fprintf(stderr, "[enqueue_fakes] ncclProxySaveOp called with a NULL op\n");
    std::fflush(stderr);
    ::abort();
  }
  if (justInquire == nullptr) {
    std::fprintf(stderr, "[enqueue_fakes] ncclProxySaveOp called with a NULL justInquire\n");
    std::fflush(stderr);
    ::abort();
  }
  g_proxySaveOpLastChannelId = op->channelId;
  g_proxySaveOpLastOpCount = op->opCount;
  g_proxySaveOpSawJustInquireIn = *justInquire;
  *justInquire = g_proxySaveOpJustInquire;
  return g_proxySaveOpResult;
}

// NOTE: ncclCommPollEventCallbacks is an INLINE function in comm.h:1020, not an
// external symbol, so it cannot be replaced by a fake. It drains
// comm->eventCallbackQueue; with an empty queue its only observable action is a
// single hipThreadExchangeStreamCaptureMode call, which hip_fakes.cc:427 already
// routes through g_hipAsyncOpsResult. waitWorkFifoAvailable tests therefore
// drive that EXISTING seam rather than adding a new one here.
ncclResult_t g_proxyStartResult = ncclSuccess;
ncclResult_t ncclProxyStart(struct ncclComm*) { return g_proxyStartResult; }

void ResetEnqueueFakes() {
  ResetHipFakes();
  ResetNcclFakes();

  g_topoGetAlgoTime = DefaultTopoGetAlgoTime;
  g_topoGetAlgoTimeCalls = 0;
  g_paramMinNchannels = 0;
  g_paramMaxNchannels = MAXCHANNELS;

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
  g_rcclCeAllReduceGraphLatchTickCalls = 0;
  g_rcclCeAllReduceGraphLatchTickLastCapturing = false;
  g_symRegType = ncclSymSendNonregRecvNonreg;
  g_getSymRegTypeResult = ncclSuccess;
  g_getSymRegTypeCalls = 0;
  g_rcclUseAinic = false;

  g_devrWindowIsMultiSegment = false;
  g_devrWindowHasSysmemSegment = false;

  g_ceImplemented = false;
  g_ceAvailable = false;
  g_ceScratchAvailable = false;
  g_hierCeAvailable = false;
  g_rcclCeAllReduceAllowed = false;

  g_proxySaveOpResult = ncclSuccess;
  g_proxySaveOpCalls = 0;
  g_proxySaveOpJustInquire = false;
  g_proxySaveOpLastComm = nullptr;
  g_proxySaveOpLastOp = nullptr;
  g_proxySaveOpLastChannelId = -1;
  g_proxySaveOpLastOpCount = 0;
  g_proxySaveOpSawJustInquireIn = false;
  // g_hipAsyncOpsResult belongs to hip_fakes and is restored by ResetHipFakes()
  // at the top of this function.
  g_proxyStartResult = ncclSuccess;
  g_enqEnv.clear();
  g_commEnsureReadyResult = ncclSuccess;
  g_recorderResult = ncclSuccess;
}

// ===========================================================================
// Tier 2 -- data symbols and string helpers
// ===========================================================================

// Algorithm/protocol name tables. topoGetAlgoInfo passes these to the RCCL
// override hooks and logs through them; they are pure data, so real names are
// used rather than placeholders -- a test asserting on a log line should see
// what production would print.
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = {
    "Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree", "PAT"};
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = {"LL", "LL128", "Simple"};

int ncclCudaDriverVersionCache = 12000;
bool ncclCudaLaunchBlocking = false;
int ncclProfilerEventMask = 0;
std::unordered_map<uint64_t, int> ncclDevFuncNameToId;

// Real-ish strings: enqueue.cc logs these, and a test that asserts on a log line
// needs them stable. Not worth a seam.
const char* ncclFuncToString(ncclFunc_t op) {
  switch (op) {
    case ncclFuncBroadcast: return "Broadcast";
    case ncclFuncReduce: return "Reduce";
    case ncclFuncAllGather: return "AllGather";
    case ncclFuncReduceScatter: return "ReduceScatter";
    case ncclFuncAllReduce: return "AllReduce";
    case ncclFuncSendRecv: return "SendRecv";
    case ncclFuncSend: return "Send";
    case ncclFuncRecv: return "Recv";
    default: return "Invalid";
  }
}
const char* ncclAlgoToString(int algo) {
  switch (algo) {
    case NCCL_ALGO_TREE: return "TREE";
    case NCCL_ALGO_RING: return "RING";
    case NCCL_ALGO_COLLNET_DIRECT: return "COLLNET_DIRECT";
    case NCCL_ALGO_COLLNET_CHAIN: return "COLLNET_CHAIN";
    case NCCL_ALGO_NVLS: return "NVLS";
    case NCCL_ALGO_NVLS_TREE: return "NVLS_TREE";
    case NCCL_ALGO_PAT: return "PAT";
    default: return "Unknown";
  }
}
// If NCCL_NUM_ALGORITHMS grows, this switch needs the new name -- otherwise a
// real algorithm silently logs as "Unknown", which is what happened with PAT.
static_assert(NCCL_NUM_ALGORITHMS == 7,
              "ncclAlgoToString above must name every algorithm; add the new case");

const char* ncclProtoToString(int proto) {
  switch (proto) {
    case NCCL_PROTO_LL: return "LL";
    case NCCL_PROTO_LL128: return "LL128";
    case NCCL_PROTO_SIMPLE: return "SIMPLE";
    default: return "Unknown";
  }
}
const char* ncclDatatypeToString(ncclDataType_t) { return "dtype"; }
const char* ncclDevRedOpToString(ncclDevRedOp_t) { return "redop"; }

void* ncclOsAlignedAlloc(size_t alignment, size_t size) {
  void* p = nullptr;
  if (posix_memalign(&p, alignment, size) != 0) return nullptr;
  return p;
}
void ncclOsAlignedFree(void* ptr) { free(ptr); }

// ncclGetSymRegType: previously a fixed success in enqueue_stubs.cc that always
// reported "neither side registered", silently steering production down one arm.
// Configurable here so a test can select the other arms or fail the query.
ncclSymRegType_t g_symRegType = ncclSymSendNonregRecvNonreg;
ncclResult_t g_getSymRegTypeResult = ncclSuccess;
int g_getSymRegTypeCalls = 0;
ncclResult_t ncclGetSymRegType(struct ncclDevrWindow*, struct ncclDevrWindow*,
                               ncclSymRegType_t* out) {
  ++g_getSymRegTypeCalls;
  if (out) *out = g_symRegType;
  return g_getSymRegTypeResult;
}
