/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Enqueue-only fake seams for the host-only `rccl-UnitTestsMicroEnqueue` binary.
// See test/host/MICROTEST_README.md.

#ifndef RCCL_TEST_HOST_ENQUEUE_FAKES_H_
#define RCCL_TEST_HOST_ENQUEUE_FAKES_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "hip_fakes.h"
#include "nccl_fakes.h"
#include "sym_kernels.h"   // ncclSymRegType_t

struct ncclComm;
struct ncclTaskColl;
struct ncclProxyOp;
struct ncclKernelPlan;
struct ncclReg;
struct ncclDevrWindow;

// -------------------------------------------------------------------------
// Tuning seams (enqueue.cc:2486-2841). updateCollCostTable/topoGetAlgoInfo read
// the cost table this fills, so a test drives algorithm selection entirely
// through g_topoGetAlgoTime rather than by constructing a real topology.
//
// TRAP: the cost table is float[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] and
// NCCL_ALGO_PROTO_IGNORE marks a cell as unusable. A fake that returns a time
// for EVERY (algo, proto) pair makes every cell selectable, which is not what
// production sees -- prefer scripting only the pairs a test cares about.
// -------------------------------------------------------------------------
extern std::function<ncclResult_t(struct ncclComm*, int coll, int algorithm, int protocol,
                                  size_t nBytes, int numPipeOps, float* time)>
    g_topoGetAlgoTime;
extern int g_topoGetAlgoTimeCalls;

// Min/max channel clamps read by topoGetAlgoInfo. Defaults mirror production.
extern int64_t g_paramMinNchannels;
extern int64_t g_paramMaxNchannels;

// -------------------------------------------------------------------------
// RCCL tuning-override seams (enqueue.cc:2556-2841). All default to no-ops so a
// test sees the *unmodified* selection, then overrides exactly one.
// -------------------------------------------------------------------------
extern std::function<void(struct ncclComm*, size_t const&, struct ncclTaskColl*)>
    g_rcclUpdateCollectiveProtocol;
extern std::function<void(struct ncclComm*, size_t const&, struct ncclTaskColl*)>
    g_rcclSetPipelining;
extern std::function<ncclResult_t(struct ncclComm*, ncclFunc_t, size_t, int&)>
    g_rcclOverrideChannels;
extern int g_rcclOverrideChannelsCalls;
extern bool g_rcclIsArchSupportedForFunc;
// Call counters for the no-op tuning hooks: a no-op that was never called and one
// that was look identical without these, so a dropped call site would be silent.
extern int g_rcclUpdateCollectiveProtocolCalls;
extern int g_rcclSetPipeliningCalls;
extern int g_rcclUpdateThreadThresholdCalls;
extern int g_rcclOptThreadBlockSizeCalls;
extern ncclResult_t g_rcclOverrideAlgorithmResult;
extern ncclResult_t g_rcclOverrideProtocolResult;
extern int g_rcclOverrideAlgorithmCalls;
extern int g_rcclOverrideProtocolCalls;
extern int g_rcclCeAllReduceGraphLatchTickCalls;
extern bool g_rcclCeAllReduceGraphLatchTickLastCapturing;
// Symmetric-registration query. Defaults to "neither side registered", but that
// is a CHOICE that steers production down one arm -- make it explicit and
// overridable rather than a fixed stub result.
extern ncclSymRegType_t g_symRegType;
extern ncclResult_t g_getSymRegTypeResult;
extern int g_getSymRegTypeCalls;
// The AINIC gate in rcclEffectiveP2pBatchEnable's gfx950 arm.
extern bool g_rcclUseAinic;

// -------------------------------------------------------------------------
// Registration / window seams (enqueue.cc:2749-2841, 3498-3560).
// getAlgoInfo's regBuff decision reads these; default to "not registered" so the
// NVLS/CollNet registration arms stay off unless a test asks for them.
// -------------------------------------------------------------------------
// ncclRegLocalIsValid / ncclDevrFindWindow / ncclStreamWaitStream are owned by
// the SHARED fakes (nccl_fakes.cc, nccl_stubs.cc) and deliberately not redefined
// here -- a second definition is a link-time duplicate. Drive them by upgrading
// the shared seam, not by shadowing.
extern bool g_devrWindowIsMultiSegment;
extern bool g_devrWindowHasSysmemSegment;

// CE (copy-engine) availability gates (enqueue.cc:3498-3560, 3789+).
extern bool g_ceImplemented;
extern bool g_ceAvailable;
extern bool g_ceScratchAvailable;
extern bool g_hierCeAvailable;
extern bool g_rcclCeAllReduceAllowed;

// -------------------------------------------------------------------------
// Proxy / launch seams. These sit on the paths the microtests deliberately do
// NOT exercise (kernel launch, proxy start); they exist so the binary links and
// so an accidental call is visible rather than silent.
// -------------------------------------------------------------------------
// Scripted env for the UUT's ncclGetEnv reads. Cleared by ResetEnqueueFakes().
// TRAP: updateCollCostTable:2527 caches NCCL_PROTO in a function-local static, so
// only the first call in the PROCESS observes it -- scripting it later is a no-op.
extern std::unordered_map<std::string, std::string> g_enqEnv;

// ncclCommEnsureReady result, and the Recorder's. Production NCCLCHECKs the
// RedOpCreate record() call, so a failing recorder is an observable arm.
extern ncclResult_t g_commEnsureReadyResult;
extern ncclResult_t g_recorderResult;

extern ncclResult_t g_proxySaveOpResult;
extern int g_proxySaveOpCalls;
// The justInquire out-param: true means "this op matters, allocate for it".
extern bool g_proxySaveOpJustInquire;
// Recorded arguments. The queued copy is made from the CALLER's op, so asserting
// on the queue cannot show WHICH op the transport was actually asked about --
// these can. The fake aborts on a null op or null justInquire.
extern struct ncclComm* g_proxySaveOpLastComm;
extern struct ncclProxyOp* g_proxySaveOpLastOp;
extern int g_proxySaveOpLastChannelId;
extern uint64_t g_proxySaveOpLastOpCount;
extern bool g_proxySaveOpSawJustInquireIn;
extern ncclResult_t g_proxyStartResult;

// waitWorkFifoAvailable escapes its spin loop via ncclCommPollEventCallbacks,
// which is INLINE in comm.h and cannot be faked. Its only observable action on an
// empty callback queue is hipThreadExchangeStreamCaptureMode, already routed
// through g_hipAsyncOpsResult (hip_fakes.h). Note that seam is a plain RESULT,
// not a callable -- it cannot count iterations or advance workFifoConsumed. A
// test that fills the FIFO must therefore make the poll FAIL, or set abortFlag;
// anything else spins forever and HANGS the suite rather than failing it.

void ResetEnqueueFakes();

#endif  // RCCL_TEST_HOST_ENQUEUE_FAKES_H_
