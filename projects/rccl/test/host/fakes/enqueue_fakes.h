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

// LINK FLOOR ONLY: a seam marked `// UNDRIVEN` is declared so the binary links
// and so an accidental call is visible, NOT because its path is covered. The
// marker travels with the declaration, so driving a seam means deleting its
// marker rather than reasoning about which block a note applies to.

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
extern int64_t g_paramMinNchannels;  // UNDRIVEN
extern int64_t g_paramMaxNchannels;  // UNDRIVEN

// -------------------------------------------------------------------------
// RCCL tuning-override seams (enqueue.cc:2556-2841). All default to no-ops so a
// test sees the *unmodified* selection, then overrides exactly one.
// -------------------------------------------------------------------------
extern std::function<void(struct ncclComm*, size_t const&, struct ncclTaskColl*)>
    g_rcclUpdateCollectiveProtocol;  // UNDRIVEN
extern std::function<void(struct ncclComm*, size_t const&, struct ncclTaskColl*)>
    g_rcclSetPipelining;  // UNDRIVEN
extern std::function<ncclResult_t(struct ncclComm*, ncclFunc_t, size_t, int&)>
    g_rcclOverrideChannels;
extern int g_rcclOverrideChannelsCalls;
extern bool g_rcclIsArchSupportedForFunc;  // UNDRIVEN
// Call counters for the no-op tuning hooks: a no-op that was never called and one
// that was look identical without these, so a dropped call site would be silent.
extern int g_rcclUpdateCollectiveProtocolCalls;
extern int g_rcclSetPipeliningCalls;
extern int g_rcclUpdateThreadThresholdCalls;
extern int g_rcclOptThreadBlockSizeCalls;
extern ncclResult_t g_rcclOverrideAlgorithmResult;  // UNDRIVEN
extern ncclResult_t g_rcclOverrideProtocolResult;  // UNDRIVEN
extern int g_rcclOverrideAlgorithmCalls;
extern int g_rcclOverrideProtocolCalls;
extern int g_rcclCeAllReduceGraphLatchTickCalls;  // UNDRIVEN
extern bool g_rcclCeAllReduceGraphLatchTickLastCapturing;  // UNDRIVEN
// Symmetric-registration query. Defaults to "neither side registered", but that
// is a CHOICE that steers production down one arm -- make it explicit and
// overridable rather than a fixed stub result.
extern ncclSymRegType_t g_symRegType;  // UNDRIVEN
extern ncclResult_t g_getSymRegTypeResult;  // UNDRIVEN
extern int g_getSymRegTypeCalls;  // UNDRIVEN
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
extern bool g_devrWindowIsMultiSegment;  // UNDRIVEN
extern bool g_devrWindowHasSysmemSegment;  // UNDRIVEN

// CE (copy-engine) availability gates (enqueue.cc:3498-3560, 3789+).
extern bool g_ceImplemented;  // UNDRIVEN
extern bool g_ceAvailable;  // UNDRIVEN
extern bool g_ceScratchAvailable;  // UNDRIVEN
extern bool g_hierCeAvailable;  // UNDRIVEN
extern bool g_rcclCeAllReduceAllowed;  // UNDRIVEN

// -------------------------------------------------------------------------
// Proxy / launch seams. addProxyOpIfNeeded reaches ncclProxySaveOp, so that one
// is driven; the launch and proxy-start seams below it are not.
// -------------------------------------------------------------------------
// Scripted env for the UUT's ncclGetEnv reads. Cleared by ResetEnqueueFakes().
// TRAP: updateCollCostTable:2527 caches NCCL_PROTO in a function-local static, so
// only the first call in the PROCESS observes it -- scripting it later is a no-op.
// Because no test writes this map, that static latches to 0 in every ordering:
// the user-NCCL_PROTO bypass of the XGMI LL128 gate is therefore UNREACHABLE in
// this binary, and --gtest_shuffle passing reflects a constant latch rather than
// proven order-independence. The first test to script NCCL_PROTO inherits a
// seed-dependent flake and should add process isolation.
// TRAP 2: topoGetAlgoInfo:2686 reads NCCL_PROTO/NCCL_ALGO through raw libc
// getenv(), NOT ncclGetEnv, so this map does not intercept them. Unreachable
// today (gated on pivotA2ANumBiRings == 3, which the fixtures zero-init), but
// one field assignment from making results depend on the CI machine's ambient
// environment.
extern std::unordered_map<std::string, std::string> g_enqEnv;  // UNDRIVEN

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
extern ncclResult_t g_proxyStartResult;  // UNDRIVEN

// waitWorkFifoAvailable escapes its spin loop via ncclCommPollEventCallbacks,
// which is INLINE in comm.h and cannot be faked. Its only observable action on an
// empty callback queue is hipThreadExchangeStreamCaptureMode, already routed
// through g_hipAsyncOpsResult (hip_fakes.h). Note that seam is a plain RESULT,
// not a callable -- it cannot count iterations or advance workFifoConsumed. A
// test that fills the FIFO must therefore make the poll FAIL, or set abortFlag;
// anything else spins forever and HANGS the suite rather than failing it.

void ResetEnqueueFakes();

#endif  // RCCL_TEST_HOST_ENQUEUE_FAKES_H_
