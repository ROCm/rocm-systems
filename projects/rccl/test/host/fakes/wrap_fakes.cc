/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only dependency floor for src/rccl_wrap.cc's microtest, compiled
// into rccl-UnitTestsMicro alongside p2p.cc/rma_proxy_progress.cc's own
// tests. #include-ing the whole 1773-line rccl_wrap.cc (via WRAP_CC_PATH)
// pulls in the DDA / CE / symmetric-kernel / hierarchical backend-selection
// machinery even though the covered functions (see wrap-test.cc) don't reach
// all of it yet. Everything below satisfies the link closure. Controllable
// seams use conservative production-like defaults; tests override them with
// ScopedHook when a branch needs different behaviour.
//
// Dependencies with an established owner live in their canonical fake files:
// collectives_fakes, strongstream_stubs, dev_runtime_fakes, ce_fakes,
// sym_kernels_fakes, transport_stubs, and tuning_fakes. This file contains
// only the remaining rccl_wrap.cc link-closure seams that have no shared owner
// yet. nccl_stubs.cc is not linked because it defines commSetUnrollFactor,
// which is a real function supplied by the unit under test itself.
//
// RCCL_PARAM / NCCL_PARAM: every RCCL_PARAM(...) invocation textually inside
// rccl_wrap.cc is redirected by wrap-test.cc to route through a g_loadParam
// std::function hook (same mechanism as init-test.cc's redirect), so a test
// can flip one param's value between cases -- see wrap-test.cc's redirector
// comment. The few ncclParamXxx / rcclParamXxx symbols rccl_wrap.cc declares
// `extern` and calls without a local RCCL_PARAM/NCCL_PARAM invocation (their
// generator lives in another .cc) are stubbed below instead.

#include "wrap_fakes.h"

#include <cstdint>
#include <functional>

#include "algorithms/dda/all_gather/dda_all_gather.h"
#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"
#include "amdsmi_wrap.h"
#include "ce_coll.h"
#include "comm.h"
#include "debug.h"
#include "dev_runtime.h"
#include "enqueue.h"
#include "group.h"
#include "nccl.h"
#include "rccl_common.h"
#include "signature-drift.h"
#include "strongstream.h"
#include "sym_kernels.h"

ASSERT_HOOK_MATCHES_PROD(g_amdSmiGetFirmwareVersion, amd_smi_getFirmwareVersion);
ASSERT_HOOK_MATCHES_PROD(g_isSymmetricKernelRequested, isSymmetricKernelRequested);
ASSERT_HOOK_MATCHES_PROD(g_allReduceShouldTakeDdaPath, rcclAllReduceShouldTakeDdaPath);
ASSERT_HOOK_MATCHES_PROD(g_commCount, ncclCommCount);

ASSERT_HOOK_MATCHES_PROD(g_allReduceDdaIpcEligible, ncclAllReduceDdaIpcEligible);
ASSERT_HOOK_MATCHES_PROD(g_allReduceDdaFabricEligible, ncclAllReduceDdaFabricEligible);
ASSERT_HOOK_MATCHES_PROD(g_allReduceDdaFabricLLEligible, ncclAllReduceDdaFabricLLEligible);
ASSERT_HOOK_MATCHES_PROD(g_allReduceDdaFabricLL128Eligible, ncclAllReduceDdaFabricLL128Eligible);
ASSERT_HOOK_MATCHES_PROD(g_allReduceDdaIpcBlocks, ncclAllReduceDdaIpcBlocks);
ASSERT_HOOK_MATCHES_PROD(g_allReduceDdaFabricBlocks, ncclAllReduceDdaFabricBlocks);
ASSERT_HOOK_MATCHES_PROD(g_allReduceDdaFabricLLBlocks, ncclAllReduceDdaFabricLLBlocks);
ASSERT_HOOK_MATCHES_PROD(g_allReduceDdaFabricLL128Blocks, ncclAllReduceDdaFabricLL128Blocks);

ASSERT_HOOK_MATCHES_PROD(g_allGatherDdaIpcEligible, ncclAllGatherDdaIpcEligible);
ASSERT_HOOK_MATCHES_PROD(g_allGatherDdaFabricEligible, ncclAllGatherDdaFabricEligible);
ASSERT_HOOK_MATCHES_PROD(g_allGatherDdaFabricLLEligible, ncclAllGatherDdaFabricLLEligible);
ASSERT_HOOK_MATCHES_PROD(g_allGatherDdaFabricLL128Eligible, ncclAllGatherDdaFabricLL128Eligible);
ASSERT_HOOK_MATCHES_PROD(g_allGatherDdaIpcBlocks, ncclAllGatherDdaIpcBlocks);
ASSERT_HOOK_MATCHES_PROD(g_allGatherDdaFabricBlocks, ncclAllGatherDdaFabricBlocks);
ASSERT_HOOK_MATCHES_PROD(g_allGatherDdaFabricLLBlocks, ncclAllGatherDdaFabricLLBlocks);
ASSERT_HOOK_MATCHES_PROD(g_allGatherDdaFabricLL128Blocks, ncclAllGatherDdaFabricLL128Blocks);

ASSERT_HOOK_MATCHES_PROD(g_reduceScatterDdaIpcEligible, ncclReduceScatterDdaIpcEligible);
ASSERT_HOOK_MATCHES_PROD(g_reduceScatterDdaFabricEligible, ncclReduceScatterDdaFabricEligible);
ASSERT_HOOK_MATCHES_PROD(g_reduceScatterDdaFabricLLEligible, ncclReduceScatterDdaFabricLLEligible);
ASSERT_HOOK_MATCHES_PROD(g_reduceScatterDdaFabricLL128Eligible, ncclReduceScatterDdaFabricLL128Eligible);
// getAlgoInfo, rcclKernelPackedChannels, and the four ReduceScatter *Blocks
// hooks are link-closure symbols with no production header declaration, so
// there is no header-visible signature to pin here.

#undef ASSERT_HOOK_MATCHES_PROD

// ---------------------------------------------------------------------------
// Logging infrastructure. WARN/INFO (debug.h) expand to ncclDebugLog(); three
// of the units under test (rcclGetAlgoProtoIndex, rcclGetAlgoName,
// rcclGetProtocolName) call WARN on their invalid-input arms, and tests assert
// on that text via gtest's stderr capture.
//
// ncclDebugLevel / ncclDebugMask / ncclDebugNoWarn / ncclDebugLog() itself
// are defined once, by fakes/nccl_fakes.cc, which this binary already links
// for p2p.cc's tests -- defining them again here would be a duplicate-symbol
// error. ncclDebugMask's default there (0) differs from what this file used
// when it was its own standalone binary (~0ULL). WARN(...) calls
// ncclDebugLog() unconditionally regardless of mask; INFO(...) call sites are
// tested with explicit ScopedDebugLogging settings in wrap-test.cc rather
// than relying on either fake's default.
// ---------------------------------------------------------------------------
FILE* ncclDebugFile = nullptr;
char  ncclLastError[1024] = {};

// ---------------------------------------------------------------------------
// ncclParamXxx / rcclParamXxx symbols NOT generated by an RCCL_PARAM/NCCL_PARAM
// invocation inside rccl_wrap.cc itself (see file header comment). Trivial
// real behaviour (return the production default) rather than abort(): these
// are read on essentially every call into the file's channel/DDA tuning
// paths, so an abort-floor here would make even unrelated tests crash before
// reaching the function they actually meant to exercise.
//
// Unlike NCCL_NUM_ALGORITHMS/ncclNumFuncs below, there's no importable
// constant for "the current default" here -- NCCL_PARAM/RCCL_PARAM bakes
// deftVal directly as an inline macro-argument literal (see param.h), not a
// separately declared symbol, so a static_assert can't check it. CMakeLists.txt
// checks the invocation text against the original sources at configure time,
// when those files are guaranteed to be available.
// ---------------------------------------------------------------------------
static int64_t DefaultParamForceCe() { return 1; }          // enqueue.cc:3796
std::function<int64_t()> g_paramForceCe = DefaultParamForceCe;
int64_t rcclParamForceCe() { return g_paramForceCe(); }

// ncclParamLaunchOrderImplicit: a settable hook, like rcclParamForceCe above,
// so a test can drive rcclDdaEnabled's
// `ncclParamLaunchOrderImplicit() != 0` disjunct independently -- the real
// default (0, "explicit launch order") favors the common case, matching the
// other three.
static int64_t DefaultParamLaunchOrderImplicit() { return 0; }  // enqueue.cc:1985
std::function<int64_t()> g_paramLaunchOrderImplicit = DefaultParamLaunchOrderImplicit;
int64_t ncclParamLaunchOrderImplicit() { return g_paramLaunchOrderImplicit(); }

// The settable env-var fake (SetMicroEnv/SetMicroEnvAbsent/ClearMicroEnv,
// the bare-getenv() link-level interposer, and ncclGetEnv) is NOT defined
// here: fakes/env_fakes.cc is the shared owner of src/misc/param.cc + getenv
// interposition for every microtest binary (see MICROTEST_README.md's
// "Where a fake belongs"). rccl_wrap.cc's several bare getenv() call sites
// (rcclSetPxn, rcclSetP2pNetChunkSize, rcclUpdateCollectiveProtocol,
// rcclUpdateThreadThreshold, ...) are covered by that same interposer.
// ncclGroupDepth is NOT faked here: group-test.cc (also part of this binary
// since the rccl-UnitTestsMicro merge) compiles the real group.cc, which
// defines it -- a second copy here would be a duplicate-symbol error. Its
// real default (0, "not grouped") is what every rccl_wrap.cc test so far
// assumes, same as when this was a hand-copied stub.

// ncclDevFuncUnrollGenerated: extern bool const[NCCL_NUM_UNROLLS]. Real array,
// not abort-floor -- commSetUnrollFactor indexes it unconditionally on every
// call, including the manual-override path the commSetUnrollFactor tests
// exercise. All-true is the safe default: "every unroll factor was built".
//
// Confirmed NOT safely upgradeable to a test seam: device.h's `extern bool
// const [...]` declaration is transitively visible in this TU too (this
// file doesn't include device.h directly, but something in its include
// chain does) -- dropping `const` here is a hard redefinition-with-
// different-type compile error, not just a lurking runtime risk. Leaving
// this const and documented, not forced through.
//
// Drift watchdog: without one, a new unroll factor added to RCCL would not be
// caught here -- the extra array slot would silently zero-initialize to false,
// quietly changing behavior. The check is a RUNTIME test
// (FakeTableDrift_UnrollCountMatchesProduction in wrap-test.cc), not a
// static_assert, for the same reason as the enum-count checks below.
const bool ncclDevFuncUnrollGenerated[NCCL_NUM_UNROLLS] = {true, true, true, true, true, true};

// ncclCommCount: real behaviour by default (de-risks future tests),
// but a controllable seam so rcclGetAlgoInfo's NCCLCHECK(ncclCommCount(...))
// failure arm (never exercised -- every prior test let this succeed) can be
// proven too.
static ncclResult_t DefaultCommCount(const ncclComm_t comm, int* count) {
  if (count) *count = comm ? comm->nRanks : 0;
  return ncclSuccess;
}
std::function<ncclResult_t(const ncclComm_t, int*)> g_commCount = DefaultCommCount;
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) { return g_commCount(comm, count); }

// ---------------------------------------------------------------------------
// Controllable seams for the top-level dispatchers (rcclSelectAllReduce/
// AllGather/ReduceScatter, rcclHierarchicalAlgoInfo, rcclGetAlgoInfo,
// rcclGetCollImplInfo, rcclSymkQuery/rcclSymKGetInfo's deep path). Without
// these each would have exactly one reachable outcome (see
// MICROTEST_README.md, "Adding more controllable
// seams" for the pattern followed here); upgraded so wrap-test.cc can drive
// each branch of those dispatchers directly. Every default below reproduces
// the safe, common "this fast path doesn't apply" case so existing tests
// upstream of these seams are unaffected -- verified by rereading each real
// call site before choosing it.
// ---------------------------------------------------------------------------

// Drives `symEligible` in all three rcclSelectXxx functions. Default false:
// "no symmetric-window kernel requested," letting the DDA/CE/Direct/plain
// paths run, matching every existing (guard-only) test's expectations.
static bool DefaultIsSymmetricKernelRequested(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t, const void*,
                                              void*) {
  return false;
}
std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t, const void*, void*)>
    g_isSymmetricKernelRequested = DefaultIsSymmetricKernelRequested;
bool isSymmetricKernelRequested(struct ncclComm* comm, ncclFunc_t coll, int symkOp, ncclDataType_t datatype,
                                size_t nElts, const void* sendbuff, void* recvbuff) {
  return g_isSymmetricKernelRequested(comm, coll, symkOp, datatype, nElts, sendbuff, recvbuff);
}

// rcclAllReduceShouldTakeDdaPath: real body lives in collectives.cc (not
// linked here), same abort-floor-turned-seam treatment as the rest. Default
// false: DDA not taken, letting CE-registered/symmetric/plain-kernel run.
static bool DefaultAllReduceShouldTakeDdaPath(const struct ncclComm*, size_t, ncclDataType_t, bool, bool) {
  return false;
}
std::function<bool(const struct ncclComm*, size_t, ncclDataType_t, bool, bool)> g_allReduceShouldTakeDdaPath =
    DefaultAllReduceShouldTakeDdaPath;
bool rcclAllReduceShouldTakeDdaPath(const struct ncclComm* comm, size_t count, ncclDataType_t dt, bool symEligible,
                                    bool ceAllReduceAllowed) {
  return g_allReduceShouldTakeDdaPath(comm, count, dt, symEligible, ceAllReduceAllowed);
}

// getAlgoInfo / rcclKernelPackedChannels: rccl_wrap.cc `extern`-declares both
// itself (their real definitions live in enqueue.cc / device-side tuning,
// outside this TU's link closure), so no separate declaration is needed here.
// Default getAlgoInfo fills in a sane, generic Ring/Simple/1-channel answer
// (the universal plain-kernel fallback every rcclSelectXxx eventually falls
// to); default rcclKernelPackedChannels passes the tuning-cap channel count
// through unpacked, a safe no-op default.
static ncclResult_t DefaultGetAlgoInfo(struct ncclComm*, struct ncclTaskColl* task, int, int, int, ncclSimInfo_t*) {
  task->algorithm = NCCL_ALGO_RING;
  task->protocol = NCCL_PROTO_SIMPLE;
  task->nMaxChannels = 1;
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, int, int, int, ncclSimInfo_t*)> g_getAlgoInfo =
    DefaultGetAlgoInfo;
ncclResult_t getAlgoInfo(struct ncclComm* comm, struct ncclTaskColl* task, int collNetSupport, int nvlsSupport,
                         int numPipeOps, ncclSimInfo_t* simInfo) {
  return g_getAlgoInfo(comm, task, collNetSupport, nvlsSupport, numPipeOps, simInfo);
}

static int DefaultKernelPackedChannels(struct ncclComm*, ncclFunc_t, size_t, ncclDataType_t, int, int nMaxChannels) {
  return nMaxChannels;
}
std::function<int(struct ncclComm*, ncclFunc_t, size_t, ncclDataType_t, int, int)> g_kernelPackedChannels =
    DefaultKernelPackedChannels;
int rcclKernelPackedChannels(struct ncclComm* comm, ncclFunc_t func, size_t count, ncclDataType_t dt, int protocol,
                             int nMaxChannels) {
  return g_kernelPackedChannels(comm, func, count, dt, protocol, nMaxChannels);
}

// rcclLL128ElemsPerThreadFromArch is `inline` in archinfo.h (transitively
// included), so no stub is needed here.

// getFirmwareVersion()'s sole dependency. Settable hook, same std::function
// shape as fakes/nccl_fakes.cc's, so a test can script a canned firmware
// response or a failure without touching the real AMD-SMI layer.
static ncclResult_t DefaultAmdSmiGetFirmwareVersion(uint32_t /*devIdx*/, uint64_t* fwVersion) {
  *fwVersion = 0;
  return ncclSuccess;
}
std::function<ncclResult_t(uint32_t, uint64_t*)> g_amdSmiGetFirmwareVersion = DefaultAmdSmiGetFirmwareVersion;
ncclResult_t amd_smi_getFirmwareVersion(uint32_t devIdx, uint64_t* fwVersion) {
  return g_amdSmiGetFirmwareVersion(devIdx, fwVersion);
}

// --- Per-collective DDA eligibility/blocks (24 functions total) ---
// Every *Eligible defaults false (DDA path not eligible by default, letting
// CE-registered/symmetric/hierarchical/Direct/plain-kernel run, matching
// every existing test's expectations). Every reachable *Blocks hook has a
// distinct 11x/12x sentinel, and selecting tests assert those values to prove
// which path supplied nMaxChannels. The unused 13x hooks remain distinct so a
// future production call cannot silently agree with a neighboring path.

#define DEFINE_DDA_REDUCTION_ELIGIBLE(hook, prod)                                                        \
  std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t)> g_##hook =     \
      [](ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) { return false; };          \
  bool prod(ncclComm* comm, const void* send, void* recv, size_t count, ncclDataType_t type,             \
            ncclRedOp_t op) {                                                                            \
    return g_##hook(comm, send, recv, count, type, op);                                                   \
  }

#define DEFINE_DDA_ALLGATHER_ELIGIBLE(hook, prod)                                                         \
  std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t)> g_##hook =                   \
      [](ncclComm*, const void*, void*, size_t, ncclDataType_t) { return false; };                        \
  bool prod(ncclComm* comm, const void* send, void* recv, size_t count, ncclDataType_t type) {            \
    return g_##hook(comm, send, recv, count, type);                                                       \
  }

#define DEFINE_DDA_BLOCKS(hook, prod, sentinel)                                                           \
  std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_##hook =                                  \
      [](ncclComm*, size_t, ncclDataType_t) { return sentinel; };                                        \
  uint32_t prod(ncclComm* comm, size_t count, ncclDataType_t type) {                                     \
    return g_##hook(comm, count, type);                                                                   \
  }

// --- AllReduce DDA (dda_all_reduce.h) ---
DEFINE_DDA_REDUCTION_ELIGIBLE(allReduceDdaIpcEligible, ncclAllReduceDdaIpcEligible)
DEFINE_DDA_REDUCTION_ELIGIBLE(allReduceDdaFabricEligible, ncclAllReduceDdaFabricEligible)
DEFINE_DDA_REDUCTION_ELIGIBLE(allReduceDdaFabricLLEligible, ncclAllReduceDdaFabricLLEligible)
DEFINE_DDA_REDUCTION_ELIGIBLE(allReduceDdaFabricLL128Eligible, ncclAllReduceDdaFabricLL128Eligible)

// The twelve *DdaBlocks fakes below each return a DISTINCT sentinel rather
// than a shared value. Their results land in decision->nMaxChannels, so if
// they all returned the same number the tests could not tell which one a
// dispatcher actually called: swapping an LL blocks call for the LL128 one
// would be invisible, as would hardcoding the field. The AllReduce and
// AllGather tests that select each path assert its sentinel, which pins both.
//
// Numbering is 1yz: y = 1 AllReduce, 2 AllGather, 3 ReduceScatter;
// z = 1 Ipc, 2 Fabric/VMM, 3 FabricLL, 4 FabricLL128 (so 111-114,
// 121-124, 131-134).
//
// The four 13x (ReduceScatter) entries are link-only: rcclSelectReduceScatter
// sets algo and protocol on its DDA paths but never nMaxChannels, and nothing
// in rccl_wrap.cc calls those four symbols. They keep distinct values anyway
// so that if a channel count is ever wired up there, the existing tests fail
// loudly rather than silently agreeing with a neighbouring path's value.
DEFINE_DDA_BLOCKS(allReduceDdaIpcBlocks, ncclAllReduceDdaIpcBlocks, 111)
DEFINE_DDA_BLOCKS(allReduceDdaFabricBlocks, ncclAllReduceDdaFabricBlocks, 112)
DEFINE_DDA_BLOCKS(allReduceDdaFabricLLBlocks, ncclAllReduceDdaFabricLLBlocks, 113)
DEFINE_DDA_BLOCKS(allReduceDdaFabricLL128Blocks, ncclAllReduceDdaFabricLL128Blocks, 114)

// --- AllGather DDA (dda_all_gather.h) ---
DEFINE_DDA_ALLGATHER_ELIGIBLE(allGatherDdaIpcEligible, ncclAllGatherDdaIpcEligible)
DEFINE_DDA_ALLGATHER_ELIGIBLE(allGatherDdaFabricEligible, ncclAllGatherDdaFabricEligible)
DEFINE_DDA_ALLGATHER_ELIGIBLE(allGatherDdaFabricLLEligible, ncclAllGatherDdaFabricLLEligible)
DEFINE_DDA_ALLGATHER_ELIGIBLE(allGatherDdaFabricLL128Eligible, ncclAllGatherDdaFabricLL128Eligible)
DEFINE_DDA_BLOCKS(allGatherDdaIpcBlocks, ncclAllGatherDdaIpcBlocks, 121)
DEFINE_DDA_BLOCKS(allGatherDdaFabricBlocks, ncclAllGatherDdaFabricBlocks, 122)
DEFINE_DDA_BLOCKS(allGatherDdaFabricLLBlocks, ncclAllGatherDdaFabricLLBlocks, 123)
DEFINE_DDA_BLOCKS(allGatherDdaFabricLL128Blocks, ncclAllGatherDdaFabricLL128Blocks, 124)

// --- ReduceScatter DDA (dda_reduce_scatter.h) ---
DEFINE_DDA_REDUCTION_ELIGIBLE(reduceScatterDdaIpcEligible, ncclReduceScatterDdaIpcEligible)
DEFINE_DDA_REDUCTION_ELIGIBLE(reduceScatterDdaFabricEligible, ncclReduceScatterDdaFabricEligible)
DEFINE_DDA_REDUCTION_ELIGIBLE(reduceScatterDdaFabricLLEligible, ncclReduceScatterDdaFabricLLEligible)
DEFINE_DDA_REDUCTION_ELIGIBLE(reduceScatterDdaFabricLL128Eligible, ncclReduceScatterDdaFabricLL128Eligible)
DEFINE_DDA_BLOCKS(reduceScatterDdaIpcBlocks, ncclReduceScatterDdaIpcBlocks, 131)
DEFINE_DDA_BLOCKS(reduceScatterDdaFabricBlocks, ncclReduceScatterDdaFabricBlocks, 132)
DEFINE_DDA_BLOCKS(reduceScatterDdaFabricLLBlocks, ncclReduceScatterDdaFabricLLBlocks, 133)
DEFINE_DDA_BLOCKS(reduceScatterDdaFabricLL128Blocks, ncclReduceScatterDdaFabricLL128Blocks, 134)

#undef DEFINE_DDA_BLOCKS
#undef DEFINE_DDA_ALLGATHER_ELIGIBLE
#undef DEFINE_DDA_REDUCTION_ELIGIBLE
