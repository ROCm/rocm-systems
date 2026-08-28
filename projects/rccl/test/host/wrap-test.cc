/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/rccl_wrap.cc (AICOMRCCL-2195).
//
// Like init-test.cc / p2p-test.cc, this TU #includes the hipified
// unit-under-test source directly (via WRAP_CC_PATH) so its helpers become
// callable, links NO librccl/HIP, and satisfies every external symbol via
// fakes/wrap_stubs.cc.
//
// Covers 17 low-dependency helpers: nine that take no ncclComm* at all, or
// touch only a handful of plain fields, plus eight more that need a real
// one-GPU comm+topology (MakeCommWithArch below) but still avoid RCCL_PARAM/
// getenv/DDA/CE/symmetric-kernel machinery beyond a cached fast-return path.
// rcclOverrideChannels, rcclSetPipelining, the WarpSpeed helpers, and
// rcclSelectAllReduce/AllGather/ReduceScatter are unreached by design; each
// depends on seams this file doesn't build. Mutation-tested directly against
// this file; residuals are documented at their own test below rather than
// here.

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "../common/LogCapture.hpp"                 // RcclUnitTesting::CaptureLog
#include "../common/ProcessIsolatedTestRunner.hpp"  // RUN_ISOLATED_TEST
#include "fakes/wrap_stubs.h"                        // SetMicroEnv/SetMicroEnvAbsent/ClearMicroEnv
#include "graph/topo.h"                              // ncclTopoSystem/ncclTopoNode (MakeCommWithArch)

// RCCL_PARAM redirector: routes every generated rcclParamXxx() through
// g_loadParam on each call (no caching), so a test can flip one param's
// value between cases -- same mechanism and env-string convention
// ("RCCL_" + env) as init-test.cc's redirect, replicated here rather than
// linking fakes/nccl_fakes.cc (this file's own fakes stay self-contained,
// see wrap_stubs.cc's header comment).
#include "param.h"

#include <functional>

static int64_t DefaultLoadParam(const char* /*env*/, int64_t deftVal) { return deftVal; }
std::function<int64_t(const char*, int64_t)> g_loadParam = DefaultLoadParam;

#undef RCCL_PARAM
#define RCCL_PARAM(name, env, deftVal) \
  int64_t rcclParam##name() { return g_loadParam(("RCCL_" env), (deftVal)); }

// WRAP_CC_PATH is defined by test/host/CMakeLists.txt as the hipified copy of
// src/rccl_wrap.cc, e.g. ${PROJECT_BINARY_DIR}/hipify/src/rccl_wrap.cc.
#include WRAP_CC_PATH

namespace {

// Zero-initialized heap ncclComm, mirroring MockComm.hpp's
// new-then-memset idiom (test/common/MockComm.hpp) without pulling in that
// header's <rccl/rccl.h> include, which targets the installed-package layout
// rather than this target's hipify-tree headers. Caller deletes.
ncclComm* MakeZeroedComm() {
  ncclComm* comm = new ncclComm();
  std::memset(comm, 0, sizeof(ncclComm));
  return comm;
}

// Comm with a real one-GPU topology wired up, arch string settable -- mirrors
// test/common/MockComm.hpp's CreateMockComm without its <rccl/rccl.h> include
// (targets the installed-package layout, not this target's hipify-tree
// headers). archName is pointed at the same buffer as the topology node's gcn
// field -- both name the same GPU on a real comm, and this keeps the two
// consistent without a second allocation. Caller must DeleteCommWithArch.
ncclComm* MakeCommWithArch(const char* arch) {
  ncclComm* comm = MakeZeroedComm();
  comm->nRanks = 1;
  comm->nNodes = 1;
  comm->pxnDisable = RCCL_VALUE_UNSET;
  comm->p2pNetChunkSize = RCCL_VALUE_UNSET;
  auto* topo = new ncclTopoSystem();
  std::memset(topo, 0, sizeof(*topo));
  comm->topo = topo;
  topo->nodes[GPU].count = 1;
  std::strncpy(topo->nodes[GPU].nodes[0].gpu.gcn, arch, sizeof(topo->nodes[GPU].nodes[0].gpu.gcn) - 1);
  topo->nodes[GPU].nodes[0].gpu.gcn[sizeof(topo->nodes[GPU].nodes[0].gpu.gcn) - 1] = '\0';
  comm->archName = topo->nodes[GPU].nodes[0].gpu.gcn;
  return comm;
}

void DeleteCommWithArch(ncclComm* comm) {
  delete comm->topo;
  delete comm;
}

}  // namespace

// ===========================================================================
// rcclIsGfx120x / rcclGetProtoForGfx120x -- static inline helpers, only
// reachable via this #include model (no external symbol to call otherwise).
// rccl_wrap.cc:89-107.
// ===========================================================================

TEST(WrapMicrotest, IsGfx120x_MatchesBothMembers) {
  EXPECT_TRUE(rcclIsGfx120x("gfx1200"));
  EXPECT_TRUE(rcclIsGfx120x("gfx1201"));
}

TEST(WrapMicrotest, IsGfx120x_RejectsOtherArch) {
  EXPECT_FALSE(rcclIsGfx120x("gfx942"));
}

// SingleNodeLLCutoffs[] is indexed directly by ncclFunc_t, so its ordering IS
// the oracle: assert the exact NCCL_PROTO_* value at each cutoff's boundary,
// not just "doesn't crash". A swapped table row (the canonical off-by-one for
// a lookup table like this) flips a boundary's proto without changing the
// return type or control flow, so return-code-only assertions cannot catch it.
TEST(WrapMicrotest, GetProtoForGfx120x_BroadcastCutoffBoundary) {
  EXPECT_EQ(NCCL_PROTO_LL, rcclGetProtoForGfx120x(ncclFuncBroadcast, 1536));
  EXPECT_EQ(NCCL_PROTO_SIMPLE, rcclGetProtoForGfx120x(ncclFuncBroadcast, 1537));
}

TEST(WrapMicrotest, GetProtoForGfx120x_AllReduceCutoffBoundary) {
  EXPECT_EQ(NCCL_PROTO_LL, rcclGetProtoForGfx120x(ncclFuncAllReduce, 16384));
  EXPECT_EQ(NCCL_PROTO_SIMPLE, rcclGetProtoForGfx120x(ncclFuncAllReduce, 16385));
}

// ncclFuncSend/Recv/SendRecv all carry a zero cutoff: any positive size falls
// straight to SIMPLE, and the only way to reach their LL arm is size == 0.
TEST(WrapMicrotest, GetProtoForGfx120x_ZeroCutoffFuncsOnlyLLAtZero) {
  EXPECT_EQ(NCCL_PROTO_LL, rcclGetProtoForGfx120x(ncclFuncSendRecv, 0));
  EXPECT_EQ(NCCL_PROTO_SIMPLE, rcclGetProtoForGfx120x(ncclFuncSendRecv, 1));
}

// collectiveFunc >= the table's own extent (8 entries: Broadcast..Recv) takes
// the guard's false arm and returns the pre-set NCCL_PROTO_SIMPLE default
// unconditionally, regardless of sizePerRank. ncclFuncAlltoAll's enum value is
// past this table (added after the eight it was sized for).
//
// Residual: a `<` -> `<=` mutant of this guard is accepted, not fixed. At
// collectiveFunc == 8 exactly, the wrong branch reads SingleNodeLLCutoffs[8],
// one past the array's end -- undefined behavior, not a defined wrong value.
// Neither a value assertion nor -fsanitize=address reliably observes it in
// this build.
TEST(WrapMicrotest, GetProtoForGfx120x_FuncBeyondTable_DefaultsSimple) {
  EXPECT_EQ(NCCL_PROTO_SIMPLE, rcclGetProtoForGfx120x(ncclFuncAlltoAll, 1));
}

// ===========================================================================
// rcclCollSupportsRing -- static inline. rccl_wrap.cc:84-87.
// ===========================================================================

TEST(WrapMicrotest, CollSupportsRing_TrueForRingEligibleFuncs) {
  EXPECT_TRUE(rcclCollSupportsRing(ncclFuncAllReduce));
  EXPECT_TRUE(rcclCollSupportsRing(ncclFuncAllGather));
  EXPECT_TRUE(rcclCollSupportsRing(ncclFuncReduceScatter));
  EXPECT_TRUE(rcclCollSupportsRing(ncclFuncBroadcast));
  EXPECT_TRUE(rcclCollSupportsRing(ncclFuncReduce));
}

TEST(WrapMicrotest, CollSupportsRing_FalseForP2pAndAlltoall) {
  EXPECT_FALSE(rcclCollSupportsRing(ncclFuncSendRecv));
  EXPECT_FALSE(rcclCollSupportsRing(ncclFuncAlltoAll));
}

// ===========================================================================
// validHsaScratchEnvSetting -- no ncclComm at all, pure function of its four
// arguments. rccl_wrap.cc:1735-1748.
// ===========================================================================

TEST(WrapMicrotest, ValidHsaScratchEnv_ExplicitEnvOverridesEverything) {
  // hsaScratchEnv == "1" short-circuits true regardless of arch/version, even
  // values that would otherwise fail every arch-specific check below.
  EXPECT_TRUE(validHsaScratchEnvSetting("1", /*hipRuntimeVersion=*/0, /*firmwareVersion=*/0, "gfx950"));
}

TEST(WrapMicrotest, ValidHsaScratchEnv_Gfx950FirmwareBoundary) {
  EXPECT_TRUE(validHsaScratchEnvSetting(nullptr, 60443484, 24, "gfx950"));
  EXPECT_FALSE(validHsaScratchEnvSetting(nullptr, 60443484, 23, "gfx950"));
  // The check is an AND of two independent thresholds; the case above only
  // ever varies firmwareVersion, so it never proves the hipRuntimeVersion
  // side is checked at all.
  EXPECT_FALSE(validHsaScratchEnvSetting(nullptr, 0, 999, "gfx950"));
}

TEST(WrapMicrotest, ValidHsaScratchEnv_Gfx942FirmwareBoundary) {
  EXPECT_TRUE(validHsaScratchEnvSetting(nullptr, 60443484, 177, "gfx942"));
  EXPECT_FALSE(validHsaScratchEnvSetting(nullptr, 60443484, 176, "gfx942"));
  EXPECT_FALSE(validHsaScratchEnvSetting(nullptr, 0, 999, "gfx942"));
}

TEST(WrapMicrotest, ValidHsaScratchEnv_UnlistedArchDefaultsTrue) {
  EXPECT_TRUE(validHsaScratchEnvSetting(nullptr, 0, 0, "gfx1100"));
}

TEST(WrapMicrotest, ValidHsaScratchEnv_EnvSetButNotOne_FallsThroughToArchCheck) {
  // "0" fails the strcmp(..., "1") == 0 check, so this exercises the
  // hsaScratchEnvSet==false branch of the OR just as much as nullptr does --
  // distinct from ValidHsaScratchEnv_Gfx950FirmwareBoundary only in showing
  // that a non-"1" string takes the same path as "unset".
  EXPECT_FALSE(validHsaScratchEnvSetting("0", 60443484, 23, "gfx950"));
}

// ===========================================================================
// rcclIsArchSupportedForFunc -- no ncclComm; takes ncclTaskColl* + archName.
// rccl_wrap.cc:1751-1771. Should match get_arch_guard() in generate.py per
// the production comment -- out of scope here (Python, not host-C++-testable
// from this binary).
// ===========================================================================

namespace {
ncclTaskColl MakeTask(int protocol, bool hasAcc) {
  ncclTaskColl task{};
  task.protocol = protocol;
  static int accSentinel = 0;
  task.acc = hasAcc ? &accSentinel : nullptr;
  return task;
}
}  // namespace

// ENABLE_LL128's state depends on the build configuration; both arms are
// written so whichever compiles in is exercised. The OCI cluster build has
// ENABLE_LL128 defined, so *_LL128_AccGatesOutGfx90a and its siblings are the
// ones that compile and run there; a local ROCm-7.0.0 build does not define
// it, which is what compiles and runs *_LL128_DisabledAtCompileTime instead --
// both arms are now verified, one per build.
#if defined(ENABLE_LL128)
TEST(WrapMicrotest, IsArchSupportedForFunc_LL128_AccGatesOutGfx90a) {
  // With acc set, gfx90a is EXCLUDED from the LL128+acc allow-list (only
  // gfx942/gfx950/gfx1250) even though it IS allowed for LL128 without acc --
  // acc is not just an extra restriction on top of the non-acc set, it swaps
  // which archs are supported entirely.
  ncclTaskColl withAcc = MakeTask(NCCL_PROTO_LL128, /*hasAcc=*/true);
  ncclTaskColl noAcc = MakeTask(NCCL_PROTO_LL128, /*hasAcc=*/false);
  EXPECT_FALSE(rcclIsArchSupportedForFunc(&withAcc, "gfx90a"));
  EXPECT_TRUE(rcclIsArchSupportedForFunc(&noAcc, "gfx90a"));
}

TEST(WrapMicrotest, IsArchSupportedForFunc_LL128_AccAllowsGfx942Gfx950Gfx1250) {
  // The positive side of the acc allow-list (gfx942/gfx950/gfx1250): the
  // AccGatesOutGfx90a test above only exercises archs that fall through this
  // OR-chain to false, so it never proves any of the three actually matches.
  // Each is tested individually since it's an OR-chain: matching later in the
  // chain doesn't prove an earlier member's own comparison ever ran.
  ncclTaskColl withAcc = MakeTask(NCCL_PROTO_LL128, /*hasAcc=*/true);
  EXPECT_TRUE(rcclIsArchSupportedForFunc(&withAcc, "gfx942"));
  EXPECT_TRUE(rcclIsArchSupportedForFunc(&withAcc, "gfx950"));
  EXPECT_TRUE(rcclIsArchSupportedForFunc(&withAcc, "gfx1250"));
}

TEST(WrapMicrotest, IsArchSupportedForFunc_LL128_NoAcc_AllListedArchsAndUnsupported) {
  // noAcc allow-list is gfx942/gfx950/gfx90a/gfx1250; the AccGatesOutGfx90a
  // test above only ever matches on gfx90a (the third member), so gfx942 and
  // gfx950 -- the first two -- are otherwise never proven to match on their
  // own comparison. A completely unlisted arch closes the all-false side.
  ncclTaskColl noAcc = MakeTask(NCCL_PROTO_LL128, /*hasAcc=*/false);
  EXPECT_TRUE(rcclIsArchSupportedForFunc(&noAcc, "gfx942"));
  EXPECT_TRUE(rcclIsArchSupportedForFunc(&noAcc, "gfx950"));
  EXPECT_TRUE(rcclIsArchSupportedForFunc(&noAcc, "gfx1250"));
  EXPECT_FALSE(rcclIsArchSupportedForFunc(&noAcc, "gfx1100"));
}
#else
TEST(WrapMicrotest, IsArchSupportedForFunc_LL128_DisabledAtCompileTime) {
  // ENABLE_LL128 not defined in this build config: the outer `if` still
  // matches on protocol == NCCL_PROTO_LL128, but its #else arm explicitly
  // sets `supported = false` -- not left at the `true` initializer. False
  // regardless of arch or acc, since the whole allow-list logic is compiled
  // out along with the #if block that would otherwise set it.
  ncclTaskColl withAcc = MakeTask(NCCL_PROTO_LL128, /*hasAcc=*/true);
  ncclTaskColl noAcc = MakeTask(NCCL_PROTO_LL128, /*hasAcc=*/false);
  EXPECT_FALSE(rcclIsArchSupportedForFunc(&withAcc, "gfx90a"));
  EXPECT_FALSE(rcclIsArchSupportedForFunc(&noAcc, "gfx942"));
}
#endif

TEST(WrapMicrotest, IsArchSupportedForFunc_NonLL128_AccRestrictsToGfx9xAnd1250) {
  ncclTaskColl withAcc = MakeTask(NCCL_PROTO_SIMPLE, /*hasAcc=*/true);
  EXPECT_FALSE(rcclIsArchSupportedForFunc(&withAcc, "gfx90a"));
  EXPECT_TRUE(rcclIsArchSupportedForFunc(&withAcc, "gfx942"));
  // gfx950 and gfx1250 are this allow-list's later OR members; the gfx942
  // check above short-circuits before ever reaching either.
  EXPECT_TRUE(rcclIsArchSupportedForFunc(&withAcc, "gfx950"));
  EXPECT_TRUE(rcclIsArchSupportedForFunc(&withAcc, "gfx1250"));
}

TEST(WrapMicrotest, IsArchSupportedForFunc_NonLL128_NoAcc_AlwaysSupported) {
  // Neither guarded branch entered: `supported` stays at its `true`
  // initializer unconditionally.
  ncclTaskColl noAcc = MakeTask(NCCL_PROTO_SIMPLE, /*hasAcc=*/false);
  EXPECT_TRUE(rcclIsArchSupportedForFunc(&noAcc, "gfx90a"));
}

// ===========================================================================
// rcclGetAlgoName -- no ncclComm; pure lookup over `algo`.
// rccl_wrap.cc:586-637. Delegates to the real ncclAlgoToString() for native
// (< NCCL_NUM_ALGORITHMS) values; wrap_stubs.cc does NOT stub that function
// (it's genuinely faked with a faithful copy of collectives.cc's switch, to
// avoid pulling collectives.cc's DDA/sym/nvtx dependency chain into this
// lean binary -- see fakes/wrap_stubs.cc).
// ===========================================================================

TEST(WrapMicrotest, GetAlgoName_NegativeIsInvalidArgument) {
  const char* name = nullptr;
  EXPECT_EQ(ncclInvalidArgument, rcclGetAlgoName(-1, &name));
}

TEST(WrapMicrotest, GetAlgoName_AtRcclAlgoCountIsInvalidArgument) {
  // RCCL_ALGO_COUNT is the enum's one-past-the-end sentinel; the outer guard
  // (`algo >= RCCL_ALGO_COUNT`) rejects it before the inner switch runs.
  //
  // Residual: an `>=` -> `>` mutant of this guard is accepted as equivalent,
  // not fixed. At algo == RCCL_ALGO_COUNT, the mutated guard lets control
  // fall into the inner switch's own `default:` arm, which prints the
  // identical WARN text and returns the identical ncclInvalidArgument -- no
  // input distinguishes the two guards, so this assertion cannot catch it.
  const char* name = nullptr;
  EXPECT_EQ(ncclInvalidArgument, rcclGetAlgoName(rcclAddonAlgos_t::RCCL_ALGO_COUNT, &name));
}

TEST(WrapMicrotest, GetAlgoName_NativeAlgoDelegatesToNcclAlgoToString) {
  const char* name = nullptr;
  ASSERT_EQ(ncclSuccess, rcclGetAlgoName(NCCL_ALGO_RING, &name));
  EXPECT_STREQ("RING", name);
}

TEST(WrapMicrotest, GetAlgoName_AddonValues_DistinctStrings) {
  const char* name = nullptr;
  ASSERT_EQ(ncclSuccess, rcclGetAlgoName(rcclAddonAlgos_t::RCCL_DIRECT_ALLGATHER, &name));
  EXPECT_STREQ("Direct", name);

  ASSERT_EQ(ncclSuccess, rcclGetAlgoName(rcclAddonAlgos_t::RCCL_HIERARCHICAL_ALLGATHER, &name));
  EXPECT_STREQ("Hier", name);

  ASSERT_EQ(ncclSuccess, rcclGetAlgoName(rcclAddonAlgos_t::RCCL_DIRECT_REDUCESCATTER, &name));
  EXPECT_STREQ("Direct", name);

  ASSERT_EQ(ncclSuccess, rcclGetAlgoName(rcclAddonAlgos_t::RCCL_HIERARCHICAL_REDUCESCATTER, &name));
  EXPECT_STREQ("Hier", name);

  ASSERT_EQ(ncclSuccess, rcclGetAlgoName(rcclAddonAlgos_t::RCCL_SYMMETRIC, &name));
  EXPECT_STREQ("SYM", name);

  ASSERT_EQ(ncclSuccess, rcclGetAlgoName(rcclAddonAlgos_t::RCCL_CE_2SHOT, &name));
  EXPECT_STREQ("CE2", name);

  ASSERT_EQ(ncclSuccess, rcclGetAlgoName(rcclAddonAlgos_t::RCCL_CE_REGISTERED, &name));
  EXPECT_STREQ("CE", name);
}

// The three DDA-fabric variants (LL / LL128 / VMM) deliberately alias to the
// same string ("protocol column distinguishes LL/LL128/Simple" per the
// production comment) -- assert at least two of the three explicitly so a
// mutant that maps one of them to a DIFFERENT wrong string (rather than just
// "DDA") is still caught, not just a mutant that breaks the alias entirely.
TEST(WrapMicrotest, GetAlgoName_DdaFabricVariantsAllAliasToDda) {
  const char* name = nullptr;
  ASSERT_EQ(ncclSuccess, rcclGetAlgoName(rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL, &name));
  EXPECT_STREQ("DDA", name);
  ASSERT_EQ(ncclSuccess, rcclGetAlgoName(rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL128, &name));
  EXPECT_STREQ("DDA", name);
  ASSERT_EQ(ncclSuccess, rcclGetAlgoName(rcclAddonAlgos_t::RCCL_DDA_IPC, &name));
  EXPECT_STREQ("DDA-IPC", name);  // NOT aliased with the fabric variants above.
}

// Dead code, not a coverage gap: the inner switch's `default:` (rccl_wrap.cc
// :629-631) can never run. RCCL_ALGO_COUNT is exactly one past the last named
// enum value, contiguous with NCCL_NUM_ALGORITHMS, and the outer guard above
// already rejects every algo outside [0, RCCL_ALGO_COUNT) -- so every value
// that reaches this switch is one of the named cases. Confirmed via
// llvm-cov: 0 hits on this arm is expected, not a test to add.

// ===========================================================================
// rcclGetProtocolName -- rccl_wrap.cc:639-646.
// ===========================================================================

TEST(WrapMicrotest, GetProtocolName_NegativeIsInvalidArgument) {
  const char* name = nullptr;
  EXPECT_EQ(ncclInvalidArgument, rcclGetProtocolName(-1, &name));
}

TEST(WrapMicrotest, GetProtocolName_AtNumProtocolsIsInvalidArgument) {
  const char* name = nullptr;
  EXPECT_EQ(ncclInvalidArgument, rcclGetProtocolName(NCCL_NUM_PROTOCOLS, &name));
}

TEST(WrapMicrotest, GetProtocolName_ValidValuesDelegateToNcclProtoToString) {
  const char* name = nullptr;
  ASSERT_EQ(ncclSuccess, rcclGetProtocolName(NCCL_PROTO_LL, &name));
  EXPECT_STREQ("LL", name);
  ASSERT_EQ(ncclSuccess, rcclGetProtocolName(NCCL_PROTO_LL128, &name));
  EXPECT_STREQ("LL128", name);
  ASSERT_EQ(ncclSuccess, rcclGetProtocolName(NCCL_PROTO_SIMPLE, &name));
  EXPECT_STREQ("SIMPLE", name);
}

// ===========================================================================
// rcclGetAlgoProtoIndex -- rccl_wrap.cc:191-207.
// ===========================================================================

TEST(WrapMicrotest, GetAlgoProtoIndex_NullEnvStrIsInvalidUsage) {
  const char* table[] = {"LL", "LL128", "SIMPLE"};
  int result = -99;
  EXPECT_EQ(ncclInvalidUsage, rcclGetAlgoProtoIndex(nullptr, table, 3, result));
  EXPECT_EQ(-99, result);  // untouched: the null-envStr arm never assigns it.
}

TEST(WrapMicrotest, GetAlgoProtoIndex_CaseInsensitiveMatchWritesIndex) {
  const char* table[] = {"LL", "LL128", "SIMPLE"};
  int result = -99;
  EXPECT_EQ(ncclSuccess, rcclGetAlgoProtoIndex("ll128", table, 3, result));
  EXPECT_EQ(1, result);
}

// static bool failedProtoWarn is a once-per-process latch (rccl_wrap.cc:199-
// 204): the WARN only fires on the first unmatched string any test in this
// binary passes in; every later mismatch silently returns ncclInvalidUsage
// with no log line. RUN_ISOLATED_TEST forks a fresh process so this is the
// first (and only) call in that image, making the WARN observable.
TEST(WrapMicrotestIsolated, GetAlgoProtoIndex_UnmatchedStringWarnsOnce) {
  RUN_ISOLATED_TEST(
      "Wrap_GetAlgoProtoIndex_UnmatchedStringWarnsOnce",
      []() {
        const char* table[] = {"LL", "LL128", "SIMPLE"};
        int result = -99;
        ncclResult_t r = ncclSuccess;
        const std::string err = RcclUnitTesting::CaptureLog([&]() { r = rcclGetAlgoProtoIndex("bogus", table, 3, result); });
        ASSERT_EQ(ncclInvalidUsage, r);
        EXPECT_EQ(-99, result);
        EXPECT_NE(std::string::npos, err.find("Invalid algo or protocol string passed bogus"));
      });
}

// ===========================================================================
// rcclUseAlltoAllGda -- rccl_wrap.cc:669-678.
// ===========================================================================

TEST(WrapMicrotest, UseAlltoAllGda_DefaultBuildAlwaysFalse) {
  // ENABLE_ROCSHMEM is OFF by default (CMakeLists.txt option default) and not
  // turned on for this microtest binary, so the entire `#ifdef
  // ENABLE_ROCSHMEM` guarded block -- including the enableRocshmem/
  // rocshmemThreshold fields themselves, which don't exist on ncclComm at
  // all in this build -- compiles out, and every input takes the
  // unconditional `return false;` tail. The true-returning branch is
  // Hardware/Structural (needs a real rocSHMEM build) -- documented here as
  // the ceiling for this build, not contrived.
  ncclComm* comm = MakeZeroedComm();
  comm->nNodes = 2;
  comm->nRanks = 16;
  EXPECT_FALSE(rcclUseAlltoAllGda(comm));
  delete comm;
}

// Second isolated case: makes TWO unmatched-string calls in the SAME
// process image, pinning that the latch actually suppresses the WARN on
// the second call rather than firing every time. A mutant deleting the
// failedProtoWarn assignment survives the single-call isolated test
// above but is killed here.
TEST(WrapMicrotestIsolated, GetAlgoProtoIndex_SecondUnmatchedCallStaysSilent) {
  RUN_ISOLATED_TEST(
      "Wrap_GetAlgoProtoIndex_SecondUnmatchedCallStaysSilent",
      []() {
        const char* table[] = {"LL", "LL128", "SIMPLE"};
        int result = -99;
        rcclGetAlgoProtoIndex("bogus", table, 3, result);  // primes the latch
        ncclResult_t r = ncclSuccess;
        const std::string err =
          RcclUnitTesting::CaptureLog([&]() { r = rcclGetAlgoProtoIndex("alsobogus", table, 3, result); });
        ASSERT_EQ(ncclInvalidUsage, r);
        EXPECT_TRUE(err.empty()) << "expected the warn-once latch to suppress this WARN, got: " << err;
      });
}

// ===========================================================================
// rcclHierarchicalTempBufferSize -- pure function of (nNodes, allGather,
// reduceScatter), no ncclComm at all. rccl_wrap.cc:680-702.
// ===========================================================================

TEST(WrapMicrotest, HierarchicalTempBufferSize_AllGatherThresholds) {
  EXPECT_EQ(0u, rcclHierarchicalTempBufferSize(7, /*allGather=*/true, /*reduceScatter=*/false));
  EXPECT_EQ(HIERARCHICAL_TEMP_BUFFER_SIZE / 4, rcclHierarchicalTempBufferSize(8, true, false));
  EXPECT_EQ(HIERARCHICAL_TEMP_BUFFER_SIZE / 4, rcclHierarchicalTempBufferSize(15, true, false));
  EXPECT_EQ(HIERARCHICAL_TEMP_BUFFER_SIZE / 2, rcclHierarchicalTempBufferSize(16, true, false));
  EXPECT_EQ(HIERARCHICAL_TEMP_BUFFER_SIZE / 2, rcclHierarchicalTempBufferSize(31, true, false));
  EXPECT_EQ(HIERARCHICAL_TEMP_BUFFER_SIZE, rcclHierarchicalTempBufferSize(32, true, false));
}

TEST(WrapMicrotest, HierarchicalTempBufferSize_ReduceScatterThresholds) {
  EXPECT_EQ(0u, rcclHierarchicalTempBufferSize(7, /*allGather=*/false, /*reduceScatter=*/true));
  EXPECT_EQ(HIERARCHICAL_TEMP_BUFFER_SIZE / 2, rcclHierarchicalTempBufferSize(8, false, true));
  EXPECT_EQ(HIERARCHICAL_TEMP_BUFFER_SIZE / 2, rcclHierarchicalTempBufferSize(15, false, true));
  EXPECT_EQ(HIERARCHICAL_TEMP_BUFFER_SIZE, rcclHierarchicalTempBufferSize(16, false, true));
}

// nNodes=9: the allGather arm alone gives 32MB (>=8,<16); the reduceScatter arm
// alone gives 64MB (>=8). Only if BOTH arms actually ran and std::max compared
// them does the result come out as reduceScatter's 64MB -- a test that only
// ever set one flag could not tell max() from "last write wins".
TEST(WrapMicrotest, HierarchicalTempBufferSize_TakesMaxOfBoth) {
  EXPECT_EQ(HIERARCHICAL_TEMP_BUFFER_SIZE / 2, rcclHierarchicalTempBufferSize(9, true, true));
}

TEST(WrapMicrotest, HierarchicalTempBufferSize_NeitherFlagIsZero) {
  EXPECT_EQ(0u, rcclHierarchicalTempBufferSize(64, false, false));
}

// ===========================================================================
// rcclCeAllReduceGraphLatchTick / rcclCeAllReduceAllowed -- plain ncclComm
// field access, no topology. rccl_wrap.cc:834-857.
// ===========================================================================

TEST(WrapMicrotest, CeAllReduceGraphLatchTick_CapturingSetsLatch) {
  ncclComm* comm = MakeZeroedComm();
  rcclCeAllReduceGraphLatchTick(comm, /*ceCapturing=*/true);
  EXPECT_TRUE(comm->ceColl.graphModeSeen);
  delete comm;
}

// Latch must stay set while still capturing even if localPersistentRefs has
// already dropped to 0 -- the clear-condition's other half (!ceCapturing) is
// what actually gates it, not localPersistentRefs alone.
TEST(WrapMicrotest, CeAllReduceGraphLatchTick_CapturingStaysLatchedRegardlessOfRefs) {
  ncclComm* comm = MakeZeroedComm();
  comm->ceColl.graphModeSeen = true;
  comm->localPersistentRefs = 0;
  rcclCeAllReduceGraphLatchTick(comm, /*ceCapturing=*/true);
  EXPECT_TRUE(comm->ceColl.graphModeSeen);
  delete comm;
}

TEST(WrapMicrotest, CeAllReduceGraphLatchTick_ClearsWhenNotCapturingAndNoRefs) {
  ncclComm* comm = MakeZeroedComm();
  comm->ceColl.graphModeSeen = true;
  comm->localPersistentRefs = 0;
  rcclCeAllReduceGraphLatchTick(comm, /*ceCapturing=*/false);
  EXPECT_FALSE(comm->ceColl.graphModeSeen);
  delete comm;
}

TEST(WrapMicrotest, CeAllReduceGraphLatchTick_StaysLatchedWhileRefsLive) {
  ncclComm* comm = MakeZeroedComm();
  comm->ceColl.graphModeSeen = true;
  comm->localPersistentRefs = 1;
  rcclCeAllReduceGraphLatchTick(comm, /*ceCapturing=*/false);
  EXPECT_TRUE(comm->ceColl.graphModeSeen);
  delete comm;
}

// Not capturing, and the latch was never set: the else-if's `&&` short-
// circuits on its first operand (graphModeSeen == false) without ever
// evaluating localPersistentRefs -- distinct from the case above, where the
// first operand is true and the second is what stops the clear.
TEST(WrapMicrotest, CeAllReduceGraphLatchTick_NoopWhenNeverLatchedAndNotCapturing) {
  ncclComm* comm = MakeZeroedComm();
  comm->ceColl.graphModeSeen = false;
  comm->localPersistentRefs = 0;
  rcclCeAllReduceGraphLatchTick(comm, /*ceCapturing=*/false);
  EXPECT_FALSE(comm->ceColl.graphModeSeen);
  delete comm;
}

TEST(WrapMicrotest, CeAllReduceAllowed_TrueWhenLatchClear) {
  ncclComm* comm = MakeZeroedComm();
  comm->ceColl.graphModeSeen = false;
  EXPECT_TRUE(rcclCeAllReduceAllowed(comm));
  delete comm;
}

TEST(WrapMicrotest, CeAllReduceAllowed_FalseWhenLatchSet) {
  ncclComm* comm = MakeZeroedComm();
  comm->ceColl.graphModeSeen = true;
  EXPECT_FALSE(rcclCeAllReduceAllowed(comm));
  delete comm;
}

// ===========================================================================
// rcclSetPxn / rcclSetP2pNetChunkSize -- rccl_wrap.cc:1368-1413. Both read a
// real environment variable via plain getenv() on the "not yet cached" path.
// This batch covers only the already-cached fast return (comm->pxnDisable /
// comm->p2pNetChunkSize already != RCCL_VALUE_UNSET), which never touches
// getenv at all -- the env-reading arch/rank computation is Structural,
// deferred to a future batch that adds a real env-controlling seam rather
// than letting this test read whatever is actually set in the environment.
// ===========================================================================

TEST(WrapMicrotest, SetPxn_AlreadyCachedReturnsStoredValueUnchanged) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  // 7 is outside {RCCL_VALUE_INVALID(-1), 0, 1}, the only values the
  // fall-through arch/rank computation can ever produce for this comm. A
  // value from that set (e.g. 1, this comm's nRanks=1 < the 64 threshold
  // for gfx942 so the real computation also yields 1) would let this test
  // pass even if the cached-value guard were broken and execution fell
  // through -- which is exactly what happened until this was caught by
  // mutation testing.
  comm->pxnDisable = 7;  // already resolved by a prior call; not RCCL_VALUE_UNSET
  int rcclPxnDisable = -100;
  rcclSetPxn(comm, rcclPxnDisable);
  EXPECT_EQ(7, rcclPxnDisable);
  EXPECT_EQ(7, comm->pxnDisable);  // untouched: the cached-value arm never reassigns it
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetP2pNetChunkSize_AlreadyCachedReturnsStoredValueUnchanged) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  // 123456 is outside {RCCL_VALUE_INVALID(-1), 1<<17, 1<<18, 1<<19}, the only
  // values the fall-through arch/rank computation can ever produce. 1<<17
  // (this comm's nRanks=1 < the 64 threshold for gfx942) would coincide with
  // the real computation's output, masking a broken cached-value guard.
  comm->p2pNetChunkSize = 123456;
  int rcclP2pNetChunkSize = -100;
  rcclSetP2pNetChunkSize(comm, rcclP2pNetChunkSize);
  EXPECT_EQ(123456, rcclP2pNetChunkSize);
  EXPECT_EQ(123456, comm->p2pNetChunkSize);
  DeleteCommWithArch(comm);
}

// ===========================================================================
// rcclGetMaxNthreads -- rccl_wrap.cc:1605-1612.
// ===========================================================================

// RCCL_GFX950_MAX_NTHREADS, RCCL_DEFAULT_MAX_NTHREADS, and RCCL_LL_MAX_NTHREADS
// are all 256 today, so a value assertion alone cannot distinguish "took the
// gfx950 arm" from "took the else arm and the constants just happen to
// match" -- documented, not fixed: an equivalent-by-current-constants
// situation like the residuals below, not a gap in this test. Both calls are
// still made, so llvm-cov shows both arms as reached; NCCL_PROTO_LL's
// assignment is arch-independent and IS a real oracle either way.
//
// Pinned rather than left to rot silently: if any of the three constants
// ever diverges, this fails to compile, forcing whoever made that change to
// come back and give GetMaxNthreads_Gfx950Arch/NonGfx950Arch real
// distinguishing assertions instead of the coincidental ones below.
static_assert(RCCL_GFX950_MAX_NTHREADS == RCCL_DEFAULT_MAX_NTHREADS &&
                  RCCL_DEFAULT_MAX_NTHREADS == RCCL_LL_MAX_NTHREADS,
              "RCCL_GFX950_MAX_NTHREADS / RCCL_DEFAULT_MAX_NTHREADS / RCCL_LL_MAX_NTHREADS "
              "diverged -- update GetMaxNthreads_Gfx950Arch/NonGfx950Arch below to assert "
              "values that actually distinguish the gfx950 vs. default arm");

TEST(WrapMicrotest, GetMaxNthreads_Gfx950Arch) {
  ncclComm* comm = MakeCommWithArch("gfx950");
  int maxNthreads[NCCL_NUM_PROTOCOLS] = {0};
  rcclGetMaxNthreads(comm, maxNthreads);
  EXPECT_EQ(RCCL_GFX950_MAX_NTHREADS, maxNthreads[NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(RCCL_GFX950_MAX_NTHREADS, maxNthreads[NCCL_PROTO_LL128]);
  EXPECT_EQ(RCCL_LL_MAX_NTHREADS, maxNthreads[NCCL_PROTO_LL]);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, GetMaxNthreads_NonGfx950Arch) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  int maxNthreads[NCCL_NUM_PROTOCOLS] = {0};
  rcclGetMaxNthreads(comm, maxNthreads);
  EXPECT_EQ(RCCL_DEFAULT_MAX_NTHREADS, maxNthreads[NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(RCCL_DEFAULT_MAX_NTHREADS, maxNthreads[NCCL_PROTO_LL128]);
  EXPECT_EQ(RCCL_LL_MAX_NTHREADS, maxNthreads[NCCL_PROTO_LL]);
  DeleteCommWithArch(comm);
}

// ===========================================================================
// rcclSetDefaultBuffSizes -- rccl_wrap.cc:1644-1652. Isolated: its own
// maxNthreads[] is a function-local static, computed once per process and
// reused by every later call regardless of arch -- an ordinary (non-isolated)
// second test with a different arch would silently read the first test's
// cached values instead of recomputing. RUN_ISOLATED_TEST forks a fresh
// process so this is the only call in that image.
// ===========================================================================

TEST(WrapMicrotestIsolated, SetDefaultBuffSizes_Gfx942Arch) {
  RUN_ISOLATED_TEST(
      "Wrap_SetDefaultBuffSizes_Gfx942Arch",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx942");
        int defaultBuffSizes[NCCL_NUM_PROTOCOLS] = {0};
        rcclSetDefaultBuffSizes(comm, defaultBuffSizes);
        // gfx942 is not gfx950, so rcclGetMaxNthreads gives RCCL_DEFAULT_MAX_NTHREADS
        // for LL128/SIMPLE and RCCL_LL_MAX_NTHREADS for LL; gfx942 is also not
        // gfx1250, so rcclLL128ElemsPerThreadFromArch's linesPerThread is 4.
        const int linesPerThread = 4;
        const int ll128DataElems = rcclLL128ElemsPerThreadFromArch("gfx942") / linesPerThread;
        EXPECT_EQ(NCCL_LL_LINES_PER_THREAD * RCCL_LL_MAX_NTHREADS * NCCL_STEPS * (int)sizeof(union ncclLLFifoLine),
                  defaultBuffSizes[NCCL_PROTO_LL]);
        EXPECT_EQ(linesPerThread * ll128DataElems * RCCL_DEFAULT_MAX_NTHREADS * NCCL_STEPS * (int)sizeof(uint64_t),
                  defaultBuffSizes[NCCL_PROTO_LL128]);
        EXPECT_EQ(1 << 22, defaultBuffSizes[NCCL_PROTO_SIMPLE]);
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclFuncMaxSendRecvCount -- rccl_wrap.cc:1654-1658. Thin wrapper delegating
// to the header-inline ncclFuncMaxSendRecvCount (enqueue.h); RCCL_EXPOSE_STATIC
// is unconditionally defined by rccl_vars.h unless something upstream already
// defined it otherwise, so RCCL_STATIC_EXPOSE_CHECK() compiles to a no-op here
// and the real computation always runs.
// ===========================================================================

TEST(WrapMicrotest, FuncMaxSendRecvCount_AllGatherMultipliesByNRanks) {
  size_t maxCount = 0;
  EXPECT_EQ(ncclSuccess, rcclFuncMaxSendRecvCount(ncclFuncAllGather, /*nRanks=*/8, /*count=*/100, maxCount));
  EXPECT_EQ(800u, maxCount);
}

TEST(WrapMicrotest, FuncMaxSendRecvCount_ReduceScatterMultipliesByNRanks) {
  size_t maxCount = 0;
  EXPECT_EQ(ncclSuccess, rcclFuncMaxSendRecvCount(ncclFuncReduceScatter, /*nRanks=*/4, /*count=*/50, maxCount));
  EXPECT_EQ(200u, maxCount);
}

TEST(WrapMicrotest, FuncMaxSendRecvCount_OtherFuncsReturnCountUnscaled) {
  size_t maxCount = 0;
  EXPECT_EQ(ncclSuccess, rcclFuncMaxSendRecvCount(ncclFuncAllReduce, /*nRanks=*/8, /*count=*/100, maxCount));
  EXPECT_EQ(100u, maxCount);
}

// ===========================================================================
// ParamDefaults_MatchProductionSource -- run-time counterpart to
// wrap_stubs.cc's static_asserts above. ncclParamMinNchannels/MaxNchannels,
// rcclParamForceCe, and ncclParamLaunchOrderImplicit hardcode the real
// NCCL_PARAM/RCCL_PARAM default they copy (see wrap_stubs.cc), but that
// default is an inline macro-argument literal with no separately importable
// constant -- unlike NCCL_NUM_ALGORITHMS or ncclNumFuncs, there's nothing a
// static_assert could check. Instead, this test reads the real
// graph/connect.cc / enqueue.cc source (via CMake-provided CONNECT_CC_PATH /
// ENQUEUE_CC_PATH, the same pattern as WRAP_CC_PATH) at run time and confirms
// the exact macro invocation text is still there: name, env var string, and
// default value all together, so renaming any part of it or changing the
// default both fail this test instead of going unnoticed.
// ===========================================================================

namespace {
std::string ReadFileOrDie(const char* path) {
  std::ifstream f(path);
  if (!f) {
    ADD_FAILURE() << "couldn't open " << path << " -- CONNECT_CC_PATH/ENQUEUE_CC_PATH stale vs the hipify tree?";
    return "";
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
}  // namespace

TEST(WrapMicrotest, ParamDefaults_MatchProductionSource) {
  const std::string connectCc = ReadFileOrDie(CONNECT_CC_PATH);
  const std::string enqueueCc = ReadFileOrDie(ENQUEUE_CC_PATH);

  EXPECT_NE(std::string::npos, connectCc.find(R"(NCCL_PARAM(MinNchannels, "MIN_NCHANNELS", -2))"))
      << "graph/connect.cc's NCCL_PARAM(MinNchannels...) changed -- update ncclParamMinNchannels() in wrap_stubs.cc";
  EXPECT_NE(std::string::npos, connectCc.find(R"(NCCL_PARAM(MaxNchannels, "MAX_NCHANNELS", -2))"))
      << "graph/connect.cc's NCCL_PARAM(MaxNchannels...) changed -- update ncclParamMaxNchannels() in wrap_stubs.cc";
  EXPECT_NE(std::string::npos, enqueueCc.find(R"(RCCL_PARAM(ForceCe, "FORCE_CE", 1))"))
      << "enqueue.cc's RCCL_PARAM(ForceCe...) changed -- update rcclParamForceCe() in wrap_stubs.cc";
  EXPECT_NE(std::string::npos, enqueueCc.find(R"(NCCL_PARAM(LaunchOrderImplicit, "LAUNCH_ORDER_IMPLICIT", 0))"))
      << "enqueue.cc's NCCL_PARAM(LaunchOrderImplicit...) changed -- update ncclParamLaunchOrderImplicit() in wrap_stubs.cc";
}

// ===========================================================================
// symkHostRedOpToDev -- rccl_wrap.cc:527-541. Pure switch, no comm/topology
// setup needed at all.
// ===========================================================================

TEST(WrapMicrotest, SymkHostRedOpToDev_MapsEachOpToItsDeviceOp) {
  EXPECT_EQ((int)ncclDevSum, symkHostRedOpToDev(ncclSum));
  EXPECT_EQ((int)ncclDevProd, symkHostRedOpToDev(ncclProd));
  EXPECT_EQ((int)ncclDevMinMax, symkHostRedOpToDev(ncclMin));
  EXPECT_EQ((int)ncclDevMinMax, symkHostRedOpToDev(ncclMax));
  EXPECT_EQ((int)ncclDevSumPostDiv, symkHostRedOpToDev(ncclAvg));
}

TEST(WrapMicrotest, SymkHostRedOpToDev_UnknownOpReturnsNegativeOne) {
  EXPECT_EQ(-1, symkHostRedOpToDev((ncclRedOp_t)9999));
}

// ===========================================================================
// rcclUpdateCollectiveProtocol -- rccl_wrap.cc:109-189. Caches getenv(
// "NCCL_PROTO") in a function-local static, so every case runs isolated.
// Covers the top-level user-override gate, one arch/size LL-threshold arm
// (gfx950 AllGather -- the gfx950/gfx942 ReduceScatter arms right below it
// are the same shape with different constants, not re-verified here), the
// gfx120x delegation + NCCL_P2P_DISABLE override (exercises the ncclGetEnv
// seam), and the nNodes>=2 minMaxLLRange-driven arm including its
// warn-once undefined-tuning fallback. ENABLE_LL128 is off in this build
// (confirmed via MICROTEST_README.md's build-config note), so that nested
// arm is out of scope here, same as rcclIsArchSupportedForFunc's precedent.
// ===========================================================================

TEST(WrapMicrotestIsolated, UpdateCollectiveProtocol_Gfx950AllGatherSmallSizeUsesLL) {
  RUN_ISOLATED_TEST(
      "Wrap_UpdateCollectiveProtocol_Gfx950AllGatherSmallSizeUsesLL",
      []() {
        SetMicroEnvAbsent("NCCL_PROTO");
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 1;
        comm->nRanks = 1;
        ncclTaskColl info{};
        info.func = ncclFuncAllGather;
        info.protocol = NCCL_PROTO_SIMPLE;
        rcclUpdateCollectiveProtocol(comm, /*nBytes=*/1024, &info);
        EXPECT_EQ(NCCL_PROTO_LL, info.protocol);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UpdateCollectiveProtocol_UserOverrideLeavesProtocolUntouched) {
  RUN_ISOLATED_TEST(
      "Wrap_UpdateCollectiveProtocol_UserOverrideLeavesProtocolUntouched",
      []() {
        SetMicroEnv("NCCL_PROTO", "LL128"); // any value: presence alone is the gate
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 1;
        comm->nRanks = 1;
        ncclTaskColl info{};
        info.func = ncclFuncAllGather;
        info.protocol = NCCL_PROTO_SIMPLE;
        rcclUpdateCollectiveProtocol(comm, /*nBytes=*/1024, &info);
        EXPECT_EQ(NCCL_PROTO_SIMPLE, info.protocol); // untouched
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UpdateCollectiveProtocol_Gfx120xDelegatesThenP2pDisableForcesSimple) {
  RUN_ISOLATED_TEST(
      "Wrap_UpdateCollectiveProtocol_Gfx120xDelegatesThenP2pDisableForcesSimple",
      []() {
        SetMicroEnvAbsent("NCCL_PROTO");
        SetMicroEnv("NCCL_P2P_DISABLE", "1");
        ncclComm* comm = MakeCommWithArch("gfx1200");
        comm->nNodes = 1;
        comm->nRanks = 1;
        ncclTaskColl info{};
        info.func = ncclFuncAllGather;
        info.protocol = NCCL_PROTO_LL; // whatever rcclGetProtoForGfx120x would pick, P2P_DISABLE overrides it
        rcclUpdateCollectiveProtocol(comm, /*nBytes=*/1024, &info);
        EXPECT_EQ(NCCL_PROTO_SIMPLE, info.protocol);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UpdateCollectiveProtocol_MultiNodeUsesTunedLLRange) {
  RUN_ISOLATED_TEST(
      "Wrap_UpdateCollectiveProtocol_MultiNodeUsesTunedLLRange",
      []() {
        SetMicroEnvAbsent("NCCL_PROTO");
        ncclComm* comm = MakeCommWithArch("gfx90a"); // not gfx942/gfx950/gfx120x: falls through to the nNodes>=2 arm
        comm->nNodes = 2;
        comm->nRanks = 4;
        comm->minMaxLLRange[RCCL_AR_TUNABLE][NCCL_PROTO_LL][RCCL_PROTOCOL_MIN_IDX] = 0;
        comm->minMaxLLRange[RCCL_AR_TUNABLE][NCCL_PROTO_LL][RCCL_PROTOCOL_MAX_IDX] = 400;
        ncclTaskColl info{};
        info.func = ncclFuncAllReduce;
        info.protocol = NCCL_PROTO_SIMPLE;
        // AllReduce's sizePerRank is nBytes directly (rcclGetSizePerRank doesn't divide for AR).
        // Exactly at llMax: distinguishes the guard's <= from a plain <.
        rcclUpdateCollectiveProtocol(comm, /*nBytes=*/400, &info);
        EXPECT_EQ(NCCL_PROTO_LL, info.protocol);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UpdateCollectiveProtocol_UndefinedTuningWarnsOnceForSupportedArch) {
  RUN_ISOLATED_TEST(
      "Wrap_UpdateCollectiveProtocol_UndefinedTuningWarnsOnceForSupportedArch",
      []() {
        SetMicroEnvAbsent("NCCL_PROTO");
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 2;
        comm->nRanks = 4;
        // minMaxLLRange left zero-initialized: both llMax and ll128Max read as RCCL_LL_LIMITS_UNDEFINED.
        ncclTaskColl info{};
        info.func = ncclFuncAllReduce;
        std::string log1 = RcclUnitTesting::CaptureLog([&]() { rcclUpdateCollectiveProtocol(comm, 1024, &info); });
        EXPECT_NE(std::string::npos, log1.find("LL cutoff points not detected"));
        std::string log2 = RcclUnitTesting::CaptureLog([&]() { rcclUpdateCollectiveProtocol(comm, 1024, &info); });
        EXPECT_EQ(std::string::npos, log2.find("LL cutoff points not detected")); // warn-once latch
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclUpdateThreadThreshold -- rccl_wrap.cc:333-356. Caches its three-name
// getenv probe in a function-local static; isolated per case.
// ===========================================================================

TEST(WrapMicrotestIsolated, UpdateThreadThreshold_TunedValueScalesByNRanks) {
  RUN_ISOLATED_TEST(
      "Wrap_UpdateThreadThreshold_TunedValueScalesByNRanks",
      []() {
        SetMicroEnvAbsent("NCCL_THREAD_THRESHOLDS");
        SetMicroEnvAbsent("NCCL_MAX_NCHANNELS");
        SetMicroEnvAbsent("NCCL_MIN_NCHANNELS");
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 2;
        comm->nRanks = 4;
        comm->minMaxLLRange[RCCL_RS_TUNABLE][NCCL_PROTO_LL][RCCL_PROTOCOL_THREAD_THRESHOLD_IDX] = 10;
        ncclTaskColl info{};
        info.func = ncclFuncReduceScatter;
        info.protocol = NCCL_PROTO_LL;
        int threadThreshold = -1;
        rcclUpdateThreadThreshold(comm, /*nBytes=*/1024, &info, threadThreshold);
        EXPECT_EQ(40, threadThreshold); // 10 * nRanks(4)
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UpdateThreadThreshold_UserOverrideLeavesThresholdUntouched) {
  RUN_ISOLATED_TEST(
      "Wrap_UpdateThreadThreshold_UserOverrideLeavesThresholdUntouched",
      []() {
        SetMicroEnv("NCCL_THREAD_THRESHOLDS", "anything");
        SetMicroEnvAbsent("NCCL_MAX_NCHANNELS");
        SetMicroEnvAbsent("NCCL_MIN_NCHANNELS");
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 2;
        comm->nRanks = 4;
        comm->minMaxLLRange[RCCL_RS_TUNABLE][NCCL_PROTO_LL][RCCL_PROTOCOL_THREAD_THRESHOLD_IDX] = 10;
        ncclTaskColl info{};
        info.func = ncclFuncReduceScatter;
        info.protocol = NCCL_PROTO_LL;
        int threadThreshold = -1;
        rcclUpdateThreadThreshold(comm, /*nBytes=*/1024, &info, threadThreshold);
        EXPECT_EQ(-1, threadThreshold); // untouched
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclOverrideProtocol / rcclOverrideAlgorithm -- rccl_wrap.cc:279-331. Both
// cache their env var and its parsed table index in function-local statics;
// isolated per case. rcclOverrideAlgorithm is structurally identical (same
// shape, algorithm/protocol swapped), so only its unset-passthrough and
// successful-override arms are re-verified here.
//
// Mutation-testing note: rcclOverrideProtocol's `protoVal > NCCL_PROTO_UNDEF`
// guard (line 293) mutated to `>=` is an equivalent mutant -- protoVal only
// ever reaches this line as either a successfully-parsed index (always > -1)
// or after an early return on parse failure, so it can never actually equal
// NCCL_PROTO_UNDEF(-1) here. No input can distinguish `>` from `>=` at this
// point. Confirmed by re-applying the mutation directly against this build
// and observing all four tests below still pass, then reverting.
// ===========================================================================

TEST(WrapMicrotestIsolated, OverrideProtocol_UnsetEnvLeavesProtocolUntouched) {
  RUN_ISOLATED_TEST(
      "Wrap_OverrideProtocol_UnsetEnvLeavesProtocolUntouched",
      []() {
        SetMicroEnvAbsent("RCCL_OVERRIDE_PROTO");
        const char* protoStr[] = {"LL", "LL128", "SIMPLE"};
        float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {};
        ncclTaskColl info{};
        info.protocol = NCCL_PROTO_SIMPLE;
        EXPECT_EQ(ncclSuccess, rcclOverrideProtocol(protoStr, table, &info));
        EXPECT_EQ(NCCL_PROTO_SIMPLE, info.protocol);
      });
}

TEST(WrapMicrotestIsolated, OverrideProtocol_ValidMatchOverridesProtocol) {
  RUN_ISOLATED_TEST(
      "Wrap_OverrideProtocol_ValidMatchOverridesProtocol",
      []() {
        SetMicroEnv("RCCL_OVERRIDE_PROTO", "LL128");
        const char* protoStr[] = {"LL", "LL128", "SIMPLE"};
        float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {}; // all zero: not NCCL_ALGO_PROTO_IGNORE
        ncclTaskColl info{};
        info.algorithm = NCCL_ALGO_TREE;
        info.protocol = NCCL_PROTO_SIMPLE;
        EXPECT_EQ(ncclSuccess, rcclOverrideProtocol(protoStr, table, &info));
        EXPECT_EQ(NCCL_PROTO_LL128, info.protocol);
      });
}

TEST(WrapMicrotestIsolated, OverrideProtocol_IgnoredComboReturnsInternalError) {
  RUN_ISOLATED_TEST(
      "Wrap_OverrideProtocol_IgnoredComboReturnsInternalError",
      []() {
        SetMicroEnv("RCCL_OVERRIDE_PROTO", "LL128");
        const char* protoStr[] = {"LL", "LL128", "SIMPLE"};
        float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {};
        table[NCCL_ALGO_TREE][NCCL_PROTO_LL128] = NCCL_ALGO_PROTO_IGNORE;
        ncclTaskColl info{};
        info.func = ncclFuncAllReduce;
        info.algorithm = NCCL_ALGO_TREE;
        info.protocol = NCCL_PROTO_SIMPLE;
        std::string log = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_EQ(ncclInternalError, rcclOverrideProtocol(protoStr, table, &info)); });
        EXPECT_NE(std::string::npos, log.find("Failed to force unsupported protocol"));
        EXPECT_EQ(NCCL_PROTO_SIMPLE, info.protocol); // untouched
      });
}

TEST(WrapMicrotestIsolated, OverrideProtocol_UnmatchedStringReturnsInvalidUsage) {
  RUN_ISOLATED_TEST(
      "Wrap_OverrideProtocol_UnmatchedStringReturnsInvalidUsage",
      []() {
        SetMicroEnv("RCCL_OVERRIDE_PROTO", "bogus");
        const char* protoStr[] = {"LL", "LL128", "SIMPLE"};
        float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {};
        ncclTaskColl info{};
        info.protocol = NCCL_PROTO_SIMPLE;
        std::string log = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_EQ(ncclInvalidUsage, rcclOverrideProtocol(protoStr, table, &info)); });
        EXPECT_NE(std::string::npos, log.find("Invalid algo or protocol string"));
        EXPECT_EQ(NCCL_PROTO_SIMPLE, info.protocol); // untouched
      });
}

TEST(WrapMicrotestIsolated, OverrideAlgorithm_UnsetEnvLeavesAlgorithmUntouched) {
  RUN_ISOLATED_TEST(
      "Wrap_OverrideAlgorithm_UnsetEnvLeavesAlgorithmUntouched",
      []() {
        SetMicroEnvAbsent("RCCL_OVERRIDE_ALGO");
        const char* algoStr[] = {"TREE", "RING", "COLLNET_DIRECT", "COLLNET_CHAIN", "NVLS", "NVLS_TREE", "PAT"};
        float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {};
        ncclTaskColl info{};
        info.algorithm = NCCL_ALGO_TREE;
        EXPECT_EQ(ncclSuccess, rcclOverrideAlgorithm(algoStr, table, &info));
        EXPECT_EQ(NCCL_ALGO_TREE, info.algorithm);
      });
}

TEST(WrapMicrotestIsolated, OverrideAlgorithm_ValidMatchOverridesAlgorithm) {
  RUN_ISOLATED_TEST(
      "Wrap_OverrideAlgorithm_ValidMatchOverridesAlgorithm",
      []() {
        SetMicroEnv("RCCL_OVERRIDE_ALGO", "RING");
        const char* algoStr[] = {"TREE", "RING", "COLLNET_DIRECT", "COLLNET_CHAIN", "NVLS", "NVLS_TREE", "PAT"};
        float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {}; // all zero: not NCCL_ALGO_PROTO_IGNORE
        ncclTaskColl info{};
        info.algorithm = NCCL_ALGO_TREE;
        info.protocol = NCCL_PROTO_SIMPLE;
        EXPECT_EQ(ncclSuccess, rcclOverrideAlgorithm(algoStr, table, &info));
        EXPECT_EQ(NCCL_ALGO_RING, info.algorithm);
      });
}

// ===========================================================================
// rcclSetPxn / rcclSetP2pNetChunkSize -- remaining getenv-driven paths
// (rccl_wrap.cc:1368-1413; the cached fast-path was covered in an earlier
// batch). Neither caches across calls itself (comm->pxnDisable/
// p2pNetChunkSize is the cache, and each test starts from RCCL_VALUE_UNSET),
// so these run in-process rather than isolated -- just SetMicroEnv/
// ClearMicroEnv around each call.
// ===========================================================================

TEST(WrapMicrotest, SetPxn_UnsupportedArchReturnsInvalidRegardlessOfEnv) {
  ncclComm* comm = MakeCommWithArch("gfx90a");
  SetMicroEnvAbsent("NCCL_PXN_DISABLE");
  int rcclPxnDisable = -100;
  rcclSetPxn(comm, rcclPxnDisable);
  EXPECT_EQ(RCCL_VALUE_INVALID, rcclPxnDisable);
  ClearMicroEnv();
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetPxn_EnvPresentReturnsInvalidAndSetsCustCollFromValue) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  SetMicroEnv("NCCL_PXN_DISABLE", "0"); // present -- early-returns INVALID regardless of its own value
  int rcclPxnDisable = -100;
  rcclSetPxn(comm, rcclPxnDisable);
  EXPECT_EQ(RCCL_VALUE_INVALID, rcclPxnDisable);
  EXPECT_TRUE(comm->enableCustColl); // gfx942 && inputStr("0") && !atoi("0")==true
  ClearMicroEnv();
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetPxn_Gfx942AboveThresholdEnablesPxn) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nRanks = 64; // >= the gfx942 threshold
  SetMicroEnvAbsent("NCCL_PXN_DISABLE");
  int rcclPxnDisable = -100;
  rcclSetPxn(comm, rcclPxnDisable);
  EXPECT_EQ(0, rcclPxnDisable);
  EXPECT_TRUE(comm->enableCustColl); // enableCustColl = !pxnDisable = !0 = true
  ClearMicroEnv();
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetPxn_Gfx942MidRangeUsesGfx942ThresholdNotGfx950s) {
  // 40 is between the two archs' real thresholds (32 for gfx950, 64 for
  // gfx942): distinguishes "used the right arch's threshold" from a
  // threshold mix-up, which nRanks=64/31 alone (the tests below) can't.
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nRanks = 40;
  SetMicroEnvAbsent("NCCL_PXN_DISABLE");
  int rcclPxnDisable = -100;
  rcclSetPxn(comm, rcclPxnDisable);
  EXPECT_EQ(1, rcclPxnDisable); // 40 < 64 (gfx942's real threshold)
  ClearMicroEnv();
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetPxn_Gfx950BelowThresholdDisablesPxn) {
  ncclComm* comm = MakeCommWithArch("gfx950");
  comm->nRanks = 31; // below the gfx950 threshold (32)
  SetMicroEnvAbsent("NCCL_PXN_DISABLE");
  int rcclPxnDisable = -100;
  rcclSetPxn(comm, rcclPxnDisable);
  EXPECT_EQ(1, rcclPxnDisable);
  EXPECT_FALSE(comm->enableCustColl); // enableCustColl = !pxnDisable = !1 = false
  ClearMicroEnv();
  DeleteCommWithArch(comm);
}

// rcclSetP2pNetChunkSize's final `else WARN(...)` arm (rccl_wrap.cc:1407-1409)
// is dead: reaching it requires archGfx942 and archGfx950 both false, but
// the guard three lines above already returns early whenever neither arch
// matches. Classified Dead, not contrived, same convention as
// rcclGetAlgoName's documented unreachable default.
TEST(WrapMicrotest, SetP2pNetChunkSize_UnsupportedArchReturnsInvalidRegardlessOfEnv) {
  ncclComm* comm = MakeCommWithArch("gfx90a");
  SetMicroEnvAbsent("NCCL_P2P_NET_CHUNKSIZE");
  int rcclP2pNetChunkSize = -100;
  rcclSetP2pNetChunkSize(comm, rcclP2pNetChunkSize);
  EXPECT_EQ(RCCL_VALUE_INVALID, rcclP2pNetChunkSize);
  ClearMicroEnv();
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetP2pNetChunkSize_Gfx942AboveThresholdUsesLargeChunk) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nRanks = 64;
  SetMicroEnvAbsent("NCCL_P2P_NET_CHUNKSIZE");
  int rcclP2pNetChunkSize = -100;
  rcclSetP2pNetChunkSize(comm, rcclP2pNetChunkSize);
  EXPECT_EQ(1 << 19, rcclP2pNetChunkSize);
  ClearMicroEnv();
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetP2pNetChunkSize_Gfx950MidRangeUsesMidChunk) {
  ncclComm* comm = MakeCommWithArch("gfx950");
  comm->nRanks = 16; // >= 16, < 32: the middle tier
  SetMicroEnvAbsent("NCCL_P2P_NET_CHUNKSIZE");
  int rcclP2pNetChunkSize = -100;
  rcclSetP2pNetChunkSize(comm, rcclP2pNetChunkSize);
  EXPECT_EQ(1 << 18, rcclP2pNetChunkSize);
  ClearMicroEnv();
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetP2pNetChunkSize_Gfx950LowRangeUsesSmallChunk) {
  // 10 is below gfx950's real mid-tier threshold (16): distinguishes the
  // low tier from a threshold that drifted lower (nRanks=16 alone, the test
  // above, can't tell 16 from a mutated ">= 8").
  ncclComm* comm = MakeCommWithArch("gfx950");
  comm->nRanks = 10;
  SetMicroEnvAbsent("NCCL_P2P_NET_CHUNKSIZE");
  int rcclP2pNetChunkSize = -100;
  rcclSetP2pNetChunkSize(comm, rcclP2pNetChunkSize);
  EXPECT_EQ(1 << 17, rcclP2pNetChunkSize);
  ClearMicroEnv();
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetP2pNetChunkSize_EnvPresentReturnsInvalid) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  SetMicroEnv("NCCL_P2P_NET_CHUNKSIZE", "12345");
  int rcclP2pNetChunkSize = -100;
  rcclSetP2pNetChunkSize(comm, rcclP2pNetChunkSize);
  EXPECT_EQ(RCCL_VALUE_INVALID, rcclP2pNetChunkSize);
  ClearMicroEnv();
  DeleteCommWithArch(comm);
}
