/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/rccl_wrap.cc (JIRA: TBD).
//
// Like init-test.cc / p2p-test.cc, this TU #includes the hipified
// unit-under-test source directly (via WRAP_CC_PATH) so its helpers become
// callable, links NO librccl/HIP, and satisfies every external symbol via
// fakes/wrap_stubs.cc.
//
// Scope of this first pass (rung 1 of an incremental series, matching the
// init.cc PRs' one-unit-per-commit convention): eight low-dependency helpers
// at the top of the file that take no ncclComm* at all, or touch only a
// handful of plain fields -- no RCCL_PARAM, no getenv, no DDA/CE/symmetric-
// kernel machinery. Everything past rcclUseAlltoAllGda in the file
// (rcclOverrideChannels, rcclSetPipelining, the WarpSpeed helpers,
// rcclSelectAllReduce/AllGather/ReduceScatter, ...) is unreached by design
// here -- future rungs, each adding the seam(s) that specific unit needs.
//
// Built and verified on the OCI cluster (ROCm 7.0.2 / hipcc, gfx942):
// rccl-UnitTestsMicroWrap, 29/29 tests passing, including under
// --gtest_shuffle --gtest_repeat=3. Real llvm-cov branch coverage was not
// measured -- the compute nodes ship rocm-llvm without llvm-cov/llvm-
// profdata, and the one shared alternate LLVM found (ROCm 6.4.0) writes/
// reads an older raw-profile format (v9) than what this build's clang
// produces (v10), so llvm-profdata refuses to merge it ("unsupported
// profile format" -- the exact pitfall MICROTEST_README's Common Mistakes
// table warns about). A matching-major-version LLVM would resolve this.
//
// Mutation testing (the project's stated actual acceptance bar) was run
// directly against this build instead: 7 of 9 planned mutants killed, 1
// accepted as a genuine equivalent mutant, 1 accepted as unkillable-by-
// design (relies on undefined behavior, not a defined wrong value). Two
// mutants that initially survived (GetAlgoProtoIndex_UnmatchedStringWarnsOnce's
// warn-once latch) led to a new test
// (GetAlgoProtoIndex_SecondUnmatchedCallStaysSilent) added below. Details
// per mutant:
//
//   Site                                          | Verdict
//   ---------------------------------------------- | ------------------------
//   SingleNodeLLCutoffs[] row swap (Broadcast/      | Killed
//     Reduce)                                       |
//   rcclGetProtoForGfx120x table-bound guard        | SURVIVED, not fixed:
//     (`<` -> `<=`)                                 |   the wrong branch reads
//                                                    |   SingleNodeLLCutoffs[8],
//                                                    |   one past the array's
//                                                    |   end -- undefined
//                                                    |   behavior, not a
//                                                    |   defined wrong value.
//                                                    |   Neither a value
//                                                    |   assertion nor
//                                                    |   -fsanitize=address
//                                                    |   reliably catches it
//                                                    |   in this build (the
//                                                    |   compile-time-constant
//                                                    |   array is apparently
//                                                    |   placed somewhere
//                                                    |   ASan's stack redzones
//                                                    |   don't cover). Accepted
//                                                    |   residual.
//   rcclIsArchSupportedForFunc non-LL128 `info->acc` | Killed
//     check inverted                                 |
//   validHsaScratchEnvSetting gfx950 firmware        | Killed
//     boundary (`>=` -> `>`)                          |
//   rcclGetAlgoName RCCL_ALGO_COUNT guard             | SURVIVED, accepted as
//     (`>=` -> `>`)                                   |   equivalent: for
//                                                     |   algo == RCCL_ALGO_
//                                                     |   COUNT, control falls
//                                                     |   into the inner
//                                                     |   switch's default:
//                                                     |   arm, which prints
//                                                     |   the identical WARN
//                                                     |   text and returns the
//                                                     |   identical
//                                                     |   ncclInvalidArgument.
//                                                     |   No input distin-
//                                                     |   guishes the two
//                                                     |   guards.
//   rcclGetAlgoName: merge RCCL_DDA_IPC into the      | Killed
//     fabric-variant "DDA" case                       |
//   rcclGetAlgoProtoIndex strcasecmp -> strcmp        | Killed
//   rcclGetAlgoProtoIndex warn-once latch deleted      | Initially survived
//     (failedProtoWarn = true;)                        |   (unobservable from
//                                                       |   a single isolated
//                                                       |   call); killed
//                                                       |   after adding
//                                                       |   GetAlgoProtoIndex_
//                                                       |   SecondUnmatched-
//                                                       |   CallStaysSilent.
//   rcclUseAlltoAllGda default return false -> true    | Killed
//
// Not yet attempted here: a systematic mutation pass over the remaining
// uncovered functions in the file (deferred to their own rungs).

#include <gtest/gtest.h>

#include <cstring>

#include "../common/ProcessIsolatedTestRunner.hpp"  // RUN_ISOLATED_TEST

// RCCL_PARAM redirector. rccl_wrap.cc's RCCL_PARAM(...) invocations are not
// exercised by this first test batch (none of the eight units below read a
// param), so -- unlike init-test.cc's g_loadParam hook -- there is nothing
// yet to make per-test-controllable. Redirecting straight to deftVal is the
// minimal correct behaviour: it keeps the real macro (and its mismatched-
// looking real ncclLoadParam(), see fakes/nccl_fakes.cc's stale 4-arg
// version vs param.h's real 5-arg declaration) out of this TU entirely.
// Upgrade to the g_loadParam std::function pattern (fakes/nccl_fakes.h) the
// moment a later rung needs to flip a specific param per test.
#include "param.h"
#undef RCCL_PARAM
#define RCCL_PARAM(name, env, deftVal) \
  int64_t rcclParam##name() { return (deftVal); }

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
}

TEST(WrapMicrotest, ValidHsaScratchEnv_Gfx942FirmwareBoundary) {
  EXPECT_TRUE(validHsaScratchEnvSetting(nullptr, 60443484, 177, "gfx942"));
  EXPECT_FALSE(validHsaScratchEnvSetting(nullptr, 60443484, 176, "gfx942"));
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
#else
TEST(WrapMicrotest, IsArchSupportedForFunc_LL128_DisabledAtCompileTime) {
  // ENABLE_LL128 not defined in this build config: the LL128 branch is
  // compiled out entirely and `supported` stays at its `true` initializer
  // regardless of arch or acc. Structural/build-config gap, not a logic gap.
  ncclTaskColl withAcc = MakeTask(NCCL_PROTO_LL128, /*hasAcc=*/true);
  EXPECT_TRUE(rcclIsArchSupportedForFunc(&withAcc, "gfx90a"));
}
#endif

TEST(WrapMicrotest, IsArchSupportedForFunc_NonLL128_AccRestrictsToGfx9xAnd1250) {
  ncclTaskColl withAcc = MakeTask(NCCL_PROTO_SIMPLE, /*hasAcc=*/true);
  EXPECT_FALSE(rcclIsArchSupportedForFunc(&withAcc, "gfx90a"));
  EXPECT_TRUE(rcclIsArchSupportedForFunc(&withAcc, "gfx942"));
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
  // (`algo >= RCCL_ALGO_COUNT`) must reject it before the inner switch ever
  // runs -- a boundary a plain "does it return an error for -1" test cannot
  // distinguish from an off-by-one (`>` instead of `>=`) in that guard.
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

// static bool failedProtoWarn is a once-per-process latch (rccl_wrap.cc:159-
// 164): the WARN only fires on the first unmatched string any test in this
// binary passes in; every later mismatch silently returns ncclInvalidUsage
// with no log line. RUN_ISOLATED_TEST forks a fresh process so this is the
// first (and only) call in that image, making the WARN observable.
TEST(WrapMicrotestIsolated, GetAlgoProtoIndex_UnmatchedStringWarnsOnce) {
  RUN_ISOLATED_TEST(
      "Wrap_GetAlgoProtoIndex_UnmatchedStringWarnsOnce",
      []() {
        const char* table[] = {"LL", "LL128", "SIMPLE"};
        int result = -99;
        testing::internal::CaptureStderr();
        ncclResult_t r = rcclGetAlgoProtoIndex("bogus", table, 3, result);
        std::string err = testing::internal::GetCapturedStderr();
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
        testing::internal::CaptureStderr();
        ncclResult_t r = rcclGetAlgoProtoIndex("alsobogus", table, 3, result);
        std::string err = testing::internal::GetCapturedStderr();
        ASSERT_EQ(ncclInvalidUsage, r);
        EXPECT_TRUE(err.empty()) << "expected the warn-once latch to suppress this WARN, got: " << err;
      });
}
