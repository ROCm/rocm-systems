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
// fakes/wrap_fakes.cc.
//
// Covers every function in rccl_wrap.cc reachable in this binary's build
// configuration, through plain comm/topology setup (MakeCommWithArch below)
// plus a wide set of controllable seams: RCCL_PARAM/NCCL_PARAM via
// g_loadParam (fakes/param_redirect.h), the settable getenv/ncclGetEnv fake
// (fakes/env_fakes.cc), and this unit's own hooks for the CE/DDA/
// symmetric-kernel dependency chain (fakes/wrap_fakes.cc).
//
// Those hooks are what make the harder functions reachable at all:
// isSymmetricKernelRequested, the graph-capture probe, window/registration
// state, the CE availability trio, the DDA path selector, all 24
// per-collective DDA eligibility/blocks functions, the symk deep path, and
// the getAlgoInfo/rcclKernelPackedChannels plain-kernel fallback were
// abort() floors before. Making them controllable is what unblocks the
// top-level dispatchers -- the three rcclSelectXxx (AllReduce/AllGather/
// ReduceScatter), rcclHierarchicalAlgoInfo, rcclGetAlgoInfo,
// rcclGetCollImplInfo, and rcclSymkQuery/rcclSymKGetInfo's deep post-guard
// path.
//
// One deliberate, permanent exclusion: the WarpSpeed helpers under
// #ifdef ENABLE_WARP_SPEED are not compiled into this binary at all (the
// flag is off) -- no seam makes uncompiled code reachable, so this is out
// of scope for this binary's build configuration, not a gap.
//
// Mutation-tested directly against this file; residuals and equivalent
// mutants found along the way are documented at their own test below
// rather than here.

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

#include "../common/LogCapture.hpp"                 // RcclUnitTesting::CaptureLog
#include "../common/ProcessIsolatedTestRunner.hpp"  // RUN_ISOLATED_TEST
#include "ScopedHook.h"                              // RAII install/restore for g_loadParam et al.
#include "fakes/env_fakes.h"                         // SetMicroEnv/SetMicroEnvAbsent/ClearMicroEnv
#include "fakes/wrap_fakes.h"                        // rccl_wrap.cc's dependency seams
#include "graph/topo.h"                              // ncclTopoSystem/ncclTopoNode (MakeCommWithArch)

// RCCL_PARAM/NCCL_PARAM redirector: routes every generated rcclParamXxx()/
// ncclParamXxx() through g_loadParam on each call (no caching), so a test can
// flip one param's value between cases. fakes/param_redirect.h is the shared
// version of exactly this mechanism (init-test.cc / p2p-test.cc / enqueue-test.cc
// all use it), including the "RCCL_" + env convention the RCCL_PARAM arm needs.
// g_loadParam itself lives in fakes/nccl_fakes.h/.cc, already part of this
// binary -- declaring a second copy here would be a duplicate-symbol error.
#include <functional>

#include "fakes/nccl_fakes.h"    // extern g_loadParam
#include "fakes/param_redirect.h"  // NCCL_PARAM/RCCL_PARAM -> g_loadParam

// WRAP_CC_PATH is defined by test/host/CMakeLists.txt as the hipified copy of
// src/rccl_wrap.cc, e.g. ${PROJECT_BINARY_DIR}/hipify/src/rccl_wrap.cc.
#include WRAP_CC_PATH

namespace {

// Zero-initialized heap ncclComm, mirroring MockComm.hpp's
// new-then-memset idiom (test/common/MockComm.hpp) without pulling in that
// header's <rccl/rccl.h> include, which targets the installed-package layout
// rather than this target's hipify-tree headers. Caller deletes.
//
// The void* cast is the compiler-recommended silencer for
// -Wnontrivial-memcall: ncclComm is not trivially copyable, so clang warns on
// a raw memset of it. Zeroing is still what this wants -- every field a test
// reads is a scalar or POD, and the tests set up exactly the ones they need.
ncclComm* MakeZeroedComm() {
  ncclComm* comm = new ncclComm();
  std::memset(static_cast<void*>(comm), 0, sizeof(ncclComm));
  return comm;
}

// Comm with a real one-GPU topology wired up, arch string settable -- mirrors
// test/common/MockComm.hpp's CreateMockComm, not included here for the same
// reason as above. archName is pointed at the same buffer as the topology
// node's gcn
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

// The single teardown for BOTH factories above, deliberately: MakeZeroedComm
// leaves comm->topo null, so `delete comm->topo` is a no-op there. Having one
// teardown that is correct for both removes the foot-gun of a plain
// `delete comm` on a MakeCommWithArch result silently leaking the
// ncclTopoSystem -- a leak nothing in this target's default configuration
// reports.
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

// The three tests around this one probe rows 0 (Broadcast), 4 (AllReduce) and
// 5 (SendRecv) only, which is NOT enough to back the "a swapped row cannot
// escape" claim above: swapping the Reduce and AllGather rows
// (rccl_wrap.cc:96-97) leaves every one of those assertions passing, because
// neither row is ever queried. These two pairs close that hole -- Reduce and
// AllGather carry distinct cutoffs (8192 vs 98304), so a swap flips both.
// Rows 3 (ReduceScatter) and 6/7 (Send/Recv) share a cutoff with a row already
// pinned here, so a swap among them is genuinely unobservable through this
// function's return value.
TEST(WrapMicrotest, GetProtoForGfx120x_ReduceAndAllGatherCutoffBoundaries) {
  EXPECT_EQ(NCCL_PROTO_LL, rcclGetProtoForGfx120x(ncclFuncReduce, 8192));
  EXPECT_EQ(NCCL_PROTO_SIMPLE, rcclGetProtoForGfx120x(ncclFuncReduce, 8193));
  EXPECT_EQ(NCCL_PROTO_LL, rcclGetProtoForGfx120x(ncclFuncAllGather, 98304));
  EXPECT_EQ(NCCL_PROTO_SIMPLE, rcclGetProtoForGfx120x(ncclFuncAllGather, 98305));
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
// arguments. rccl_wrap.cc:1737-1750.
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
  // ...and 0-vs-60443484 alone only proves *some* runtime check exists, not
  // where its threshold sits: any mutant lowering the constant into
  // (0, 60443484) survives both. One below the real threshold pins it, the
  // same way 23/24 pins the firmware side.
  EXPECT_FALSE(validHsaScratchEnvSetting(nullptr, 60443483, 24, "gfx950"));
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
// rccl_wrap.cc:1753-1773. Should match get_arch_guard() in generate.py per
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
// (< NCCL_NUM_ALGORITHMS) values; wrap_fakes.cc does NOT stub that function
// (it's genuinely faked with a faithful copy of collectives.cc's switch, to
// avoid pulling collectives.cc's DDA/sym/nvtx dependency chain into this
// lean binary -- see fakes/wrap_fakes.cc).
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

// NCCL_ALGO_TREE is 0, the first value the `algo < 0` lower bound must let
// THROUGH; every other test here uses RING (1) or an addon value (>= 7), so
// without it nothing sits on that edge and a `<` -> `<=` mutant survives (a
// mutation sweep confirmed exactly that). Asserted alongside RING because
// both are native values proving the same delegation to ncclAlgoToString.
TEST(WrapMicrotest, GetAlgoName_NativeAlgoDelegatesToNcclAlgoToString) {
  const char* name = nullptr;
  ASSERT_EQ(ncclSuccess, rcclGetAlgoName(NCCL_ALGO_RING, &name));
  EXPECT_STREQ("RING", name);

  ASSERT_EQ(ncclSuccess, rcclGetAlgoName(NCCL_ALGO_TREE, &name)) << "algo 0 must not be rejected as out of range";
  EXPECT_STREQ("TREE", name);
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
        const std::string err =
          RcclUnitTesting::CaptureLog([&]() { r = rcclGetAlgoProtoIndex("bogus", table, 3, result); });
        ASSERT_EQ(ncclInvalidUsage, r);
        EXPECT_EQ(-99, result);
        EXPECT_NE(std::string::npos, err.find("Invalid algo or protocol string passed bogus"));
      });
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

// Closes the cross-purpose gap flagged above: failedProtoWarn is shared
// across rcclOverrideProtocol's and rcclOverrideAlgorithm's independent call
// sites, not scoped per-purpose. Fails a protocol parse first (through the
// real rcclOverrideProtocol entry point), then an algorithm parse (through
// the real rcclOverrideAlgorithm entry point) in the same isolated process --
// both still correctly return ncclInvalidUsage, but the second, genuinely
// different problem's diagnostic is silently swallowed by the first's latch.
TEST(WrapMicrotestIsolated, GetAlgoProtoIndex_CrossPurposeLatchSuppressesAlgoWarnAfterProtoFailure) {
  RUN_ISOLATED_TEST(
      "Wrap_GetAlgoProtoIndex_CrossPurposeLatchSuppressesAlgoWarnAfterProtoFailure",
      []() {
        SetMicroEnv("RCCL_OVERRIDE_PROTO", "bogusproto");
        const char* protoStr[] = {"LL", "LL128", "SIMPLE"};
        float protoTable[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {};
        ncclTaskColl protoInfo{};
        const std::string protoLog = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_EQ(ncclInvalidUsage, rcclOverrideProtocol(protoStr, protoTable, &protoInfo)); });
        EXPECT_NE(std::string::npos, protoLog.find("Invalid algo or protocol string"));  // primes the shared latch

        SetMicroEnv("RCCL_OVERRIDE_ALGO", "bogusalgo");
        const char* algoStr[] = {"TREE", "RING", "COLLNET_DIRECT", "COLLNET_CHAIN", "NVLS", "NVLS_TREE", "PAT"};
        float algoTable[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {};
        ncclTaskColl algoInfo{};
        const std::string algoLog = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_EQ(ncclInvalidUsage, rcclOverrideAlgorithm(algoStr, algoTable, &algoInfo)); });
        EXPECT_TRUE(algoLog.empty())
            << "a genuinely different bad RCCL_OVERRIDE_ALGO string should still warn -- it doesn't, because "
               "failedProtoWarn is shared with rcclOverrideProtocol's earlier failure. Got: "
            << algoLog;
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
  DeleteCommWithArch(comm);
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
  DeleteCommWithArch(comm);
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
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, CeAllReduceGraphLatchTick_ClearsWhenNotCapturingAndNoRefs) {
  ncclComm* comm = MakeZeroedComm();
  comm->ceColl.graphModeSeen = true;
  comm->localPersistentRefs = 0;
  rcclCeAllReduceGraphLatchTick(comm, /*ceCapturing=*/false);
  EXPECT_FALSE(comm->ceColl.graphModeSeen);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, CeAllReduceGraphLatchTick_StaysLatchedWhileRefsLive) {
  ncclComm* comm = MakeZeroedComm();
  comm->ceColl.graphModeSeen = true;
  comm->localPersistentRefs = 1;
  rcclCeAllReduceGraphLatchTick(comm, /*ceCapturing=*/false);
  EXPECT_TRUE(comm->ceColl.graphModeSeen);
  DeleteCommWithArch(comm);
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
  DeleteCommWithArch(comm);
}

// The five tests above pin every behavioural path through this function, but
// all of them run at the default (suppressed) debug level, so neither INFO
// call site's logging arm is ever taken -- llvm-cov reports the function at
// 58.82% branch coverage for exactly that reason. This raises the level and
// asserts the message CONTENT of both transitions, which nothing else does:
// a mutant swapping the "set" and "cleared" strings, or reporting the wrong
// rank, passes all five tests above and fails here.
TEST(WrapMicrotestIsolated, CeAllReduceGraphLatchTick_BothTransitionsLogTheirOwnMessage) {
  RUN_ISOLATED_TEST(
      "Wrap_CeAllReduceGraphLatchTick_BothTransitionsLogTheirOwnMessage",
      []() {
        RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_ALL);
        ncclComm* comm = MakeZeroedComm();
        comm->rank = 3;

        const std::string setLog =
          RcclUnitTesting::CaptureLog([&]() { rcclCeAllReduceGraphLatchTick(comm, /*ceCapturing=*/true); });
        EXPECT_TRUE(comm->ceColl.graphModeSeen);
        EXPECT_NE(std::string::npos, setLog.find("graph latch set (rank 3)"));

        // Latch already set, still capturing: the `!graphModeSeen` guard means
        // this must NOT log a second time.
        const std::string repeatLog =
          RcclUnitTesting::CaptureLog([&]() { rcclCeAllReduceGraphLatchTick(comm, /*ceCapturing=*/true); });
        EXPECT_EQ(std::string::npos, repeatLog.find("graph latch set"));

        comm->localPersistentRefs = 0;
        const std::string clearLog =
          RcclUnitTesting::CaptureLog([&]() { rcclCeAllReduceGraphLatchTick(comm, /*ceCapturing=*/false); });
        EXPECT_FALSE(comm->ceColl.graphModeSeen);
        EXPECT_NE(std::string::npos, clearLog.find("graph latch cleared (rank 3)"));

        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotest, CeAllReduceAllowed_TrueWhenLatchClear) {
  ncclComm* comm = MakeZeroedComm();
  comm->ceColl.graphModeSeen = false;
  EXPECT_TRUE(rcclCeAllReduceAllowed(comm));
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, CeAllReduceAllowed_FalseWhenLatchSet) {
  ncclComm* comm = MakeZeroedComm();
  comm->ceColl.graphModeSeen = true;
  EXPECT_FALSE(rcclCeAllReduceAllowed(comm));
  DeleteCommWithArch(comm);
}

// ===========================================================================
// rcclSetPxn / rcclSetP2pNetChunkSize -- rccl_wrap.cc:1370-1415. Both read a
// real environment variable via plain getenv() on the "not yet cached" path.
// The tests here cover the already-cached fast return (comm->pxnDisable /
// comm->p2pNetChunkSize already != RCCL_VALUE_UNSET), which never touches
// getenv at all. The env-reading arch/rank computation needs a real
// env-controlling seam rather than whatever happens to be set in the
// environment, so it is covered separately further down this file.
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
// rcclGetMaxNthreads -- rccl_wrap.cc:1607-1614.
// ===========================================================================

// RCCL_GFX950_MAX_NTHREADS, RCCL_DEFAULT_MAX_NTHREADS, and RCCL_LL_MAX_NTHREADS
// are all 256 today, so a value assertion alone cannot distinguish "took the
// gfx950 arm" from "took the else arm and the constants just happen to
// match". That is a TRUE equivalent mutant while the constants are equal --
// no input to this function can tell the arms apart, so the two tests below
// deliberately do not claim to. Both calls are still made, so llvm-cov shows
// both arms as reached; NCCL_PROTO_LL's assignment is arch-independent and
// IS a real oracle either way.
//
// The divergence tripwire is a RUNTIME check, not a static_assert: this is a
// shared binary (p2p/rma/group/devcomm/enqueue all link it), and a
// compile-time assert here would break every one of those builds over a
// constant none of them read. One failing test names the problem without
// taking the rest of the suite down with it.
TEST(WrapMicrotest, GetMaxNthreads_ConstantsStillEquivalent) {
  EXPECT_EQ(RCCL_GFX950_MAX_NTHREADS, RCCL_DEFAULT_MAX_NTHREADS)
      << "constants diverged -- GetMaxNthreads_Gfx950Arch/NonGfx950Arch below can now "
         "assert values that actually distinguish the gfx950 vs. default arm; give them "
         "real oracles and delete this test";
  EXPECT_EQ(RCCL_DEFAULT_MAX_NTHREADS, RCCL_LL_MAX_NTHREADS)
      << "constants diverged -- see above";
}

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
// rcclSetDefaultBuffSizes -- rccl_wrap.cc:1646-1654. Isolated: its own
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
        // for LL128/SIMPLE and RCCL_LL_MAX_NTHREADS for LL.
        //
        // kGfx942ElemsPerThread is the LITERAL 28, not
        // rcclLL128ElemsPerThreadFromArch("gfx942"): deriving it from the same
        // helper production uses makes the assertion compare production
        // against itself, so a change to rcclLL128DataElemsFromArch moves both
        // sides equally and is never caught. 28 = 4 lines/thread (gfx9xx) *
        // 7 data elems/line (64B line / 8B elem, minus one for the tag) --
        // see archinfo.h:45-57. If that layout legitimately changes, this
        // number must be updated by hand, which is the point.
        constexpr int kGfx942ElemsPerThread = 28;
        EXPECT_EQ(NCCL_LL_LINES_PER_THREAD * RCCL_LL_MAX_NTHREADS * NCCL_STEPS * (int)sizeof(union ncclLLFifoLine),
                  defaultBuffSizes[NCCL_PROTO_LL]);
        EXPECT_EQ(kGfx942ElemsPerThread * RCCL_DEFAULT_MAX_NTHREADS * NCCL_STEPS * (int)sizeof(uint64_t),
                  defaultBuffSizes[NCCL_PROTO_LL128]);
        EXPECT_EQ(1 << 22, defaultBuffSizes[NCCL_PROTO_SIMPLE]);
        DeleteCommWithArch(comm);
      });
}

// The gfx942 case above cannot tell an arch-dependent LL128 size from a
// hardcoded one: every gfx9xx arch yields the same 28 elems/thread, so a
// mutant that ignores comm->archName entirely still passes it. gfx1250 is
// the only arch in this repo with a different LL128 line layout -- 128B
// lines give 16 elems, 15 of them data, at 8 lines/thread = 120 -- so it is
// what actually proves the value is computed from the arch.
//
// 120 is written as a literal for the same reason 28 is above: deriving it
// from rcclLL128ElemsPerThreadFromArch would compare production against
// itself. See archinfo.h:42-57; if that layout changes, update this by hand.
//
// Isolated for the same reason as the test above -- maxNthreads[] is a
// function-local static, so this must be the only call in its process.
TEST(WrapMicrotestIsolated, SetDefaultBuffSizes_Gfx1250UsesWiderLL128Line) {
  RUN_ISOLATED_TEST(
      "Wrap_SetDefaultBuffSizes_Gfx1250UsesWiderLL128Line",
      []() {
        // First call on a DIFFERENT arch, so the `maxNthreads[SIMPLE] == 0`
        // guard is entered here and taken the other way on the gfx1250 call
        // below -- the only way to reach that guard's false arm, since every
        // other test in this file calls the function once per process.
        ncclComm* warm = MakeCommWithArch("gfx942");
        int warmSizes[NCCL_NUM_PROTOCOLS] = {0};
        rcclSetDefaultBuffSizes(warm, warmSizes);
        DeleteCommWithArch(warm);

        ncclComm* comm = MakeCommWithArch("gfx1250");
        int defaultBuffSizes[NCCL_NUM_PROTOCOLS] = {0};
        rcclSetDefaultBuffSizes(comm, defaultBuffSizes);
        constexpr int kGfx1250ElemsPerThread = 120;  // 8 lines/thread * 15 data elems
        EXPECT_EQ(kGfx1250ElemsPerThread * RCCL_DEFAULT_MAX_NTHREADS * NCCL_STEPS * (int)sizeof(uint64_t),
                  defaultBuffSizes[NCCL_PROTO_LL128]);
        // LL and SIMPLE are arch-independent here, so they must match the
        // gfx942 case exactly -- pinning that only the LL128 term moved.
        EXPECT_EQ(NCCL_LL_LINES_PER_THREAD * RCCL_LL_MAX_NTHREADS * NCCL_STEPS * (int)sizeof(union ncclLLFifoLine),
                  defaultBuffSizes[NCCL_PROTO_LL]);
        EXPECT_EQ(1 << 22, defaultBuffSizes[NCCL_PROTO_SIMPLE]);

        // The cache holds maxNthreads[], NOT the arch: LL/SIMPLE come from the
        // cached (gfx942) thread counts and are unchanged, while LL128 is
        // recomputed from THIS call's comm->archName and must differ. A mutant
        // that cached the whole result, or that recomputed maxNthreads every
        // call, breaks one of these two halves.
        EXPECT_EQ(warmSizes[NCCL_PROTO_LL], defaultBuffSizes[NCCL_PROTO_LL]);
        EXPECT_NE(warmSizes[NCCL_PROTO_LL128], defaultBuffSizes[NCCL_PROTO_LL128]);
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclFuncMaxSendRecvCount -- rccl_wrap.cc:1656-1660. Thin wrapper delegating
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
// ParamDefaults_MatchProductionSource -- the same drift-guard idea as the
// FakeTableDrift_* tests above, for values no constant can express. ncclParamMinNchannels/MaxNchannels,
// rcclParamForceCe, and ncclParamLaunchOrderImplicit hardcode the real
// NCCL_PARAM/RCCL_PARAM default they copy (see wrap_fakes.cc), but that
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
// Returns nullopt when the path isn't readable. CONNECT_CC_PATH /
// ENQUEUE_CC_PATH are absolute build-tree hipify paths baked in at compile
// time, and this binary is rocm_installed + registered into the installed
// CTest file, which is written to work from the installed location -- on a
// relocated run the hipify tree is gone. That's an unavailable oracle, not a
// drift, so the caller SKIPs rather than failing.
std::optional<std::string> ReadFileIfPresent(const char* path) {
  std::ifstream f(path);
  if (!f) return std::nullopt;
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
}  // namespace

TEST(WrapMicrotest, ParamDefaults_MatchProductionSource) {
  const auto connectOpt = ReadFileIfPresent(CONNECT_CC_PATH);
  const auto enqueueOpt = ReadFileIfPresent(ENQUEUE_CC_PATH);
  if (!connectOpt || !enqueueOpt) {
    GTEST_SKIP() << "hipify sources not readable (relocated/installed run): "
                 << CONNECT_CC_PATH << " / " << ENQUEUE_CC_PATH;
  }
  const std::string& connectCc = *connectOpt;
  const std::string& enqueueCc = *enqueueOpt;

  EXPECT_NE(std::string::npos, connectCc.find(R"(NCCL_PARAM(MinNchannels, "MIN_NCHANNELS", -2))"))
      << "graph/connect.cc's NCCL_PARAM(MinNchannels...) changed -- update ncclParamMinNchannels() in wrap_fakes.cc";
  EXPECT_NE(std::string::npos, connectCc.find(R"(NCCL_PARAM(MaxNchannels, "MAX_NCHANNELS", -2))"))
      << "graph/connect.cc's NCCL_PARAM(MaxNchannels...) changed -- update ncclParamMaxNchannels() in wrap_fakes.cc";
  EXPECT_NE(std::string::npos, enqueueCc.find(R"(RCCL_PARAM(ForceCe, "FORCE_CE", 1))"))
      << "enqueue.cc's RCCL_PARAM(ForceCe...) changed -- update rcclParamForceCe() in wrap_fakes.cc";
  EXPECT_NE(std::string::npos, enqueueCc.find(R"(NCCL_PARAM(LaunchOrderImplicit, "LAUNCH_ORDER_IMPLICIT", 0))"))
      << "enqueue.cc's NCCL_PARAM(LaunchOrderImplicit...) changed -- update "
         "ncclParamLaunchOrderImplicit() in wrap_fakes.cc";
}

// ===========================================================================
// Drift watchdogs for the hand-copied tables in fakes/wrap_fakes.cc.
//
// Those tables (ncclAlgoToString / ncclProtoToString / ncclFuncToString /
// ncclDatatypeToString, and ncclDevFuncUnrollGenerated[]) copy production
// switch statements by hand, because linking the real collectives.cc would
// drag in the DDA/symmetric-kernel/nvtx chain this binary avoids. If RCCL
// adds an algorithm, protocol, collective or datatype, a hand copy silently
// returns "Unknown" (or, for the unroll array, silently zero-fills a new
// slot to false) instead of failing.
//
// These are runtime tests rather than static_asserts on purpose. This binary
// is shared -- p2p, rma, group and devcomm tests all link the same
// executable -- so a compile-time assert would break their builds over
// constants none of them read. A failing test is just as loud, and CI runs
// it, but the blast radius stops at the suite that actually owns the tables.
// ===========================================================================

// One test for all four string tables: they share a failure mode (a new enum
// value falls through to "Unknown") and a fix (update the matching switch in
// fakes/wrap_fakes.cc), so splitting them buys nothing a reviewer can act on.
TEST(WrapMicrotest, FakeTableDrift_StringTableCountsMatchProduction) {
  EXPECT_EQ(7, NCCL_NUM_ALGORITHMS)
      << "NCCL_NUM_ALGORITHMS changed -- update ncclAlgoToString's switch in "
         "fakes/wrap_fakes.cc to match collectives.cc:115, then update this count";
  EXPECT_EQ(3, NCCL_NUM_PROTOCOLS)
      << "NCCL_NUM_PROTOCOLS changed -- update ncclProtoToString's switch in "
         "fakes/wrap_fakes.cc to match collectives.cc:136, then update this count";
  EXPECT_EQ(19, ncclNumFuncs)
      << "ncclFunc_t changed -- update ncclFuncToString's switch in "
         "fakes/wrap_fakes.cc to match collectives.cc:32, then update this count";
  EXPECT_EQ(12, ncclNumTypes)
      << "ncclDataType_t changed -- update ncclDatatypeToString's switch in "
         "fakes/wrap_fakes.cc to match collectives.cc:86, then update this count";
}

TEST(WrapMicrotest, FakeTableDrift_UnrollCountMatchesProduction) {
  EXPECT_EQ(6, NCCL_NUM_UNROLLS)
      << "NCCL_NUM_UNROLLS changed -- add/remove a `true` entry in "
         "ncclDevFuncUnrollGenerated[] (fakes/wrap_fakes.cc) to match, then update this count";
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
// (gfx950 AllGather, plus its own boundary; the gfx950/gfx942 ReduceScatter
// arms right below it are the same shape with different constants and are
// covered by their own tests here), the gfx120x delegation +
// NCCL_P2P_DISABLE override (exercises the ncclGetEnv seam), and the
// nNodes>=2 minMaxLLRange-driven arm including its warn-once
// undefined-tuning fallback.
//
// ENABLE_LL128's nested arm follows the same convention as
// rcclIsArchSupportedForFunc above: whether it compiles in depends on the
// build (test/CMakeLists.txt appends ENABLE_LL128 when LL128_ENABLED), so it
// is not asserted here rather than assumed off.
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

// The test above uses nBytes=1024, three orders of magnitude below the
// threshold, so it cannot tell `sizePerRank <= 88448` from `< 88448` -- a
// mutation sweep confirmed the `<=` -> `<` mutant survives it. nRanks=1 makes
// sizePerRank == nBytes, so these two pin the exact cutoff: 88448 still LL,
// 88449 falls through to the SIMPLE it was seeded with.
TEST(WrapMicrotestIsolated, UpdateCollectiveProtocol_Gfx950AllGatherLLCutoffBoundary) {
  RUN_ISOLATED_TEST(
      "Wrap_UpdateCollectiveProtocol_Gfx950AllGatherLLCutoffBoundary",
      []() {
        SetMicroEnvAbsent("NCCL_PROTO");
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 1;
        comm->nRanks = 1;
        ncclTaskColl atCutoff{};
        atCutoff.func = ncclFuncAllGather;
        atCutoff.protocol = NCCL_PROTO_SIMPLE;
        rcclUpdateCollectiveProtocol(comm, /*nBytes=*/88448, &atCutoff);
        EXPECT_EQ(NCCL_PROTO_LL, atCutoff.protocol);

        ncclTaskColl pastCutoff{};
        pastCutoff.func = ncclFuncAllGather;
        pastCutoff.protocol = NCCL_PROTO_SIMPLE;
        rcclUpdateCollectiveProtocol(comm, /*nBytes=*/88449, &pastCutoff);
        EXPECT_EQ(NCCL_PROTO_SIMPLE, pastCutoff.protocol);
        DeleteCommWithArch(comm);
      });
}

// Closes two entirely-unexecuted branches (llvm-cov: 0 hits in either
// direction on all three sub-conditions) -- every prior gfx950/gfx942 test
// used AllGather, never ReduceScatter, so these elseif arms (gated on
// info->func == ncclFuncReduceScatter specifically) had never even been
// reached, let alone had their own sizePerRank threshold checked.
TEST(WrapMicrotestIsolated, UpdateCollectiveProtocol_Gfx950ReduceScatterSmallSizeUsesLL) {
  RUN_ISOLATED_TEST(
      "Wrap_UpdateCollectiveProtocol_Gfx950ReduceScatterSmallSizeUsesLL",
      []() {
        SetMicroEnvAbsent("NCCL_PROTO");
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 1;
        comm->nRanks = 1; // sizePerRank == nBytes for ReduceScatter
        ncclTaskColl info{};
        info.func = ncclFuncReduceScatter;
        info.protocol = NCCL_PROTO_SIMPLE;
        rcclUpdateCollectiveProtocol(comm, /*nBytes=*/1048576, &info); // exactly at the 1MiB threshold
        EXPECT_EQ(NCCL_PROTO_LL, info.protocol);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UpdateCollectiveProtocol_Gfx942ReduceScatterSmallSizeUsesLL) {
  RUN_ISOLATED_TEST(
      "Wrap_UpdateCollectiveProtocol_Gfx942ReduceScatterSmallSizeUsesLL",
      []() {
        SetMicroEnvAbsent("NCCL_PROTO");
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 1;
        comm->nRanks = 1; // sizePerRank == nBytes for ReduceScatter
        ncclTaskColl info{};
        info.func = ncclFuncReduceScatter;
        info.protocol = NCCL_PROTO_SIMPLE;
        rcclUpdateCollectiveProtocol(comm, /*nBytes=*/352128, &info); // exactly at the threshold
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

// Resolves the open question flagged above: a multi-node gfx120x comm still
// matches this branch's own condition (it doesn't require single-node), but
// the ONLY protocol-setting call inside it is nested in `if (nNodes == 1)`.
// With NCCL_P2P_DISABLE absent too, nothing inside this branch touches
// info->protocol at all -- and because the branch matched, the nNodes>=2
// tuning-model branch below it never runs either. This proves, rather than
// just suspects, that multi-node gfx120x collectives get zero protocol
// auto-tuning from this function: the sentinel value set below survives
// completely untouched.
TEST(WrapMicrotestIsolated, UpdateCollectiveProtocol_Gfx120xMultiNodeGetsNoAutoTuning) {
  RUN_ISOLATED_TEST(
      "Wrap_UpdateCollectiveProtocol_Gfx120xMultiNodeGetsNoAutoTuning",
      []() {
        SetMicroEnvAbsent("NCCL_PROTO");
        SetMicroEnvAbsent("NCCL_P2P_DISABLE");
        ncclComm* comm = MakeCommWithArch("gfx1200");
        comm->nNodes = 2;
        comm->nRanks = 2;
        ncclTaskColl info{};
        info.func = ncclFuncAllGather;
        info.protocol = NCCL_PROTO_LL128;  // sentinel: not what rcclGetProtoForGfx120x nor the nNodes>=2 arm would pick
        rcclUpdateCollectiveProtocol(comm, /*nBytes=*/1024, &info);
        EXPECT_EQ(NCCL_PROTO_LL128, info.protocol)
            << "sentinel untouched -- confirms zero auto-tuning for multi-node gfx120x";
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

// The inner `tunedThreshold != RCCL_LL_LIMITS_UNDEFINED` guard had only ever
// been proven on its true side: every other test here either sets a real
// tuned value or is stopped by the outer env/nNodes guard before reaching
// this line at all. Identical setup to the test above, but with
// minMaxLLRange left zero-initialized, so the tuning table has no entry for
// this coll/protocol and the caller's threadThreshold must survive untouched.
TEST(WrapMicrotestIsolated, UpdateThreadThreshold_UndefinedTunedValueLeavesThresholdUntouched) {
  RUN_ISOLATED_TEST(
      "Wrap_UpdateThreadThreshold_UndefinedTunedValueLeavesThresholdUntouched",
      []() {
        SetMicroEnvAbsent("NCCL_THREAD_THRESHOLDS");
        SetMicroEnvAbsent("NCCL_MAX_NCHANNELS");
        SetMicroEnvAbsent("NCCL_MIN_NCHANNELS");
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 2;
        comm->nRanks = 4;
        // minMaxLLRange left zero-initialized == RCCL_LL_LIMITS_UNDEFINED.
        ncclTaskColl info{};
        info.func = ncclFuncReduceScatter;
        info.protocol = NCCL_PROTO_LL;
        int threadThreshold = -1;
        rcclUpdateThreadThreshold(comm, /*nBytes=*/1024, &info, threadThreshold);
        EXPECT_EQ(-1, threadThreshold); // untouched -- no tuned entry to scale
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

// The override test above only ever exercises the first of the three
// cascaded getenv probes (NCCL_THREAD_THRESHOLDS present short-circuits the
// other two). These two isolate the second and third probes as the one that
// actually finds a value, so deleting either line still gets caught.
TEST(WrapMicrotestIsolated, UpdateThreadThreshold_SecondEnvProbeAloneOverrides) {
  RUN_ISOLATED_TEST(
      "Wrap_UpdateThreadThreshold_SecondEnvProbeAloneOverrides",
      []() {
        SetMicroEnvAbsent("NCCL_THREAD_THRESHOLDS");
        SetMicroEnv("NCCL_MAX_NCHANNELS", "anything");
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

TEST(WrapMicrotestIsolated, UpdateThreadThreshold_ThirdEnvProbeAloneOverrides) {
  RUN_ISOLATED_TEST(
      "Wrap_UpdateThreadThreshold_ThirdEnvProbeAloneOverrides",
      []() {
        SetMicroEnvAbsent("NCCL_THREAD_THRESHOLDS");
        SetMicroEnvAbsent("NCCL_MAX_NCHANNELS");
        SetMicroEnv("NCCL_MIN_NCHANNELS", "anything");
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
// rcclSetPipelining -- rccl_wrap.cc:358-387. Gated on RCCL_PARAM
// disableReduceCopyPipelining and PipelineAllDTypes, both already routed
// through g_loadParam by this file's own RCCL_PARAM redirect above -- no new
// seam needed. No caching, no isolation needed.
// ===========================================================================

TEST(WrapMicrotest, SetPipelining_ParamDisabledLeavesPipelineOff) {
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deft) {
    return std::strcmp(env, "RCCL_DISABLE_REDUCE_COPY_PIPELINING") == 0 ? int64_t(1) : deft;
  });
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 1;
  ncclTaskColl info{};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclBfloat16;
  rcclSetPipelining(comm, /*nBytes=*/1024, &info);
  EXPECT_EQ(0, info.pipeline); // would be 1 without the disable check short-circuiting first
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetPipelining_Gfx950ArchLeavesPipelineOffRegardlessOfParam) {
  ncclComm* comm = MakeCommWithArch("gfx950");
  comm->nNodes = 1;
  ncclTaskColl info{};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclBfloat16;
  rcclSetPipelining(comm, /*nBytes=*/1024, &info);
  EXPECT_EQ(0, info.pipeline);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetPipelining_Gfx942Bf16AllReduceSingleNodeSetsPipeline) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 1;
  ncclTaskColl info{};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclBfloat16;
  rcclSetPipelining(comm, /*nBytes=*/1024, &info);
  EXPECT_EQ(1, info.pipeline);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetPipelining_Gfx942Bf16AllReduceMultiNodeAtLimitSetsPipeline) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 2; // log2i(2) == 1, so the bf16 limit equation gives exactly 512MB
  ncclTaskColl info{};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclBfloat16;
  rcclSetPipelining(comm, /*nBytes=*/(1ULL << 29), &info);
  EXPECT_EQ(1, info.pipeline);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetPipelining_Gfx942Bf16AllReduceMultiNodeAboveLimitLeavesPipelineOff) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 2; // same 512MB limit as above, one byte past it
  ncclTaskColl info{};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclBfloat16;
  rcclSetPipelining(comm, /*nBytes=*/(1ULL << 29) + 1, &info);
  EXPECT_EQ(0, info.pipeline);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetPipelining_Gfx942Bf16ReduceScatterSetsPipelineRegardlessOfSize) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 4; // arbitrary >1: no size gate applies to this func, unlike AllReduce
  ncclTaskColl info{};
  info.func = ncclFuncReduceScatter;
  info.datatype = ncclBfloat16;
  rcclSetPipelining(comm, /*nBytes=*/(1ULL << 40), &info); // deliberately huge
  EXPECT_EQ(1, info.pipeline);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetPipelining_Gfx942Bf16ReduceSetsPipeline) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 1;
  ncclTaskColl info{};
  info.func = ncclFuncReduce;
  info.datatype = ncclBfloat16;
  rcclSetPipelining(comm, /*nBytes=*/1024, &info);
  EXPECT_EQ(1, info.pipeline);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetPipelining_Gfx942Bf16OtherFuncDefaultArmLeavesPipelineOff) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 1;
  ncclTaskColl info{};
  info.func = ncclFuncAllGather; // not AllReduce/ReduceScatter/Reduce -- falls to switch's default: arm
  info.datatype = ncclBfloat16;
  rcclSetPipelining(comm, /*nBytes=*/1024, &info);
  EXPECT_EQ(0, info.pipeline);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetPipelining_Gfx942NonBf16WithoutOverrideLeavesPipelineOff) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 1;
  ncclTaskColl info{};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclFloat32; // not bf16; PipelineAllDTypes defaults to 0, so dtypeOK is false
  rcclSetPipelining(comm, /*nBytes=*/1024, &info);
  EXPECT_EQ(0, info.pipeline);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetPipelining_Gfx942NonBf16WithAllDTypesOverrideSetsPipeline) {
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deft) {
    return std::strcmp(env, "RCCL_PIPELINE_ALL_DATA_TYPES") == 0 ? int64_t(1) : deft;
  });
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 1;
  ncclTaskColl info{};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclFloat32; // not bf16, but the override makes dtypeOK true anyway
  rcclSetPipelining(comm, /*nBytes=*/1024, &info);
  EXPECT_EQ(1, info.pipeline);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetPipelining_NonGfx942ArchLeavesPipelineOff) {
  ncclComm* comm = MakeCommWithArch("gfx90a"); // not gfx942, not gfx950
  comm->nNodes = 1;
  ncclTaskColl info{};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclBfloat16;
  rcclSetPipelining(comm, /*nBytes=*/1024, &info);
  EXPECT_EQ(0, info.pipeline);
  DeleteCommWithArch(comm);
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

// rcclOverrideProtocol's own `static bool validInput` latch (rccl_wrap.cc:283-
// 284) had never seen its early-return arm fire (llvm-cov: [True: 0, False:
// 5]) -- every prior test made exactly one call per isolated process. A
// second call after a failed parse must short-circuit via the cached
// validInput=false rather than re-parsing (and re-warning).
TEST(WrapMicrotestIsolated, OverrideProtocol_SecondCallAfterFailureReturnsInvalidUsageWithoutReparsing) {
  RUN_ISOLATED_TEST(
      "Wrap_OverrideProtocol_SecondCallAfterFailureReturnsInvalidUsageWithoutReparsing",
      []() {
        SetMicroEnv("RCCL_OVERRIDE_PROTO", "bogus");
        const char* protoStr[] = {"LL", "LL128", "SIMPLE"};
        float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {};
        ncclTaskColl info{};
        info.protocol = NCCL_PROTO_SIMPLE;
        EXPECT_EQ(ncclInvalidUsage, rcclOverrideProtocol(protoStr, table, &info));
        std::string log = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_EQ(ncclInvalidUsage, rcclOverrideProtocol(protoStr, table, &info)); });
        EXPECT_EQ(std::string::npos, log.find("Invalid algo or protocol string"))
            << "second call must hit the cached validInput=false early-return, not re-parse and re-warn";
        EXPECT_EQ(NCCL_PROTO_SIMPLE, info.protocol);
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

// Complementary proof for rcclOverrideAlgorithm's own, separate `static bool
// validInput` latch (rccl_wrap.cc:310-311) -- same never-fired early-return
// arm as rcclOverrideProtocol's above, a distinct static local to this
// function.
TEST(WrapMicrotestIsolated, OverrideAlgorithm_SecondCallAfterFailureReturnsInvalidUsageWithoutReparsing) {
  RUN_ISOLATED_TEST(
      "Wrap_OverrideAlgorithm_SecondCallAfterFailureReturnsInvalidUsageWithoutReparsing",
      []() {
        SetMicroEnv("RCCL_OVERRIDE_ALGO", "bogus");
        const char* algoStr[] = {"TREE", "RING", "COLLNET_DIRECT", "COLLNET_CHAIN", "NVLS", "NVLS_TREE", "PAT"};
        float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {};
        ncclTaskColl info{};
        info.algorithm = NCCL_ALGO_TREE;
        EXPECT_EQ(ncclInvalidUsage, rcclOverrideAlgorithm(algoStr, table, &info));
        std::string log = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_EQ(ncclInvalidUsage, rcclOverrideAlgorithm(algoStr, table, &info)); });
        EXPECT_EQ(std::string::npos, log.find("Invalid algo or protocol string"))
            << "second call must hit the cached validInput=false early-return, not re-parse and re-warn";
        EXPECT_EQ(NCCL_ALGO_TREE, info.algorithm);
      });
}

// ===========================================================================
// rcclSetPxn / rcclSetP2pNetChunkSize -- remaining getenv-driven paths
// (rccl_wrap.cc:1370-1415; the cached fast-path is covered earlier in this
// file). Neither caches across calls itself (comm->pxnDisable/
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

// enableCustColl's `(archGfx942 || archGfx950) && (inputStr && !atoi(inputStr))`
// assignment is only OBSERVABLE when the function early-returns, because the
// non-early-return path overwrites it with `!pxnDisable` further down. The
// tests that do assert it all use a supported arch, so none can tell the outer
// `&&` from `||` -- a mutation sweep confirmed that mutant survives. An
// unsupported arch WITH the env var present is the one combination that
// separates them: the arch half is false while the env half is true, so `&&`
// gives false and `||` gives true.
TEST(WrapMicrotest, SetPxn_UnsupportedArchWithEnvLeavesCustCollFalse) {
  ncclComm* comm = MakeCommWithArch("gfx90a");
  comm->enableCustColl = true; // seeded true so a failure to assign is visible
  SetMicroEnv("NCCL_PXN_DISABLE", "0");
  int rcclPxnDisable = -100;
  rcclSetPxn(comm, rcclPxnDisable);
  EXPECT_EQ(RCCL_VALUE_INVALID, rcclPxnDisable);
  EXPECT_FALSE(comm->enableCustColl)
      << "unsupported arch must clear enableCustColl even when NCCL_PXN_DISABLE is set";
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

// Closes the gap flagged above: the only prior gfx950 test used
// nRanks=31 (below threshold, disabled). This proves gfx950's own ">= 32"
// comparison also correctly enables PXN on the other side of its boundary --
// previously only proven for gfx942 (a completely different branch).
TEST(WrapMicrotest, SetPxn_Gfx950AtOrAboveThresholdEnablesPxn) {
  ncclComm* comm = MakeCommWithArch("gfx950");
  comm->nRanks = 32; // exactly at the gfx950 threshold
  SetMicroEnvAbsent("NCCL_PXN_DISABLE");
  int rcclPxnDisable = -100;
  rcclSetPxn(comm, rcclPxnDisable);
  EXPECT_EQ(0, rcclPxnDisable);
  EXPECT_TRUE(comm->enableCustColl); // enableCustColl = !pxnDisable = !0 = true
  ClearMicroEnv();
  DeleteCommWithArch(comm);
}

// rcclSetP2pNetChunkSize's final `else WARN(...)` arm (rccl_wrap.cc:1409-1411)
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

TEST(WrapMicrotest, SetP2pNetChunkSize_Gfx942BelowThresholdUsesSmallChunk) {
  // Below 64: distinguishes the gfx942 ternary's small-chunk arm from the
  // large-chunk one above -- the only other gfx942 case uses nRanks = 64.
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nRanks = 10;
  SetMicroEnvAbsent("NCCL_P2P_NET_CHUNKSIZE");
  int rcclP2pNetChunkSize = -100;
  rcclSetP2pNetChunkSize(comm, rcclP2pNetChunkSize);
  EXPECT_EQ(1 << 17, rcclP2pNetChunkSize);
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

// Closes the gap flagged above: gfx950's high tier (nRanks >= 32, 1<<19) had
// no test at all -- the only place 1<<19 was previously verified was via
// gfx942's completely separate above-threshold test.
TEST(WrapMicrotest, SetP2pNetChunkSize_Gfx950HighRangeUsesLargeChunk) {
  ncclComm* comm = MakeCommWithArch("gfx950");
  comm->nRanks = 32; // exactly at gfx950's high-tier threshold
  SetMicroEnvAbsent("NCCL_P2P_NET_CHUNKSIZE");
  int rcclP2pNetChunkSize = -100;
  rcclSetP2pNetChunkSize(comm, rcclP2pNetChunkSize);
  EXPECT_EQ(1 << 19, rcclP2pNetChunkSize);
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

// ===========================================================================
// rcclUseHierarchicalAllGather -- rccl_wrap.cc:706-713. No cached statics
// (rcclParamHierarchicalAllGather's real default is 1, the "enabled" value,
// so the interesting branches are reachable without the g_loadParam seam);
// plain comm-field setup, no isolation needed.
// ===========================================================================

TEST(WrapMicrotest, UseHierarchicalAllGather_FewerThan8NodesReturnsFalse) {
  ncclComm* comm = MakeZeroedComm();
  comm->nNodes = 7;
  EXPECT_FALSE(rcclUseHierarchicalAllGather(comm, /*msgSize=*/1024));
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, UseHierarchicalAllGather_NotInitializedReturnsFalse) {
  ncclComm* comm = MakeZeroedComm();
  comm->nNodes = 8;
  comm->hierarchicalCommsInitialized = false;
  EXPECT_FALSE(rcclUseHierarchicalAllGather(comm, /*msgSize=*/1024));
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, UseHierarchicalAllGather_WithinThresholdReturnsTrue) {
  ncclComm* comm = MakeZeroedComm();
  comm->nNodes = 8; // rcclHierarchicalTempBufferSize(8, true, false) == 32MiB
  comm->hierarchicalCommsInitialized = true;
  // Exactly at the threshold: distinguishes the guard's <= from a plain <.
  EXPECT_TRUE(rcclUseHierarchicalAllGather(comm, /*msgSize=*/1ull << 25));
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, UseHierarchicalAllGather_AboveThresholdReturnsFalse) {
  ncclComm* comm = MakeZeroedComm();
  comm->nNodes = 8;
  comm->hierarchicalCommsInitialized = true;
  EXPECT_FALSE(rcclUseHierarchicalAllGather(comm, /*msgSize=*/(1ull << 25) + 1)); // > 32MiB
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotestIsolated, UseHierarchicalAllGather_ParamDisabledReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseHierarchicalAllGather_ParamDisabledReturnsFalse",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_HIERARCHICAL_ALLGATHER") == 0 ? int64_t(0) : deft;
        };
        ncclComm* comm = MakeZeroedComm();
        comm->nNodes = 8;
        comm->hierarchicalCommsInitialized = true;
        EXPECT_FALSE(rcclUseHierarchicalAllGather(comm, /*msgSize=*/1024));
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclUseAllGatherDirect -- rccl_wrap.cc:715-764. Caches the RCCL_PARAM
// disable-flag and a real getenv() threshold probe, both in function-local
// statics; isolated per case.
// ===========================================================================

TEST(WrapMicrotestIsolated, UseAllGatherDirect_ParamDisabledReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseAllGatherDirect_ParamDisabledReturnsFalse",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_DIRECT_ALLGATHER_DISABLE") == 0 ? int64_t(1) : deft;
        };
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nRanks = 8; // rankMultiple == 0 -- isolates this guard from the final !rankMultiple conjunct
        size_t msgSize = 1024;
        EXPECT_FALSE(rcclUseAllGatherDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseAllGatherDirect_Gfx950WithinAutoThresholdReturnsTrue) {
  RUN_ISOLATED_TEST(
      "Wrap_UseAllGatherDirect_Gfx950WithinAutoThresholdReturnsTrue",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 1; // auto threshold -> 8MiB
        comm->nRanks = 8; // rankMultiple == 0
        size_t msgSize = 1024;
        EXPECT_TRUE(rcclUseAllGatherDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseAllGatherDirect_CtaPolicyZeroDisablesOnSingleNode) {
  RUN_ISOLATED_TEST(
      "Wrap_UseAllGatherDirect_CtaPolicyZeroDisablesOnSingleNode",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 1;
        comm->nRanks = 8; // rankMultiple == 0 -- isolates this guard from the final !rankMultiple conjunct
        comm->symmetricSupport = 1;
        comm->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
        size_t msgSize = 1024;
        EXPECT_FALSE(rcclUseAllGatherDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

// Closes a real gap: every prior gfx950 test used nNodes==1 (the "else if
// (comm->nNodes < 64)" arm at rccl_wrap.cc:751 had never executed at all --
// llvm-cov showed 0 hits in either direction). nNodes=2 skips the ==1 arm
// and lands in the node-scaled branch (threshold = nNodes * 2097152).
TEST(WrapMicrotestIsolated, UseAllGatherDirect_Gfx950MultiNodeUnder64UsesNodeScaledThreshold) {
  RUN_ISOLATED_TEST(
      "Wrap_UseAllGatherDirect_Gfx950MultiNodeUnder64UsesNodeScaledThreshold",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 2; // > 1 (skips the ==1 arm), < 64 (this arm's own gate)
        comm->nRanks = 8; // rankMultiple == 0
        size_t msgSize = 4194304; // exactly nNodes(2) * 2097152 -- at the computed threshold
        EXPECT_TRUE(rcclUseAllGatherDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

// Pins that gfx942 takes the ELSE-IF arm (flat 4MiB), not the gfx950 one.
// Every other test here uses gfx950, so nothing separated the two arms: a
// sweep showed the `!userThresholdInput && IsArchMatch(gfx950)` -> `||` mutant
// survives, because `!userThresholdInput` alone is then enough to enter the
// gfx950 block on ANY arch, which for single-node raises the threshold to
// 8MiB. msgSize sits between the two (4MiB < 6MiB < 8MiB), so the correct arm
// rejects it and the mutated one accepts it.
TEST(WrapMicrotestIsolated, UseAllGatherDirect_Gfx942UsesItsOwnFlatThresholdNotGfx950s) {
  RUN_ISOLATED_TEST(
      "Wrap_UseAllGatherDirect_Gfx942UsesItsOwnFlatThresholdNotGfx950s",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 1;
        comm->nRanks = 8;          // rankMultiple == 0
        comm->symmetricSupport = 0; // keeps the CTA-policy guard above from firing
        size_t msgSize = 6000000;   // > gfx942's 4194304, < gfx950 single-node's 8388608
        EXPECT_FALSE(rcclUseAllGatherDirect(comm, msgSize))
            << "gfx942's threshold is a flat 4MiB; it must not pick up gfx950's 8MiB single-node value";
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseAllGatherDirect_NonMultipleOf8RanksReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseAllGatherDirect_NonMultipleOf8RanksReturnsFalse",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 1;
        comm->nRanks = 9; // 9 % 8 != 0
        size_t msgSize = 1024;
        EXPECT_FALSE(rcclUseAllGatherDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

// rcclUseAinic() had no controllable seam before (hardcoded false), so this
// branch had literally never fired -- now a seam, closing a real,
// previously-structural gap.
TEST(WrapMicrotestIsolated, UseAllGatherDirect_AinicDisablesDirect) {
  RUN_ISOLATED_TEST(
      "Wrap_UseAllGatherDirect_AinicDisablesDirect",
      []() {
        ScopedHook useAinic(g_useAinic, []() { return true; });
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 1;
        comm->nRanks = 8;
        size_t msgSize = 1024;
        EXPECT_FALSE(rcclUseAllGatherDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclUseReduceScatterDirect -- rccl_wrap.cc:1319-1357. Caches the
// RCCL_PARAM disable-flag in a function-local static; isolated per case.
// ===========================================================================

TEST(WrapMicrotestIsolated, UseReduceScatterDirect_ParamDisabledReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseReduceScatterDirect_ParamDisabledReturnsFalse",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_DIRECT_REDUCE_SCATTER_DISABLE") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeCommWithArch("gfx950");
        // nNodes=8 + a small msgSize would return true unconditionally (line
        // 1353) if this guard were deleted -- isolates the guard from the
        // nNodes==1 default, which otherwise falls through to the same
        // false via the unrelated final `return false;`.
        comm->nNodes = 8;
        size_t msgSize = 1024;
        EXPECT_FALSE(rcclUseReduceScatterDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseReduceScatterDirect_NonGfx950ReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseReduceScatterDirect_NonGfx950ReturnsFalse",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx942");
        // Same reasoning as the ParamDisabledReturnsFalse test above: nNodes=8
        // would make this call return true if the arch guard were deleted.
        comm->nNodes = 8;
        size_t msgSize = 1024;
        EXPECT_FALSE(rcclUseReduceScatterDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

// ncclPxnDisable() had no controllable seam before (hardcoded 0), so this
// branch had literally never fired -- now a seam, closing a real,
// previously-structural gap.
TEST(WrapMicrotestIsolated, UseReduceScatterDirect_PxnDisabledReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseReduceScatterDirect_PxnDisabledReturnsFalse",
      []() {
        ScopedHook pxnDisable(g_pxnDisable, [](struct ncclComm*) { return 1; });
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 8;
        size_t msgSize = 1024;
        EXPECT_FALSE(rcclUseReduceScatterDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseReduceScatterDirect_TwoNodesInRangeReturnsTrue) {
  RUN_ISOLATED_TEST(
      "Wrap_UseReduceScatterDirect_TwoNodesInRangeReturnsTrue",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 2;
        size_t msgSize = 1048576; // within [128KiB, 2MiB]
        EXPECT_TRUE(rcclUseReduceScatterDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseReduceScatterDirect_TwoNodesAtUpperBoundReturnsTrue) {
  RUN_ISOLATED_TEST(
      "Wrap_UseReduceScatterDirect_TwoNodesAtUpperBoundReturnsTrue",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 2;
        // Exactly at the 2MiB upper bound: the sibling test above (1MiB) is
        // comfortably inside the range and can't distinguish this guard's
        // <= from a plain <.
        size_t msgSize = 2097152;
        EXPECT_TRUE(rcclUseReduceScatterDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseReduceScatterDirect_TwoNodesBelowRangeReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseReduceScatterDirect_TwoNodesBelowRangeReturnsFalse",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 2;
        size_t msgSize = 65536; // below the 128KiB floor
        EXPECT_FALSE(rcclUseReduceScatterDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

// Complementary proof to TwoNodesAtUpperBoundReturnsTrue: msgSize is above
// the 2MiB tier ceiling but still under the 8MiB hard limit (so it reaches
// this guard at all, rather than returning false earlier at line 1349).
TEST(WrapMicrotestIsolated, UseReduceScatterDirect_TwoNodesAboveUpperBoundReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseReduceScatterDirect_TwoNodesAboveUpperBoundReturnsFalse",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 2;
        size_t msgSize = 4194304; // > 2MiB tier ceiling, < 8MiB hard limit
        EXPECT_FALSE(rcclUseReduceScatterDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseReduceScatterDirect_FourNodesUpToLimitReturnsTrue) {
  RUN_ISOLATED_TEST(
      "Wrap_UseReduceScatterDirect_FourNodesUpToLimitReturnsTrue",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 4;
        size_t msgSize = 4194304; // exactly at the 4-node 4MiB limit
        EXPECT_TRUE(rcclUseReduceScatterDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseReduceScatterDirect_EightNodesUnconditionallyTrue) {
  RUN_ISOLATED_TEST(
      "Wrap_UseReduceScatterDirect_EightNodesUnconditionallyTrue",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 8;
        size_t msgSize = 8388608; // at the 8MiB hard limit
        EXPECT_TRUE(rcclUseReduceScatterDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseReduceScatterDirect_SixteenNodesWithinLimitReturnsTrue) {
  RUN_ISOLATED_TEST(
      "Wrap_UseReduceScatterDirect_SixteenNodesWithinLimitReturnsTrue",
      []() {
        // The sibling test below only ever exceeds the 8MiB hard limit, which
        // returns false via the earlier `msgSize > threshold` guard (line
        // 1348) before the nNodes == 16 disjunct (line 1353) is ever reached.
        // This one stays within the limit, so the disjunct itself is what
        // has to fire to get true.
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 16;
        size_t msgSize = 8388608;
        EXPECT_TRUE(rcclUseReduceScatterDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseReduceScatterDirect_SixteenNodesAboveHardLimitReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseReduceScatterDirect_SixteenNodesAboveHardLimitReturnsFalse",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 16;
        size_t msgSize = 8388608 + 1;
        EXPECT_FALSE(rcclUseReduceScatterDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

// The final `return false` (rccl_wrap.cc:1357) had never executed at all --
// every prior test used a node count matching one of the four tiered
// checks (2/4/8/16). A node count outside all four (32) falls through every
// one of them and reaches this fallback directly.
TEST(WrapMicrotestIsolated, UseReduceScatterDirect_UnsupportedNodeCountReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseReduceScatterDirect_UnsupportedNodeCountReturnsFalse",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 32; // not 2, 4, 8, or 16
        size_t msgSize = 1024; // comfortably within the 8MiB hard limit
        EXPECT_FALSE(rcclUseReduceScatterDirect(comm, msgSize));
        DeleteCommWithArch(comm);
      });
}

// *** CONFIRMED REAL BUG, EMPIRICALLY VERIFIED *** in rcclUseReduceScatterDirect
// (rccl_wrap.cc): `size_t threshold = ...; if (threshold > -1)` -- comparing an
// unsigned size_t against the literal -1 promotes -1 to SIZE_MAX, so this
// condition is ALWAYS FALSE regardless of what RCCL_DIRECT_REDUCE_SCATTER_THRESHOLD
// is set to. Execution always falls to the `else` branch, hardcoding
// threshold=8388608 no matter what the user configures. This test proves it
// directly: override the threshold to 100 via g_loadParam, use a msgSize
// (1024) that's within the (buggy) 8MiB default but ABOVE the requested
// 100-byte override -- if the override worked, this should return false; it
// doesn't.
//
// This also explains why the bug was never caught by any test above: every
// other test in this section relies on the real default (8388608) rather
// than overriding it, so they all pass regardless of whether the override
// path works. This is the only test in the whole file that actually
// exercises RCCL_DIRECT_REDUCE_SCATTER_THRESHOLD's override path.
//
// The assertion below (EXPECT_TRUE) PINS the current, buggy behavior -- not
// the intended one -- so this test passes today. If/when the production
// `threshold > -1` bug gets fixed to `threshold != (size_t)-1` (matching the
// working pattern used by rcclUseAllGatherDirect's equivalent check), this
// assertion must flip to EXPECT_FALSE. Left failing-if-fixed deliberately,
// so fixing the bug forces a conscious update here rather than silently
// leaving a stale, wrong-direction test behind.
TEST(WrapMicrotestIsolated, UseReduceScatterDirect_ThresholdOverride_PinsConfirmedBug) {
  RUN_ISOLATED_TEST(
      "Wrap_UseReduceScatterDirect_ThresholdOverride_PinsConfirmedBug",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_DIRECT_REDUCE_SCATTER_THRESHOLD") == 0 ? int64_t(100) : deft;
        };
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 8;
        size_t msgSize = 1024; // > the requested 100-byte override, but within the buggy 8MiB fallback
        EXPECT_TRUE(rcclUseReduceScatterDirect(comm, msgSize))
            << "This pins CURRENT (buggy) behavior -- see the comment above. If this now "
               "fails, the production threshold bug was fixed; flip this to EXPECT_FALSE.";
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclUseHierarchicalReduceScatter -- rccl_wrap.cc:1361-1368. Unlike its
// AllGather sibling, rcclParamHierarchicalReduceScatter's real default is 0
// ("disabled"), so without the g_loadParam seam this function can only be
// proven to always return false -- the true-arm needs the seam.
// ===========================================================================

TEST(WrapMicrotest, UseHierarchicalReduceScatter_DefaultParamAlwaysFalse) {
  ncclComm* comm = MakeZeroedComm();
  comm->nNodes = 8;
  comm->hierarchicalCommsInitialized = true;
  EXPECT_FALSE(rcclUseHierarchicalReduceScatter(comm, /*msgSize=*/1024)); // param defaults to 0, not 1
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotestIsolated, UseHierarchicalReduceScatter_EightNodesParamEnabledReturnsTrue) {
  // 8 nodes: distinguishes the real "< 8" gate from a mutated "< 16" -- RS's
  // own threshold is already nonzero (64MiB) at 8 nodes, so the real gate
  // lets this through while a widened one would reject it.
  RUN_ISOLATED_TEST(
      "Wrap_UseHierarchicalReduceScatter_EightNodesParamEnabledReturnsTrue",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_HIERARCHICAL_REDUCE_SCATTER") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeZeroedComm();
        comm->nNodes = 8;
        comm->hierarchicalCommsInitialized = true;
        EXPECT_TRUE(rcclUseHierarchicalReduceScatter(comm, /*msgSize=*/1024));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseHierarchicalReduceScatter_ParamEnabledWithinThresholdReturnsTrue) {
  RUN_ISOLATED_TEST(
      "Wrap_UseHierarchicalReduceScatter_ParamEnabledWithinThresholdReturnsTrue",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_HIERARCHICAL_REDUCE_SCATTER") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeZeroedComm();
        // rcclHierarchicalTempBufferSize(8, false, true) == 0 (RS needs >= 16 nodes); use 16 for a nonzero threshold.
        comm->nNodes = 16; // 128MiB threshold
        comm->hierarchicalCommsInitialized = true;
        EXPECT_TRUE(rcclUseHierarchicalReduceScatter(comm, /*msgSize=*/1024));
        DeleteCommWithArch(comm);
      });
}

// The three tests above only ever return false via the param-default or
// node-count gates; `threshold > 0 && msgSize <= threshold`'s own false arm
// (msgSize too large) had zero coverage. 128MiB + 1 is one byte past the
// 16-node threshold computed above.
TEST(WrapMicrotestIsolated, UseHierarchicalReduceScatter_ParamEnabledMsgSizeAboveThresholdReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseHierarchicalReduceScatter_ParamEnabledMsgSizeAboveThresholdReturnsFalse",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_HIERARCHICAL_REDUCE_SCATTER") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeZeroedComm();
        comm->nNodes = 16; // 128MiB threshold
        comm->hierarchicalCommsInitialized = true;
        EXPECT_FALSE(rcclUseHierarchicalReduceScatter(comm, /*msgSize=*/128 * 1024 * 1024 + 1));
        DeleteCommWithArch(comm);
      });
}

// DefaultParamAlwaysFalse (above, earlier in this file) never sets
// hierarchicalCommsInitialized, so it can't distinguish "param disabled"
// from "not initialized" -- both are false simultaneously there. This
// isolates the third disjunct the same way the AllGather sibling's
// NotInitializedReturnsFalse does: node count and param both point toward
// "true", only hierarchicalCommsInitialized stays false.
TEST(WrapMicrotestIsolated, UseHierarchicalReduceScatter_NotInitializedReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseHierarchicalReduceScatter_NotInitializedReturnsFalse",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_HIERARCHICAL_REDUCE_SCATTER") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeZeroedComm();
        comm->nNodes = 16;
        comm->hierarchicalCommsInitialized = false;
        EXPECT_FALSE(rcclUseHierarchicalReduceScatter(comm, /*msgSize=*/1024));
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclOptThreadBlockSize -- rccl_wrap.cc:1616-1644. Isolated: its own
// maxNthreads[] is a function-local static, same pattern as the already-
// covered rcclSetDefaultBuffSizes.
// ===========================================================================

TEST(WrapMicrotestIsolated, OptThreadBlockSize_UserOverrideUsedDirectlyWhenAligned) {
  RUN_ISOLATED_TEST(
      "Wrap_OptThreadBlockSize_UserOverrideUsedDirectlyWhenAligned",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_THREADS_PER_BLOCK") == 0 ? int64_t(192) : deft; // 192 = 3*64, aligned
        };
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->WarpSize = 64;
        ncclTaskColl info{};
        int nThreads = -1;
        rcclOptThreadBlockSize(comm, &info, /*nBytes=*/1024, nThreads);
        EXPECT_EQ(192, nThreads);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, OptThreadBlockSize_UserOverrideRoundedUpToWarpMultiple) {
  RUN_ISOLATED_TEST(
      "Wrap_OptThreadBlockSize_UserOverrideRoundedUpToWarpMultiple",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_THREADS_PER_BLOCK") == 0 ? int64_t(200) : deft; // not a multiple of 64
        };
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->WarpSize = 64;
        ncclTaskColl info{};
        int nThreads = -1;
        rcclOptThreadBlockSize(comm, &info, /*nBytes=*/1024, nThreads);
        EXPECT_EQ(256, nThreads); // (200/64 + 1) * 64
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, OptThreadBlockSize_UserOverrideBumpedUpToMinimum) {
  RUN_ISOLATED_TEST(
      "Wrap_OptThreadBlockSize_UserOverrideBumpedUpToMinimum",
      []() {
        // The Aligned test above (192 = 3*64) already sits exactly at the
        // minimum, so its body never actually runs -- this uses a value
        // below it (and still a warp multiple, so the rounding-up branch
        // above it doesn't fire either) to reach the bump itself.
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_THREADS_PER_BLOCK") == 0 ? int64_t(64) : deft;
        };
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->WarpSize = 64;
        ncclTaskColl info{};
        int nThreads = -1;
        rcclOptThreadBlockSize(comm, &info, /*nBytes=*/1024, nThreads);
        EXPECT_EQ(192, nThreads); // 3 * WarpSize(64)
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, OptThreadBlockSize_UserOverrideClampedToMax) {
  RUN_ISOLATED_TEST(
      "Wrap_OptThreadBlockSize_UserOverrideClampedToMax",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_THREADS_PER_BLOCK") == 0 ? int64_t(1024) : deft;
        };
        ncclComm* comm = MakeCommWithArch("gfx942"); // maxNthreads[SIMPLE] == RCCL_DEFAULT_MAX_NTHREADS (256)
        comm->WarpSize = 64;
        ncclTaskColl info{};
        int nThreads = -1;
        rcclOptThreadBlockSize(comm, &info, /*nBytes=*/1024, nThreads);
        EXPECT_EQ(RCCL_DEFAULT_MAX_NTHREADS, nThreads);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, OptThreadBlockSize_TreeAlgorithmUsesMaxThreads) {
  RUN_ISOLATED_TEST(
      "Wrap_OptThreadBlockSize_TreeAlgorithmUsesMaxThreads",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 2; // avoid the nNodes==1 arm so the algorithm arm is what sets nThreads
        ncclTaskColl info{};
        info.algorithm = NCCL_ALGO_TREE;
        // info.protocol defaults to 0 (NCCL_PROTO_LL); left at that default,
        // the "else if (protocol == LL)" arm right below the Tree arm
        // silently overwrites nThreads with the *same* value (all of
        // RCCL_DEFAULT_MAX_NTHREADS/RCCL_LL_MAX_NTHREADS/
        // RCCL_GFX950_MAX_NTHREADS/RCCL_SINGLE_NODE_MAX_NTHREADS are 256 --
        // rccl_common.h:50-53), masking a deletion of the Tree arm entirely.
        // SIMPLE keeps that else-if from firing so only the Tree arm can set
        // nThreads.
        info.protocol = NCCL_PROTO_SIMPLE;
        info.func = ncclFuncAllReduce;
        int nThreads = -1;
        rcclOptThreadBlockSize(comm, &info, /*nBytes=*/1024, nThreads);
        EXPECT_EQ(RCCL_DEFAULT_MAX_NTHREADS, nThreads);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, OptThreadBlockSize_SingleNodeUsesHalfThreads) {
  RUN_ISOLATED_TEST(
      "Wrap_OptThreadBlockSize_SingleNodeUsesHalfThreads",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 1;
        ncclTaskColl info{};
        info.algorithm = NCCL_ALGO_RING; // not TREE/PAT, so nNodes==1 is what sets nThreads
        // Same reasoning as the Tree test above: left at the default LL,
        // mutating away the nNodes==1 condition would fall into the
        // protocol==LL else-if arm, which sets the identical 256 value --
        // SIMPLE closes that off too.
        info.protocol = NCCL_PROTO_SIMPLE;
        info.func = ncclFuncAllReduce;
        int nThreads = -1;
        rcclOptThreadBlockSize(comm, &info, /*nBytes=*/1024, nThreads);
        EXPECT_EQ(RCCL_SINGLE_NODE_MAX_NTHREADS, nThreads);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, OptThreadBlockSize_ReduceScatterSmallCountUsesLLThreads) {
  RUN_ISOLATED_TEST(
      "Wrap_OptThreadBlockSize_ReduceScatterSmallCountUsesLLThreads",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 2;
        comm->nRanks = 4;
        ncclTaskColl info{};
        info.algorithm = NCCL_ALGO_RING;
        info.func = ncclFuncReduceScatter;
        info.protocol = NCCL_PROTO_SIMPLE;
        int nThreads = -1;
        rcclOptThreadBlockSize(comm, &info, /*nBytes=*/1024 /* divUp(1024,4)=256 <= 524288 */, nThreads);
        EXPECT_EQ(RCCL_LL_MAX_NTHREADS, nThreads);
        DeleteCommWithArch(comm);
      });
}

// The test above sits at divUp==256, three orders of magnitude under the
// cutoff, so it cannot tell `<= 524288` from `< 524288` -- a mutation sweep
// confirmed that mutant survives it. With RING/SIMPLE at nNodes>1 none of the
// earlier arms fire, so nThreads keeps the caller's sentinel unless THIS
// condition assigns it: -1 vs RCCL_LL_MAX_NTHREADS is the oracle.
TEST(WrapMicrotestIsolated, OptThreadBlockSize_ReduceScatterSmallCountCutoffBoundary) {
  RUN_ISOLATED_TEST(
      "Wrap_OptThreadBlockSize_ReduceScatterSmallCountCutoffBoundary",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 2;
        comm->nRanks = 4;
        ncclTaskColl info{};
        info.algorithm = NCCL_ALGO_RING;
        info.func = ncclFuncReduceScatter;
        info.protocol = NCCL_PROTO_SIMPLE;

        // divUp(2097152, 4) == 524288 exactly -- still "small".
        int atCutoff = -1;
        rcclOptThreadBlockSize(comm, &info, /*nBytes=*/2097152, atCutoff);
        EXPECT_EQ(RCCL_LL_MAX_NTHREADS, atCutoff);

        // divUp(2097153, 4) == 524289 -- one past, so nThreads is left alone.
        int pastCutoff = -1;
        rcclOptThreadBlockSize(comm, &info, /*nBytes=*/2097153, pastCutoff);
        EXPECT_EQ(-1, pastCutoff) << "one byte past the cutoff must not take the LL-threads arm";
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// commSetUnrollFactor -- rccl_wrap.cc:1662-1711. No caching, no isolation
// needed. ncclDevFuncUnrollGenerated is an all-true const array in
// wrap_fakes.cc, so the "not built for this arch" fallback arms (both the
// user-override one and the default-selection one) are unreachable here.
// That is structural, not a gap someone forgot to close: device.h declares
// the table `extern bool const`, and that declaration is transitively
// visible in the fakes TU, so defining it non-const to make it settable is a
// hard compile error rather than a change anyone can just make. Documented
// rather than contrived around; these are the 22 uncovered lines the PR
// description accounts for.
// ===========================================================================

TEST(WrapMicrotest, SetUnrollFactor_ValidUserOverrideSucceeds) {
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deft) {
    return std::strcmp(env, "RCCL_UNROLL_FACTOR") == 0 ? int64_t(NCCL_UNROLL_2) : deft;
  });
  ncclComm* comm = MakeCommWithArch("gfx942");
  EXPECT_EQ(ncclSuccess, commSetUnrollFactor(comm));
  EXPECT_EQ(NCCL_UNROLL_2, comm->unroll);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetUnrollFactor_OutOfRangeOverrideReturnsInvalidArgument) {
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deft) {
    return std::strcmp(env, "RCCL_UNROLL_FACTOR") == 0 ? int64_t(99) : deft;
  });
  ncclComm* comm = MakeCommWithArch("gfx942");
  std::string log = RcclUnitTesting::CaptureLog([&]() { EXPECT_EQ(ncclInvalidArgument, commSetUnrollFactor(comm)); });
  EXPECT_NE(std::string::npos, log.find("Invalid RCCL_UNROLL_FACTOR"));
  DeleteCommWithArch(comm);
}

// The two tests above use NCCL_UNROLL_2 (comfortably valid) and 99 (wildly
// invalid), so neither sits on the `unroll >= NCCL_NUM_UNROLLS` edge -- a
// mutation sweep confirmed the `>=` -> `>` mutant survives both, because it
// only changes the verdict at exactly NCCL_NUM_UNROLLS. These two pin it:
// the last in-range index is accepted, the first out-of-range one is not.
TEST(WrapMicrotest, SetUnrollFactor_UpperBoundIsExclusive) {
  {
    ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deft) {
      return std::strcmp(env, "RCCL_UNROLL_FACTOR") == 0 ? int64_t(NCCL_NUM_UNROLLS - 1) : deft;
    });
    ncclComm* comm = MakeCommWithArch("gfx942");
    EXPECT_EQ(ncclSuccess, commSetUnrollFactor(comm)) << "NCCL_NUM_UNROLLS-1 is the last valid index";
    EXPECT_EQ(NCCL_NUM_UNROLLS - 1, comm->unroll);
    DeleteCommWithArch(comm);
  }
  {
    ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deft) {
      return std::strcmp(env, "RCCL_UNROLL_FACTOR") == 0 ? int64_t(NCCL_NUM_UNROLLS) : deft;
    });
    ncclComm* comm = MakeCommWithArch("gfx942");
    std::string log = RcclUnitTesting::CaptureLog([&]() { EXPECT_EQ(ncclInvalidArgument, commSetUnrollFactor(comm)); });
    EXPECT_NE(std::string::npos, log.find("Invalid RCCL_UNROLL_FACTOR"));
    DeleteCommWithArch(comm);
  }
}

TEST(WrapMicrotest, SetUnrollFactor_Gfx950SingleNodeDefaultsToUnroll1) {
  ncclComm* comm = MakeCommWithArch("gfx950");
  comm->nNodes = 1;
  EXPECT_EQ(ncclSuccess, commSetUnrollFactor(comm));
  EXPECT_EQ(NCCL_UNROLL_1, comm->unroll);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetUnrollFactor_Gfx950MultiNodeDefaultsToUnroll2) {
  ncclComm* comm = MakeCommWithArch("gfx950");
  comm->nNodes = 2;
  EXPECT_EQ(ncclSuccess, commSetUnrollFactor(comm));
  EXPECT_EQ(NCCL_UNROLL_2, comm->unroll);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetUnrollFactor_Gfx908DefaultsToUnroll2) {
  ncclComm* comm = MakeCommWithArch("gfx908");
  EXPECT_EQ(ncclSuccess, commSetUnrollFactor(comm));
  EXPECT_EQ(NCCL_UNROLL_2, comm->unroll);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetUnrollFactor_UnlistedArchDefaultsToUnroll4) {
  ncclComm* comm = MakeCommWithArch("gfx90a");
  EXPECT_EQ(ncclSuccess, commSetUnrollFactor(comm));
  EXPECT_EQ(NCCL_UNROLL_4, comm->unroll);
  DeleteCommWithArch(comm);
}

// Closes gap 1 of 2 flagged above: the production condition is
// `gfx908 || (gfx942 && cuCount > 80)` -- an OR whose second disjunct is a
// compound AND. The gfx908 test above only proves the FIRST disjunct; this
// proves the second disjunct independently, including that cuCount is
// actually read (a broken threshold or an `&&` turned `||` would go
// uncaught otherwise).
TEST(WrapMicrotest, SetUnrollFactor_Gfx942HighCuCountDefaultsToUnroll2) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->cuCount = 81; // > 80
  EXPECT_EQ(ncclSuccess, commSetUnrollFactor(comm));
  EXPECT_EQ(NCCL_UNROLL_2, comm->unroll);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, SetUnrollFactor_Gfx942AtOrBelowCuCountThresholdDefaultsToUnroll4) {
  // 80 itself: distinguishes the guard's > from a plain >=, and proves this
  // gfx942 comm falls through to the generic default rather than NCCL_UNROLL_2
  // when cuCount doesn't clear the threshold.
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->cuCount = 80;
  EXPECT_EQ(ncclSuccess, commSetUnrollFactor(comm));
  EXPECT_EQ(NCCL_UNROLL_4, comm->unroll);
  DeleteCommWithArch(comm);
}

// Closes gap 2 of 2 flagged above: gfx1250 had no test at all, a completely
// untested arch branch distinct from "unlisted defaults to 4".
TEST(WrapMicrotest, SetUnrollFactor_Gfx1250DefaultsToUnroll32) {
  ncclComm* comm = MakeCommWithArch("gfx1250");
  EXPECT_EQ(ncclSuccess, commSetUnrollFactor(comm));
  EXPECT_EQ(NCCL_UNROLL_32, comm->unroll);
  DeleteCommWithArch(comm);
}

// ncclDevFuncUnrollGenerated stays a hardcoded const all-true array (see
// wrap_fakes.cc's comment: confirmed NOT safely convertible to a test seam
// -- device.h's const extern declaration is transitively visible in that
// TU too, so dropping const is a hard compile error, not just a runtime
// risk). commSetUnrollFactor's two "wasn't built for this arch" fallback
// arms stay a documented, structural coverage gap.

// ===========================================================================
// rcclCommSetP2pShiftSize -- rccl_wrap.cc:1714-1725. No caching (the
// RCCL_PARAM redirector doesn't cache), no isolation needed.
// ===========================================================================

TEST(WrapMicrotest, CommSetP2pShiftSize_AtOrAboveLog2UsesBitReversal) {
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deft) {
    // Exactly at nChannelsLog2: distinguishes the guard's >= from a plain >.
    return std::strcmp(env, "RCCL_P2P_SHIFT_SIZE") == 0 ? int64_t(2) : deft;
  });
  ncclComm* comm = MakeZeroedComm();
  comm->p2pnChannels = 4; // countOneBits(4-1) == countOneBits(0b11) == 2; shiftSize(2) >= 2
  EXPECT_EQ(ncclSuccess, rcclCommSetP2pShiftSize(comm));
  EXPECT_EQ(-1, comm->p2pChannelShiftSize);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, CommSetP2pShiftSize_BelowLog2UsesExactValue) {
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deft) {
    return std::strcmp(env, "RCCL_P2P_SHIFT_SIZE") == 0 ? int64_t(1) : deft;
  });
  ncclComm* comm = MakeZeroedComm();
  comm->p2pnChannels = 4; // countOneBits(3) == 2; shiftSize(1) < 2
  EXPECT_EQ(ncclSuccess, rcclCommSetP2pShiftSize(comm));
  EXPECT_EQ(1, comm->p2pChannelShiftSize);
  DeleteCommWithArch(comm);
}

// ===========================================================================
// rcclOverrideChannels -- rccl_wrap.cc:214-276. rcclParamChannelTuningEnable's
// real default is 1 ("enabled"), so the loop body is reachable without the
// g_loadParam seam; the seam is only needed to reach the disabled arm.
// ncclParamMinNchannels/MaxNchannels are hardcoded scalar fakes (not
// RCCL_PARAM-generated inside this file), always -2/-2.
// ===========================================================================

TEST(WrapMicrotest, OverrideChannels_FewerThan2NodesLeavesNcUntouched) {
  ncclComm* comm = MakeZeroedComm();
  comm->nNodes = 1;
  int nc = -100;
  EXPECT_EQ(ncclSuccess, rcclOverrideChannels(comm, ncclFuncAllReduce, /*nBytes=*/1024, nc));
  EXPECT_EQ(-100, nc);
  DeleteCommWithArch(comm);
}

// nNodes == 2 is the first value the `nNodes < 2` guard must let THROUGH.
// The test above uses 1 and the tuning tests below use 4, so neither can tell
// `< 2` from `<= 2` -- a mutation sweep confirmed that mutant survives them.
// This pins the boundary: at exactly 2 nodes the guard must not fire, so the
// function proceeds far enough to overwrite nc.
TEST(WrapMicrotest, OverrideChannels_ExactlyTwoNodesIsNotSkipped) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 2;
  comm->nRanks = 4;
  comm->nChannels = 32;
  comm->config.minCTAs = 1;
  comm->config.maxCTAs = 64;
  comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][0] = 1;
  comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][1] = 2048;
  comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][2] = 8;
  int nc = -100;
  EXPECT_EQ(ncclSuccess, rcclOverrideChannels(comm, ncclFuncAllReduce, /*nBytes=*/4096, nc));
  EXPECT_NE(-100, nc) << "nNodes==2 must pass the `nNodes < 2` guard, not be skipped by it";
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, OverrideChannels_SingleGpuPerNodeLeavesNcUntouched) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 4;
  comm->nRanks = 4; // one GPU per node, not gfx1151
  int nc = -100;
  EXPECT_EQ(ncclSuccess, rcclOverrideChannels(comm, ncclFuncAllReduce, /*nBytes=*/1024, nc));
  EXPECT_EQ(-100, nc);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotestIsolated, OverrideChannels_DisabledByParamLeavesNcUntouched) {
  RUN_ISOLATED_TEST(
      "Wrap_OverrideChannels_DisabledByParamLeavesNcUntouched",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CHANNEL_TUNING_ENABLE") == 0 ? int64_t(0) : deft;
        };
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 4;
        comm->nRanks = 8;
        int nc = -100;
        EXPECT_EQ(ncclSuccess, rcclOverrideChannels(comm, ncclFuncAllReduce, /*nBytes=*/1024, nc));
        EXPECT_EQ(-100, nc);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotest, OverrideChannels_MatchedThresholdWithinBoundsOverridesNc) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 4;
  comm->nRanks = 8;
  comm->nChannels = 32; // maxNChannels = max(nChannels/scalingFactor, ncclParamMaxNchannels()=-2)
  comm->config.minCTAs = 1;
  comm->config.maxCTAs = 64;
  // bytesPerRank = divUp(nBytes, nRanks) = divUp(16384, 8) = 2048, exactly at
  // maxByteThreshold -- distinguishes the range check's <= from a plain <.
  // minByteThreshold (nonzero: 0 collides with CHAN_THRESHOLDS_UNDEFINED)
  comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][0] = 1;
  comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][1] = 2048; // maxByteThreshold
  comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][2] = 8;    // channelCount
  int nc = -100;
  EXPECT_EQ(ncclSuccess, rcclOverrideChannels(comm, ncclFuncAllReduce, /*nBytes=*/16384, nc));
  EXPECT_EQ(8, nc);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, OverrideChannels_MatchedThresholdOutsideCtaBoundsLeavesNcUntouched) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 4;
  comm->nRanks = 8;
  comm->nChannels = 32; // maxNChannels well above channelCount -- isolates the CTA-bounds check being tested
  comm->config.minCTAs = 1;
  comm->config.maxCTAs = 4; // channelCount(8) below is above maxCTAs
  comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][0] = 1; // nonzero: 0 collides with CHAN_THRESHOLDS_UNDEFINED
  comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][1] = 2048;
  comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][2] = 8;
  int nc = -100;
  EXPECT_EQ(ncclSuccess, rcclOverrideChannels(comm, ncclFuncAllReduce, /*nBytes=*/8192, nc));
  EXPECT_EQ(-100, nc); // conflicting bounds -- not overridden
  DeleteCommWithArch(comm);
}

// gfx1151 is the one arch where the "single GPU per node" skip is
// deliberately NOT applied (the `!IsArchMatch(...,"gfx1151")` conjunct);
// every prior single-GPU-per-node test used gfx942, so this conjunct's own
// false side (gfx1151 present) had never fired. Same threshold setup as
// MatchedThresholdWithinBoundsOverridesNc above, just with nRanks==nNodes
// and gfx1151 -- proving the guard doesn't early-return here.
TEST(WrapMicrotest, OverrideChannels_Gfx1151SingleGpuPerNodeStillAppliesTuning) {
  ncclComm* comm = MakeCommWithArch("gfx1151");
  comm->nNodes = 8;
  comm->nRanks = 8; // one GPU per node -- would skip on any other arch
  comm->nChannels = 32;
  comm->config.minCTAs = 1;
  comm->config.maxCTAs = 64;
  comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][0] = 1;
  comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][1] = 2048;
  comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][2] = 8;
  int nc = -100;
  EXPECT_EQ(ncclSuccess, rcclOverrideChannels(comm, ncclFuncAllReduce, /*nBytes=*/16384, nc));
  EXPECT_EQ(8, nc); // tuning applied -- proves the single-GPU-per-node skip didn't fire
  DeleteCommWithArch(comm);
}

// The undefined-threshold `||` has two disjuncts; every prior test only
// ever triggered it via minByteThreshold (the first, short-circuiting the
// second). This proves maxByteThreshold's own check independently:
// minByteThreshold is defined (nonzero) but maxByteThreshold is left at its
// zero-init default.
TEST(WrapMicrotest, OverrideChannels_UndefinedMaxThresholdBreaksLoop) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 4;
  comm->nRanks = 8;
  comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][0] = 1; // defined
  // [0][1] (maxByteThreshold) left at its zero-init default (undefined).
  int nc = -100;
  EXPECT_EQ(ncclSuccess, rcclOverrideChannels(comm, ncclFuncAllReduce, /*nBytes=*/1024, nc));
  EXPECT_EQ(-100, nc);
  DeleteCommWithArch(comm);
}

// ===========================================================================
// rcclUseCeAllReduce -- rccl_wrap.cc:766-831. Caches rcclParamCeAllReduce
// ("enabled") and rcclParamForceCeAllReduce ("force") in function-local
// statics -- rcclParamCeAllReduce's real default (0) fully blocked this
// function before the g_loadParam seam existed. Isolated per case.
// ===========================================================================

TEST(WrapMicrotestIsolated, UseCeAllReduce_DisabledByDefaultWarnsOnce) {
  RUN_ISOLATED_TEST(
      "Wrap_UseCeAllReduce_DisabledByDefaultWarnsOnce",
      []() {
        // "CE AllReduce not enabled" is an INFO log, gated on ncclDebugLevel
        // (defaults to suppressed) unlike this file's WARN-based messages.
        RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_ALL);
        ncclComm* comm = MakeZeroedComm();
        std::string log1 = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_FALSE(rcclUseCeAllReduce(comm, 8, ncclFloat32, ncclSum, nullptr)); });
        EXPECT_NE(std::string::npos, log1.find("CE AllReduce not enabled"));
        std::string log2 = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_FALSE(rcclUseCeAllReduce(comm, 8, ncclFloat32, ncclSum, nullptr)); });
        EXPECT_EQ(std::string::npos, log2.find("CE AllReduce not enabled")); // warn-once latch
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseCeAllReduce_BiasBufferPresentReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseCeAllReduce_BiasBufferPresentReturnsFalse",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeZeroedComm();
        int bias = 0;
        EXPECT_FALSE(rcclUseCeAllReduce(comm, 8, ncclFloat32, ncclSum, &bias));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseCeAllReduce_NoSymmetricSupportReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseCeAllReduce_NoSymmetricSupportReturnsFalse",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 0;
        std::string log = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_FALSE(rcclUseCeAllReduce(comm, 8, ncclFloat32, ncclSum, nullptr)); });
        EXPECT_NE(std::string::npos, log.find("symmetric support is not enabled"));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseCeAllReduce_MultiNodeReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseCeAllReduce_MultiNodeReturnsFalse",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        comm->nNodes = 2;
        std::string log = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_FALSE(rcclUseCeAllReduce(comm, 8, ncclFloat32, ncclSum, nullptr)); });
        EXPECT_NE(std::string::npos, log.find("nNodes is not 1"));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseCeAllReduce_CountNotDivisibleByNRanksReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseCeAllReduce_CountNotDivisibleByNRanksReturnsFalse",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        comm->nNodes = 1;
        comm->nRanks = 4;
        std::string log = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_FALSE(rcclUseCeAllReduce(comm, /*count=*/5, ncclFloat32, ncclSum, nullptr)); });
        EXPECT_NE(std::string::npos, log.find("is not divisible by nRanks"));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseCeAllReduce_MsgTooLargeReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseCeAllReduce_MsgTooLargeReturnsFalse",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        comm->nNodes = 1;
        comm->nRanks = 1;
        // count * sizeof(float) must exceed NCCL_CE_AR_MAX_MSG_BYTES (256MiB).
        size_t count = (256ull * 1024 * 1024 / 4) + 1;
        std::string log = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_FALSE(rcclUseCeAllReduce(comm, count, ncclFloat32, ncclSum, nullptr)); });
        EXPECT_NE(std::string::npos, log.find("msgBytes"));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseCeAllReduce_NonZeroCtaPolicyWithoutForceReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseCeAllReduce_NonZeroCtaPolicyWithoutForceReturnsFalse",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        comm->nNodes = 1;
        comm->nRanks = 1;
        comm->config.CTAPolicy = NCCL_CTA_POLICY_DEFAULT;
        std::string log = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_FALSE(rcclUseCeAllReduce(comm, 8, ncclFloat32, ncclSum, nullptr)); });
        EXPECT_NE(std::string::npos, log.find("CTA policy is not ZERO"));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseCeAllReduce_UnsupportedOpReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseCeAllReduce_UnsupportedOpReturnsFalse",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        comm->nNodes = 1;
        comm->nRanks = 1;
        comm->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
        EXPECT_FALSE(rcclUseCeAllReduce(comm, 8, ncclFloat32, ncclAvg, nullptr));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseCeAllReduce_Float8DatatypeReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_UseCeAllReduce_Float8DatatypeReturnsFalse",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        comm->nNodes = 1;
        comm->nRanks = 1;
        comm->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
        EXPECT_FALSE(rcclUseCeAllReduce(comm, 8, ncclFloat8e4m3, ncclSum, nullptr));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, UseCeAllReduce_ForceBypassesCtaPolicyAndAllValidReturnsTrue) {
  RUN_ISOLATED_TEST(
      "Wrap_UseCeAllReduce_ForceBypassesCtaPolicyAndAllValidReturnsTrue",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          if (std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0) return int64_t(1);
          if (std::strcmp(env, "RCCL_FORCE_CE_ALLREDUCE") == 0) return int64_t(1);
          return deft;
        };
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        comm->nNodes = 1;
        comm->nRanks = 1;
        comm->config.CTAPolicy = NCCL_CTA_POLICY_DEFAULT; // not ZERO -- force must bypass this
        EXPECT_TRUE(rcclUseCeAllReduce(comm, 8, ncclFloat32, ncclSum, nullptr));
        DeleteCommWithArch(comm);
      });
}

// Documents/pins a real production gap (does not change production code):
// of this function's ten early-return guards, eight emit a WARN with NO
// log-once protection. The other two are the exception that shows the intent:
// the disabled-by-default guard latches its message behind a
// `static bool warnedDisabled`, and the `acc != nullptr` guard is silent.
// Every other test in this
// section is single-call-per-process (RUN_ISOLATED_TEST), so none of them
// can reveal this -- this test deliberately calls the SAME non-qualifying
// case (nNodes != 1) twice in one process and shows the WARN fires both
// times, proving the spam-risk is real: any workload that repeatedly calls
// this with the same non-qualifying inputs re-triggers a WARN on every
// single call.
TEST(WrapMicrotestIsolated, UseCeAllReduce_NonLogOnceGuardWarnsOnEveryCall) {
  RUN_ISOLATED_TEST(
      "Wrap_UseCeAllReduce_NonLogOnceGuardWarnsOnEveryCall",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        comm->nNodes = 2; // fails the "nNodes is not 1" guard, which has no log-once protection
        std::string log1 = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_FALSE(rcclUseCeAllReduce(comm, 8, ncclFloat32, ncclSum, nullptr)); });
        std::string log2 = RcclUnitTesting::CaptureLog(
            [&]() { EXPECT_FALSE(rcclUseCeAllReduce(comm, 8, ncclFloat32, ncclSum, nullptr)); });
        EXPECT_NE(std::string::npos, log1.find("nNodes is not 1"));
        EXPECT_NE(std::string::npos, log2.find("nNodes is not 1"))
            << "if this now finds nothing, the production WARN gained log-once protection -- "
               "update this test's expectation to match.";
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclDdaEnabled -- rccl_wrap.cc:648-667. rcclParamDdaEnable's real default
// (1) and ncclGroupDepth's real default (0) both favor the "enabled" path,
// so most branches are reachable without the seam; not cached, no isolation
// needed.
// ===========================================================================

TEST(WrapMicrotest, DdaEnabled_DisabledByParamReturnsFalse) {
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deft) {
    return std::strcmp(env, "RCCL_DDA_ENABLE") == 0 ? int64_t(0) : deft;
  });
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nRanks = 8;
  EXPECT_FALSE(rcclDdaEnabled(comm, /*totalBytes=*/1024, /*gfx942Default=*/2048, 0, 0));
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, DdaEnabled_Gfx1250UsesCallerDefaultThreshold) {
  ncclComm* comm = MakeCommWithArch("gfx1250");
  EXPECT_TRUE(rcclDdaEnabled(comm, /*totalBytes=*/1024, 0, 0, /*gfx1250Default=*/2048));
  DeleteCommWithArch(comm);
}

// Same ternary-with-fallback pattern as the gfx950 test below
// (DdaEnabled_Gfx950FallsBackToParamThresholdWhenDefaultZero), but for
// gfx1250 -- previously only the caller-provides-a-real-default case was
// exercised; this closes the fallback-path gap.
TEST(WrapMicrotest, DdaEnabled_Gfx1250FallsBackToParamThresholdWhenDefaultZero) {
  ncclComm* comm = MakeCommWithArch("gfx1250");
  // gfx1250Default == 0 -> falls back to rcclParamDdaThreshold()'s real default (128MiB); well within it.
  EXPECT_TRUE(rcclDdaEnabled(comm, /*totalBytes=*/1024, 0, 0, /*gfx1250Default=*/0));
  DeleteCommWithArch(comm);
}

// Closes gap 1 of 3 flagged in the 3-way disable guard: only
// rcclParamDdaEnable had a dedicated test proving it alone disables DDA.
// This proves the second disjunct, ncclParamLaunchOrderImplicit() != 0,
// independently -- everything else here (arch/rank/size) is set up
// identically to a case that would otherwise return true.
TEST(WrapMicrotest, DdaEnabled_ImplicitLaunchOrderReturnsFalse) {
  ScopedHook launchOrder(g_paramLaunchOrderImplicit, []() { return int64_t(1); });
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nRanks = 8;
  EXPECT_FALSE(rcclDdaEnabled(comm, /*totalBytes=*/1024, /*gfx942Default=*/2048, 0, 0));
  DeleteCommWithArch(comm);
}

// Closes gap 2 of 3: proves the third disjunct, ncclGroupDepth != 0,
// independently. ncclGroupDepth is the real thread_local global from
// group.cc (already linked, see wrap-test.cc's file header comment) -- no
// seam needed, just set and restore it directly around the call.
TEST(WrapMicrotest, DdaEnabled_InsideGroupReturnsFalse) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nRanks = 8;
  ncclGroupDepth = 1;
  EXPECT_FALSE(rcclDdaEnabled(comm, /*totalBytes=*/1024, /*gfx942Default=*/2048, 0, 0));
  ncclGroupDepth = 0; // restore: other tests in this binary assume the default
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, DdaEnabled_Gfx942BelowRankThresholdReturnsFalse) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nRanks = 7; // below the 8-rank floor
  EXPECT_FALSE(rcclDdaEnabled(comm, /*totalBytes=*/1024, /*gfx942Default=*/2048, 0, 0));
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, DdaEnabled_Gfx942WithinThresholdReturnsTrue) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nRanks = 8;
  EXPECT_TRUE(rcclDdaEnabled(comm, /*totalBytes=*/2048, /*gfx942Default=*/2048, 0, 0));
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, DdaEnabled_Gfx942AboveThresholdReturnsFalse) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nRanks = 8;
  EXPECT_FALSE(rcclDdaEnabled(comm, /*totalBytes=*/2049, /*gfx942Default=*/2048, 0, 0));
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, DdaEnabled_Gfx950FallsBackToParamThresholdWhenDefaultZero) {
  ncclComm* comm = MakeCommWithArch("gfx950");
  comm->nRanks = 8;
  // gfx950Default == 0 -> falls back to rcclParamDdaThreshold()'s real default (128MiB); well within it.
  EXPECT_TRUE(rcclDdaEnabled(comm, /*totalBytes=*/1024, 0, /*gfx950Default=*/0, 0));
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, DdaEnabled_UnsupportedArchReturnsFalse) {
  ncclComm* comm = MakeCommWithArch("gfx90a");
  comm->nRanks = 8;
  EXPECT_FALSE(rcclDdaEnabled(comm, /*totalBytes=*/1024, /*gfx942Default=*/2048, /*gfx950Default=*/2048, 0));
  DeleteCommWithArch(comm);
}

// ===========================================================================
// rcclSymkQuery -- rccl_wrap.cc:547-568 (static). Only the four early-return
// guards, reachable through plain comm/arg setup: everything past
// ncclSymkInitOnce() is an abort-floor stub (ncclSymkInitOnce/Available/
// PickKernel) in the default fake configuration; the deep path past these
// guards is covered further down, once those stubs are driven as hooks.
// rcclSymKGetInfo -- rccl_wrap.cc:571-583. This section covers its
// null-arg-check arm only; its success path (via rcclSymkQuery) and its
// fall-through path (via rcclGetCollImplInfo) are both exercised further
// down, once those callees' seams are driven.
// ===========================================================================

// Isolated (not just plain TEST()): every guard here is one mutation away
// from falling through into the ncclSymkInitOnce()/etc. abort floor, which
// would crash the whole binary in an in-process test instead of just
// failing one case.
// EXPECT_FALSE alone is a weak oracle here: almost every guard in this
// function returns false, so "returned false" does not prove the NULL check
// is what did it. A sweep confirmed that -- turning the `||` into `&&` still
// passes, because the function then falls through to ncclSymkInitOnce, whose
// fake also yields false. (The mutant makes the second operand dereference a
// null comm, which is undefined behaviour the compiler is free to fold away,
// so it does not even crash.) The seam below is a tripwire: reaching it at all
// means the guard did not return early.
TEST(WrapMicrotestIsolated, SymkQuery_NullCommReturnsBeforeTouchingSymk) {
  RUN_ISOLATED_TEST(
      "Wrap_SymkQuery_NullCommReturnsBeforeTouchingSymk",
      []() {
        static bool symkReached = false;
        symkReached = false;
        ScopedHook initOnce(g_symkInitOnce, [](struct ncclComm*) {
          symkReached = true;
          return ncclSuccess;
        });
        int algo, protocol, maxChannels;
        EXPECT_FALSE(rcclSymkQuery(nullptr, ncclFuncAllReduce, 8, ncclFloat32, ncclSum, &algo, &protocol,
                                    &maxChannels));
        EXPECT_FALSE(symkReached) << "a null comm must return at the first guard, not fall through to symk init";
      });
}

TEST(WrapMicrotestIsolated, SymkQuery_NoSymmetricSupportReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_SymkQuery_NoSymmetricSupportReturnsFalse",
      []() {
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 0;
        int algo, protocol, maxChannels;
        EXPECT_FALSE(
            rcclSymkQuery(comm, ncclFuncAllReduce, 8, ncclFloat32, ncclSum, &algo, &protocol, &maxChannels));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SymkQuery_UnsupportedCollectiveReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_SymkQuery_UnsupportedCollectiveReturnsFalse",
      []() {
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        int algo, protocol, maxChannels;
        EXPECT_FALSE(
            rcclSymkQuery(comm, ncclFuncBroadcast, 8, ncclFloat32, ncclSum, &algo, &protocol, &maxChannels));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SymkQuery_UnmappedRedOpReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_SymkQuery_UnmappedRedOpReturnsFalse",
      []() {
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        int algo, protocol, maxChannels;
        // ncclFuncAllReduce routes op through symkHostRedOpToDev; an op it maps to
        // -1 short-circuits before any abort-floor symk function is reached.
        EXPECT_FALSE(rcclSymkQuery(comm, ncclFuncAllReduce, 8, ncclFloat32, (ncclRedOp_t)9999, &algo, &protocol,
                                    &maxChannels));
        DeleteCommWithArch(comm);
      });
}

// Mutation-testing note: removing any single arm of this null check (e.g.
// dropping the algo==nullptr check) is an equivalent mutant --
// rcclGetCollImplInfo (rccl_wrap.cc:484, what this function falls through
// to whenever rcclSymkQuery returns false) performs the exact same
// algo/protocol/maxChannels null check redundantly, so no input can
// observe the difference. Confirmed by re-applying the mutation directly
// against this build and observing the test still pass, then reverting.
TEST(WrapMicrotestIsolated, SymKGetInfo_NullOutParamReturnsInvalidArgument) {
  RUN_ISOLATED_TEST(
      "Wrap_SymKGetInfo_NullOutParamReturnsInvalidArgument",
      []() {
        ncclComm* comm = MakeZeroedComm();
        int protocol, maxChannels;
        EXPECT_EQ(ncclInvalidArgument,
                  rcclSymKGetInfo(comm, ncclFuncAllReduce, 8, ncclFloat32, ncclSum, /*algo=*/nullptr, &protocol,
                                  &maxChannels));
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// getFirmwareVersion -- rccl_wrap.cc:1728-1735. amd_smi_getFirmwareVersion
// upgraded to a settable hook (fakes/wrap_fakes.h).
// ===========================================================================

TEST(WrapMicrotest, GetFirmwareVersion_SuccessReturnsReportedVersion) {
  ScopedHook fwVersion(g_amdSmiGetFirmwareVersion, [](uint32_t, uint64_t* fw) {
    *fw = 42;
    return ncclSuccess;
  });
  EXPECT_EQ(42, getFirmwareVersion());
}

TEST(WrapMicrotest, GetFirmwareVersion_FailureReturnsNegativeOne) {
  ScopedHook fwVersion(g_amdSmiGetFirmwareVersion, [](uint32_t, uint64_t*) { return ncclInternalError; });
  EXPECT_EQ(-1, getFirmwareVersion());
}

// ===========================================================================
// rcclSelectAllReduce -- rccl_wrap.cc:864-1026. The master AllReduce decision
// function: symmetric -> CE 2-shot -> DDA (fabric LL/LL128/VMM + IPC) -> CE
// registered -> symmetric (reported) -> plain kernel, in that priority
// order. Every test is isolated: rcclUseCeAllReduce (called internally to
// compute ceAllReduceAllowed) caches its enabled/force flags in
// function-local statics, so any test touching RCCL_CE_ALLREDUCE/
// RCCL_FORCE_CE_ALLREDUCE needs a fresh process, and for consistency every
// test here uses RUN_ISOLATED_TEST regardless of whether that specific case
// happens to touch it.
// ===========================================================================

namespace {
// Minimal comm good enough for rcclSelectAllReduce's non-CE/non-DDA-specific
// fields; individual tests override nRanks/etc. as needed. Must have a real
// archName (not a zeroed comm): rcclSelectAllReduce's DDA step unconditionally
// computes IsArchMatch(comm->archName, "gfx1250") once execution reaches it
// (step 4, right after the CE-2-shot early return) regardless of which branch
// is ultimately taken, and IsArchMatch does not null-check its input -- a
// zeroed comm crashes (confirmed via a real test run) the moment any test
// setup doesn't return before that line. gfx90a is neither DDA- nor
// CE-force-eligible, so it doesn't perturb tests that don't care about arch.
ncclComm* MakeSelectComm() {
  ncclComm* comm = MakeCommWithArch("gfx90a");
  comm->nRanks = 1;
  comm->nNodes = 1;
  return comm;
}
}  // namespace

// Proves the `op == ncclSum` guard on symEligible is real, not just
// isSymmetricKernelRequested's own answer: the seam unconditionally says
// "requested", but op is ncclProd -- if the guard were deleted, symEligible
// would wrongly become true and SYMMETRIC would be chosen instead of the
// plain-kernel fallback.
TEST(WrapMicrotestIsolated, SelectAllReduce_SymmetricGatedOnSumOpOnly) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_SymmetricGatedOnSumOpOnly",
      []() {
        ScopedHook symRequested(g_isSymmetricKernelRequested, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                                  size_t, const void*, void*) { return true; });
        ncclComm* comm = MakeSelectComm();
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclProd,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_SYMMETRIC, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// With everything else at its default (safe "not eligible") seam behavior,
// a symmetric-requested sum-op AllReduce lands on SYMMETRIC -- CE-registered
// (the only other branch that could beat it) stays false because
// g_ceAvailable defaults false.
TEST(WrapMicrotestIsolated, SelectAllReduce_SymmetricEligibleChoosesSymmetric) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_SymmetricEligibleChoosesSymmetric",
      []() {
        ScopedHook symRequested(g_isSymmetricKernelRequested, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                                  size_t, const void*, void*) { return true; });
        ncclComm* comm = MakeSelectComm();
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_SYMMETRIC, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// CE registered wins over symmetric when both are eligible, per the
// production comment's documented precedence -- distinct from the test
// above, which only proves symmetric wins when CE ISN'T eligible.
TEST(WrapMicrotestIsolated, SelectAllReduce_CeRegisteredBeatsSymmetricWhenBothEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_CeRegisteredBeatsSymmetricWhenBothEligible",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ScopedHook symRequested(g_isSymmetricKernelRequested, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                                  size_t, const void*, void*) { return true; });
        ScopedHook ceAvailable(
            g_ceAvailable,
            [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) { return true; });
        ncclComm* comm = MakeSelectComm();
        comm->symmetricSupport = 1;
        comm->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// ceAllReduceOpSupported's OR-chain -- op==Sum||Prod||Min||Max -- had only
// ever been proven via Sum (every prior test). ncclProd proves the second
// disjunct fires independently (distinguishing it from a chain that only
// ever checks Sum), still reaching CE_REGISTERED.
TEST(WrapMicrotestIsolated, SelectAllReduce_CeRegisteredChosenWithProdOp) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_CeRegisteredChosenWithProdOp",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ScopedHook ceAvailable(
            g_ceAvailable,
            [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) { return true; });
        ncclComm* comm = MakeSelectComm();
        comm->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclProd,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// Every other CE test hooks g_ceAvailable TRUE and varies something else, so
// nothing proved ncclCeAvailable()'s own false answer is respected. That let
// the `!ceCapturing && ncclCeAvailable(...)` -> `||` mutant survive a sweep:
// with capture off, `!ceCapturing` is true, so `||` short-circuits to true and
// CE is selected even though the availability probe said no. Everything here
// is arranged to favour CE -- supported op, divisible count, CTA_POLICY_ZERO,
// RCCL_CE_ALLREDUCE on -- leaving the probe as the only thing saying no.
TEST(WrapMicrotestIsolated, SelectAllReduce_CeRegisteredNotChosenWhenProbeSaysUnavailable) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_CeRegisteredNotChosenWhenProbeSaysUnavailable",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ScopedHook ceAvailable(
            g_ceAvailable,
            [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) { return false; });
        ncclComm* comm = MakeSelectComm();
        comm->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo)
            << "ncclCeAvailable() returning false must block CE even when everything else favours it";
        DeleteCommWithArch(comm);
      });
}

// Complementary proof: an op the whole OR-chain rejects (ncclAvg) makes
// ceAllReduceOpSupported false, which forces ceAvailable false regardless
// of every other CE-eligibility input -- the chain's "none match" case had
// never been reached before (every prior test used a supported op).
TEST(WrapMicrotestIsolated, SelectAllReduce_CeRegisteredNotChosenWithUnsupportedOp) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_CeRegisteredNotChosenWithUnsupportedOp",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ScopedHook ceAvailable(
            g_ceAvailable,
            [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) { return true; });
        ncclComm* comm = MakeSelectComm();
        comm->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclAvg,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// CE registered chosen on its own gates: ceAvailable (via ncclCeAvailable
// seam), no sysmem segment (default), and CTAPolicy ZERO. hasSysmemSegment's
// default-false is what actually lets this through -- worth knowing this
// test would fail if that default ever changed without a matching update
// here.
TEST(WrapMicrotestIsolated, SelectAllReduce_CeRegisteredChosenWhenAvailableAndPolicyZero) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_CeRegisteredChosenWhenAvailableAndPolicyZero",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ScopedHook ceAvailable(
            g_ceAvailable,
            [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) { return true; });
        ncclComm* comm = MakeSelectComm();
        comm->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// g_devrWindowHasSysmemSegment had a controllable seam (wrap_fakes.cc) but
// no test had ever driven it true anywhere in this file (llvm-cov: 0 hits
// on the true side of ncclDevrWindowHasSysmemSegment at every call site).
// Same setup as CeRegisteredChosenWhenAvailableAndPolicyZero above, but with
// the sysmem-segment guard forced true -- proves it blocks CE_REGISTERED
// even when every other CE-eligibility input stays identical.
TEST(WrapMicrotestIsolated, SelectAllReduce_SysmemSegmentBlocksCeRegistered) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_SysmemSegmentBlocksCeRegistered",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ScopedHook ceAvailable(
            g_ceAvailable,
            [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) { return true; });
        ScopedHook hasSysmem(g_devrWindowHasSysmemSegment, [](struct ncclDevrWindow*) { return true; });
        ncclComm* comm = MakeSelectComm();
        comm->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// Complementary proof: recvWin's own side of hasSysmemSegment's `||`
// (sendWin's already proven above -- both windows got the same hook return
// value there, so recvWin's own check, gated by short-circuit, never ran).
// Distinguished via sentinel addresses (never dereferenced by the fakes).
TEST(WrapMicrotestIsolated, SelectAllReduce_RecvWinSysmemSegmentBlocksCeRegistered) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_RecvWinSysmemSegmentBlocksCeRegistered",
      []() {
        int sendSentinel = 0, recvSentinel = 0;
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ScopedHook ceAvailable(
            g_ceAvailable,
            [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) { return true; });
        ScopedHook findWindow(g_devrFindWindow, [&](struct ncclComm*, void const* ptr, struct ncclDevrWindow** window) {
          *window = reinterpret_cast<struct ncclDevrWindow*>(ptr == &recvSentinel ? 0x2 : 0x1);
          return ncclSuccess;
        });
        ScopedHook hasSysmem(g_devrWindowHasSysmemSegment, [](struct ncclDevrWindow* w) {
          return w == reinterpret_cast<struct ncclDevrWindow*>(0x2); // true only for recvWin's sentinel
        });
        ncclComm* comm = MakeSelectComm();
        comm->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, &sendSentinel, &recvSentinel, /*count=*/8, ncclFloat32,
                                                    ncclSum, /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// CE 2-shot: needs ceAllReduceAllowed (rcclUseCeAllReduce eligible +
// force-or-symReg) AND a non-null ceARTmpBuf. Uses `force` (via
// RCCL_FORCE_CE_ALLREDUCE) as the "force || symReg" side, matching
// rcclUseCeAllReduce's own ForceBypassesCtaPolicy test precedent.
TEST(WrapMicrotestIsolated, SelectAllReduce_CeTwoShotChosenWhenEligibleAndStagingBufferReady) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_CeTwoShotChosenWhenEligibleAndStagingBufferReady",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          if (std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0) return int64_t(1);
          if (std::strcmp(env, "RCCL_FORCE_CE_ALLREDUCE") == 0) return int64_t(1);
          return deft;
        };
        ncclComm* comm = MakeSelectComm();
        comm->symmetricSupport = 1;
        // force bypasses this, matching rcclUseCeAllReduce's precedent
        comm->config.CTAPolicy = NCCL_CTA_POLICY_DEFAULT;
        uint8_t stagingBuf[16];
        comm->ceColl.ceARTmpBuf = stagingBuf;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_CE_2SHOT, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// ceAllReduceAllowed's final conjunct is `force || symReg`; every prior
// test drove it true via one side or the other, but never both false at
// once. Same setup as above (staging buffer ready, rcclUseCeAllReduce
// otherwise eligible), but neither RCCL_FORCE_CE_ALLREDUCE nor a CE-
// available seam is set -- CE 2-shot must not fire despite the buffer
// being ready.
TEST(WrapMicrotestIsolated, SelectAllReduce_CeTwoShotNotChosenWhenNeitherForceNorSymRegEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_CeTwoShotNotChosenWhenNeitherForceNorSymRegEligible",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeSelectComm();
        comm->symmetricSupport = 1;
        comm->config.CTAPolicy = NCCL_CTA_POLICY_ZERO; // rcclUseCeAllReduce itself still eligible
        uint8_t stagingBuf[16];
        comm->ceColl.ceARTmpBuf = stagingBuf;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_CE_2SHOT, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// Complementary proof: ceAllReduceAllowed is true (force set), but
// ceARTmpBuf is left null -- the "staging buffer initialized" conjunct's
// own false side had never fired (every prior test either set the buffer
// or made ceAllReduceAllowed false first).
TEST(WrapMicrotestIsolated, SelectAllReduce_CeTwoShotNotChosenWhenStagingBufferNotInitialized) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_CeTwoShotNotChosenWhenStagingBufferNotInitialized",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          if (std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0) return int64_t(1);
          if (std::strcmp(env, "RCCL_FORCE_CE_ALLREDUCE") == 0) return int64_t(1);
          return deft;
        };
        ncclComm* comm = MakeSelectComm();
        comm->symmetricSupport = 1;
        comm->config.CTAPolicy = NCCL_CTA_POLICY_DEFAULT;
        // comm->ceColl.ceARTmpBuf left at its zero-init default (nullptr).
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_CE_2SHOT, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// DDA fabric LL: gfx1250 arch, rcclAllReduceShouldTakeDdaPath eligible
// (seam), and the LL-specific eligibility check passes.
TEST(WrapMicrotestIsolated, SelectAllReduce_DdaFabricLLChosenOnGfx1250) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_DdaFabricLLChosenOnGfx1250",
      []() {
        ScopedHook shouldTakeDda(g_allReduceShouldTakeDdaPath,
                                 [](const struct ncclComm*, size_t, ncclDataType_t, bool, bool) { return true; });
        ScopedHook llEligible(g_allReduceDdaFabricLLEligible,
                              [](ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx1250");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL, decision.algo);
        EXPECT_EQ(113u, decision.nMaxChannels);  // proves ncclAllReduceDdaFabricLLBlocks ran, not a sibling

        EXPECT_EQ(NCCL_PROTO_LL, decision.protocol);
        DeleteCommWithArch(comm);
      });
}

// DDA fabric LL128: LL not eligible, LL128 is, and RCCL_DDA_LL128 is on --
// distinct from the LL test above, proving the LL arm's own eligibility
// check (not just DDA-path-taken) gates which fabric tier is chosen.
TEST(WrapMicrotestIsolated, SelectAllReduce_DdaFabricLL128ChosenWhenLLNotEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_DdaFabricLL128ChosenWhenLLNotEligible",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_DDA_LL128") == 0 ? int64_t(1) : deft;
        };
        ScopedHook shouldTakeDda(g_allReduceShouldTakeDdaPath,
                                 [](const struct ncclComm*, size_t, ncclDataType_t, bool, bool) { return true; });
        ScopedHook ll128Eligible(g_allReduceDdaFabricLL128Eligible,
                                 [](ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) {
                                   return true;
                                 });
        // g_allReduceDdaFabricLLEligible left at its default (false).
        ncclComm* comm = MakeCommWithArch("gfx1250");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL128, decision.algo);
        EXPECT_EQ(114u, decision.nMaxChannels);  // proves ncclAllReduceDdaFabricLL128Blocks ran, not a sibling

        EXPECT_EQ(NCCL_PROTO_LL128, decision.protocol);
        DeleteCommWithArch(comm);
      });
}

// Distinguishes the LL128 branch's own threshold check (msgBytes <=
// rcclParamDdaLL128Threshold(), default 32MiB) from mere LL128-
// ineligibility (the test above): RCCL_DDA_LL128 stays enabled and the
// LL128 eligibility seam stays true-capable, but msgBytes (~38MiB) exceeds
// the threshold, so the second conjunct -- not eligibility -- is what
// falls through to VMM.
TEST(WrapMicrotestIsolated, SelectAllReduce_DdaFabricVmmChosenWhenLL128ThresholdExceeded) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_DdaFabricVmmChosenWhenLL128ThresholdExceeded",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_DDA_LL128") == 0 ? int64_t(1) : deft;
        };
        ScopedHook shouldTakeDda(g_allReduceShouldTakeDdaPath,
                                 [](const struct ncclComm*, size_t, ncclDataType_t, bool, bool) { return true; });
        ScopedHook ll128Eligible(g_allReduceDdaFabricLL128Eligible,
                                 [](ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) {
                                   return true;
                                 });
        ScopedHook vmmEligible(g_allReduceDdaFabricEligible,
                               [](ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx1250");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        // msgBytes = count(10000000) * sizeof(float32)(4) = ~38MiB, > DdaLL128Threshold (32MiB).
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/10000000, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_FABRIC_VMM, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// DDA fabric VMM: neither LL nor LL128 eligible, falls to the flat/tree
// two-shot VMM path -- the last of gfx1250's three tiers.
TEST(WrapMicrotestIsolated, SelectAllReduce_DdaFabricVmmChosenWhenNeitherLLNorLL128Eligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_DdaFabricVmmChosenWhenNeitherLLNorLL128Eligible",
      []() {
        ScopedHook shouldTakeDda(g_allReduceShouldTakeDdaPath,
                                 [](const struct ncclComm*, size_t, ncclDataType_t, bool, bool) { return true; });
        ScopedHook vmmEligible(g_allReduceDdaFabricEligible,
                               [](ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) { return true; });
        // g_allReduceDdaFabricLLEligible / LL128Eligible left at their default (false).
        ncclComm* comm = MakeCommWithArch("gfx1250");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_FABRIC_VMM, decision.algo);
        EXPECT_EQ(112u, decision.nMaxChannels);  // proves ncclAllReduceDdaFabricBlocks ran, not a sibling

        DeleteCommWithArch(comm);
      });
}

// DDA IPC: non-gfx1250 arch takes the else branch of the fabric-vs-IPC
// split entirely -- proving that split, not just "DDA path taken".
TEST(WrapMicrotestIsolated, SelectAllReduce_DdaIpcChosenOnNonGfx1250Arch) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_DdaIpcChosenOnNonGfx1250Arch",
      []() {
        ScopedHook shouldTakeDda(g_allReduceShouldTakeDdaPath,
                                 [](const struct ncclComm*, size_t, ncclDataType_t, bool, bool) { return true; });
        ScopedHook ipcEligible(g_allReduceDdaIpcEligible,
                               [](ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx942"); // not gfx1250
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_IPC, decision.algo);
        EXPECT_EQ(111u, decision.nMaxChannels);  // proves ncclAllReduceDdaIpcBlocks ran, not a sibling

        DeleteCommWithArch(comm);
      });
}

// Plain-kernel fallback, query mode: nothing else eligible (every seam at
// its default), so decision comes entirely from getAlgoInfo's result --
// proven by overriding getAlgoInfo's default to a distinctive, otherwise-
// impossible channel count.
TEST(WrapMicrotestIsolated, SelectAllReduce_PlainKernelFallbackReportsGetAlgoInfoResultInQueryMode) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_PlainKernelFallbackReportsGetAlgoInfoResultInQueryMode",
      []() {
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->algorithm = NCCL_ALGO_TREE;
          task->protocol = NCCL_PROTO_LL128;
          task->nMaxChannels = 7;
          return ncclSuccess;
        });
        ncclComm* comm = MakeSelectComm();
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ(NCCL_ALGO_TREE, decision.algo);
        EXPECT_EQ(NCCL_PROTO_LL128, decision.protocol);
        EXPECT_EQ(7, decision.nMaxChannels);
        DeleteCommWithArch(comm);
      });
}

// Live mode (query=false) skips getAlgoInfo entirely on the plain-kernel
// fallback path -- taskAppend recomputes it downstream instead. Proves this
// by overriding getAlgoInfo to report an otherwise-impossible channel count
// and confirming decision.nMaxChannels stays at its untouched placeholder
// (0) rather than picking up 999.
TEST(WrapMicrotestIsolated, SelectAllReduce_LiveModeNeverCallsGetAlgoInfo) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_LiveModeNeverCallsGetAlgoInfo",
      []() {
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->nMaxChannels = 999;
          return ncclSuccess;
        });
        ncclComm* comm = MakeSelectComm();
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/false,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ(0, decision.nMaxChannels);
        EXPECT_EQ(0, getAlgo.calls);
        DeleteCommWithArch(comm);
      });
}

// query=true reads ceCapturing directly from graphCapturingHint and never
// probes the real stream (the query runs outside capture, so the stream
// can't reveal graph mode) -- proven via ScopedHook's call counter, not the
// return value, since forcing a "currently capturing" ncclCudaGraph safely
// depends on a ROCM_VERSION-conditional field this test shouldn't assume.
TEST(WrapMicrotestIsolated, SelectAllReduce_QueryModeNeverProbesCapturingGraph) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_QueryModeNeverProbesCapturingGraph",
      []() {
        ScopedHook graphProbe(g_cudaGetCapturingGraph,
                              [](struct ncclCudaGraph*, hipStream_t, int) { return ncclSuccess; });
        ncclComm* comm = MakeSelectComm();
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/true,
                                                    /*graphCapturingHint=*/true, &decision));
        EXPECT_EQ(0, graphProbe.calls);
        EXPECT_TRUE(decision.ceCapturing); // came from graphCapturingHint, not a probe
        DeleteCommWithArch(comm);
      });
}

// Live mode (query=false) DOES probe the real stream via
// ncclCudaGetCapturingGraph -- the complementary proof to the query-mode
// test above.
TEST(WrapMicrotestIsolated, SelectAllReduce_LiveModeProbesCapturingGraph) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_LiveModeProbesCapturingGraph",
      []() {
        ScopedHook graphProbe(g_cudaGetCapturingGraph, [](struct ncclCudaGraph* graph, hipStream_t, int mode) {
          *graph = ncclCudaGraphNone(mode);
          return ncclSuccess;
        });
        ncclComm* comm = MakeSelectComm();
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                    /*stream=*/nullptr, /*query=*/false,
                                                    /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ(1, graphProbe.calls);
        EXPECT_FALSE(decision.ceCapturing); // ncclCudaGraphNone() is not-capturing by construction
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclSelectAllGather -- rccl_wrap.cc:1031-1202. Same overall shape as
// rcclSelectAllReduce, different priority order: DDA -> Hierarchical -> CE
// (force-scratch + registered) -> symmetric (reported) -> Direct -> plain
// kernel. Isolated throughout for the same reason (rcclUseCeAllReduce is not
// called here, but rcclUseAllGatherDirect's own disable-flag/threshold
// statics are reached via the Direct branch and the Hierarchical branch's
// sub-comm dispatch).
// ===========================================================================

TEST(WrapMicrotestIsolated, SelectAllGather_DdaFabricLLChosenOnGfx1250) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_DdaFabricLLChosenOnGfx1250",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_DDA_LL") == 0 ? int64_t(1) : deft;
        };
        ScopedHook llEligible(g_allGatherDdaFabricLLEligible,
                              [](ncclComm*, const void*, void*, size_t, ncclDataType_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx1250");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL, decision.algo);
        EXPECT_EQ(123u, decision.nMaxChannels);  // proves ncclAllGatherDdaFabricLLBlocks ran, not a sibling

        EXPECT_EQ(NCCL_PROTO_LL, decision.protocol);
        DeleteCommWithArch(comm);
      });
}

// The test above sits at msgSize=32 and the LL128 one below at 64KiB, so both
// are far from the 32KiB cutoff and neither can tell `msgSize <= threshold`
// from `<` -- a mutation sweep confirmed that mutant survives. msgSize is
// nRanks * sendcount * typeSize, so with nRanks=1 and float32 a sendcount of
// 8192 lands exactly on rcclParamDdaLLThreshold()'s 32768 default.
TEST(WrapMicrotestIsolated, SelectAllGather_DdaFabricLLThresholdBoundary) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_DdaFabricLLThresholdBoundary",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_DDA_LL") == 0 ? int64_t(1) : deft;
        };
        ScopedHook llEligible(g_allGatherDdaFabricLLEligible,
                              [](ncclComm*, const void*, void*, size_t, ncclDataType_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx1250");
        comm->nRanks = 1;
        comm->nNodes = 1;

        rcclCollDecision atCutoff{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8192, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &atCutoff));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL, atCutoff.algo) << "msgSize == 32768 is still eligible";

        // 8193 * 4 = 32772, one step past the threshold.
        rcclCollDecision pastCutoff{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8193, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &pastCutoff));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL, pastCutoff.algo)
            << "past the threshold the LL branch must be skipped";
        DeleteCommWithArch(comm);
      });
}

// Distinguishes rcclParamDdaLL()'s own value (explicitly disabled here) from
// mere LL-ineligibility (the sibling test below leaves RCCL_DDA_LL at its
// real default, 1/enabled, and instead makes the eligibility seam false) --
// rcclParamDdaLL() itself had never been proven false.
// The LL branch is a three-way `&&`: rcclParamDdaLL(), the msgSize threshold,
// and eligibility. This isolates the FIRST conjunct -- RCCL_DDA_LL is
// explicitly 0 while eligibility stays true-capable and msgSize is tiny, so
// the param alone is what skips LL and falls through to the LL128 check.
TEST(WrapMicrotestIsolated, SelectAllGather_DdaLLExplicitlyDisabledFallsToLL128Check) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_DdaLLExplicitlyDisabledFallsToLL128Check",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          if (std::strcmp(env, "RCCL_DDA_LL") == 0) return int64_t(0);
          if (std::strcmp(env, "RCCL_DDA_LL128") == 0) return int64_t(1);
          return deft;
        };
        ScopedHook llEligible(g_allGatherDdaFabricLLEligible,
                              [](ncclComm*, const void*, void*, size_t, ncclDataType_t) { return true; });
        ScopedHook ll128Eligible(g_allGatherDdaFabricLL128Eligible,
                                 [](ncclComm*, const void*, void*, size_t, ncclDataType_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx1250");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL128, decision.algo);
        EXPECT_EQ(NCCL_PROTO_LL128, decision.protocol);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SelectAllGather_DdaFabricLL128ChosenWhenLLNotEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_DdaFabricLL128ChosenWhenLLNotEligible",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_DDA_LL128") == 0 ? int64_t(1) : deft;
        };
        ScopedHook ll128Eligible(g_allGatherDdaFabricLL128Eligible,
                                 [](ncclComm*, const void*, void*, size_t, ncclDataType_t) { return true; });
        // g_allGatherDdaFabricLLEligible left at its default (false).
        ncclComm* comm = MakeCommWithArch("gfx1250");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL128, decision.algo);
        EXPECT_EQ(124u, decision.nMaxChannels);  // proves ncclAllGatherDdaFabricLL128Blocks ran, not a sibling

        EXPECT_EQ(NCCL_PROTO_LL128, decision.protocol);
        DeleteCommWithArch(comm);
      });
}

// Distinguishes the LL branch's threshold check (msgSize <=
// rcclParamDdaLLThreshold(), default 32KiB) from mere LL-ineligibility
// (the test above): rcclParamDdaLL() stays at its real default (1,
// enabled) and g_allGatherDdaFabricLLEligible is left true-capable, but
// msgSize (64KiB) exceeds the threshold, so the LL branch's second
// conjunct -- not eligibility -- is what skips it, falling to LL128.
TEST(WrapMicrotestIsolated, SelectAllGather_DdaFabricLL128ChosenWhenLLThresholdExceeded) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_DdaFabricLL128ChosenWhenLLThresholdExceeded",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_DDA_LL128") == 0 ? int64_t(1) : deft;
        };
        ScopedHook llEligible(g_allGatherDdaFabricLLEligible,
                              [](ncclComm*, const void*, void*, size_t, ncclDataType_t) { return true; });
        ScopedHook ll128Eligible(g_allGatherDdaFabricLL128Eligible,
                                 [](ncclComm*, const void*, void*, size_t, ncclDataType_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx1250");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        // totalBytes = nRanks(1) * sendcount(16384) * sizeof(float32)(4) = 64KiB,
        // > DdaLLThreshold (32KiB) but well under DdaLL128Threshold (32MiB).
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/16384, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL128, decision.algo);
        EXPECT_EQ(NCCL_PROTO_LL128, decision.protocol);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SelectAllGather_DdaFabricVmmChosenWhenNeitherLLNorLL128Eligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_DdaFabricVmmChosenWhenNeitherLLNorLL128Eligible",
      []() {
        ScopedHook vmmEligible(g_allGatherDdaFabricEligible,
                               [](ncclComm*, const void*, void*, size_t, ncclDataType_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx1250");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_FABRIC_VMM, decision.algo);
        EXPECT_EQ(122u, decision.nMaxChannels);  // proves ncclAllGatherDdaFabricBlocks ran, not a sibling

        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SelectAllGather_DdaIpcChosenOnNonGfx1250Arch) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_DdaIpcChosenOnNonGfx1250Arch",
      []() {
        ScopedHook ipcEligible(g_allGatherDdaIpcEligible,
                               [](ncclComm*, const void*, void*, size_t, ncclDataType_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx942"); // not gfx1250
        // rcclDdaEnabled is called directly here (unlike AllReduce's seam-gated path);
        // it needs nRanks >= 8 for gfx942.
        comm->nRanks = 8;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_IPC, decision.algo);
        EXPECT_EQ(121u, decision.nMaxChannels);  // proves ncclAllGatherDdaIpcBlocks ran, not a sibling

        DeleteCommWithArch(comm);
      });
}

// Proves `!symEligible` really gates DDA here: with DDA fully eligible
// (arch, rcclDdaEnabled via a large threshold default, and the IPC
// eligibility seam) AND symEligible true, DDA must NOT be chosen -- live
// mode (query=false), with nothing else set up, falls all the way to the
// plain-kernel placeholder, confirming DDA was skipped rather than merely
// reporting something else specific.
TEST(WrapMicrotestIsolated, SelectAllGather_DdaGatedOnNotSymEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_DdaGatedOnNotSymEligible",
      []() {
        ScopedHook symRequested(g_isSymmetricKernelRequested, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                                  size_t, const void*, void*) { return true; });
        ScopedHook ipcEligible(g_allGatherDdaIpcEligible,
                               [](ncclComm*, const void*, void*, size_t, ncclDataType_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx942");
        // 9, not 8 or 1: >=8 so rcclDdaEnabled's own rank floor doesn't confound this test
        // (it would pass regardless of symEligible), but NOT a multiple of 8, so
        // rcclUseAllGatherDirect's rankMultiple check can't accidentally fire instead and
        // mask a broken symEligible gate behind a different, still-plausible-looking outcome.
        comm->nRanks = 9;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/false, /*graphCapturingHint=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_DDA_IPC, decision.algo);
        EXPECT_EQ(NCCL_ALGO_RING, decision.algo); // fell all the way to the plain-kernel placeholder
        DeleteCommWithArch(comm);
      });
}

// Hierarchical, live mode: the inner (query-only) inter/intra-comm
// dereferencing never runs, so no sub-comms need to be set up here --
// just the eligibility gates (outside a group, 8+ nodes, param enabled,
// sub-comms marked initialized, within the scratch-buffer threshold).
TEST(WrapMicrotestIsolated, SelectAllGather_HierarchicalChosenLiveMode) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_HierarchicalChosenLiveMode",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 8; // rcclHierarchicalTempBufferSize(8, true, false) == 32MiB
        comm->nRanks = 8;
        comm->hierarchicalCommsInitialized = true;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess,
                  rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32, /*query=*/false,
                                      /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_HIERARCHICAL_ALLGATHER, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// CE force-scratch (branch #2): rcclParamForceCe() is a hardcoded-true stub
// (wrap_fakes.cc), so this only needs ceScratch true (seam), a reg-type that
// is neither fully-registered variant (the default, NonregNonreg, already
// satisfies this), no sysmem segment (default), a non-null ddaScratch, and
// totalBytes within ddaScratchBytes.
TEST(WrapMicrotestIsolated, SelectAllGather_CeForceScratchChosen) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_CeForceScratchChosen",
      []() {
        ScopedHook ceScratch(g_ceScratchAvailable, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                       ncclSymRegType_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nRanks = 1;
        comm->nNodes = 1;
        uint8_t scratch[64];
        comm->ddaScratch = scratch;
        comm->ddaScratchBytes = sizeof(scratch);
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// The mirror of the test above, and the reason it is needed: with the scratch
// path otherwise fully armed (ddaScratch set, size within bounds, ForceCe on,
// reg-type acceptable), the availability probe is the ONLY input still saying
// no. A sweep showed the `!ceCapturing && ncclCeScratchAvailable(...)` -> `||`
// mutant survives without this -- capture is off, so `||` short-circuits true
// and branch #2 fires despite the probe. ceAvailable is left at its default
// (false) so branch #3 cannot mask the result.
TEST(WrapMicrotestIsolated, SelectAllGather_CeForceScratchNotChosenWhenProbeSaysUnavailable) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_CeForceScratchNotChosenWhenProbeSaysUnavailable",
      []() {
        ScopedHook ceScratch(g_ceScratchAvailable, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                       ncclSymRegType_t) { return false; });
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nRanks = 1;
        comm->nNodes = 1;
        uint8_t scratch[64];
        comm->ddaScratch = scratch;
        comm->ddaScratchBytes = sizeof(scratch);
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo)
            << "ncclCeScratchAvailable() returning false must block the force-scratch branch";
        DeleteCommWithArch(comm);
      });
}

// CE registered via symmetric windows (branch #3): distinct from the
// force-scratch test above by leaving ceScratch/ddaScratch at their
// defaults (unset) so branch #2 can't fire, and instead making ceAvailable
// (a different seam) true.
TEST(WrapMicrotestIsolated, SelectAllGather_CeRegisteredViaSymmetricWindowsChosen) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_CeRegisteredViaSymmetricWindowsChosen",
      []() {
        ScopedHook ceAvailable(
            g_ceAvailable,
            [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nRanks = 1;
        comm->nNodes = 1;
        // Branch #3 requires CTA_POLICY_ZERO as of #11389, matching the
        // AllReduce CE-registered gate.
        comm->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo);

        // ...and the CTA-policy conjunct itself: same everything, policy back
        // at its default, so CE must NOT be selected. Without this half, a
        // mutant dropping the new conjunct would go unnoticed.
        comm->config.CTAPolicy = 0;
        rcclCollDecision noPolicy{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &noPolicy));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, noPolicy.algo)
            << "CE via symmetric windows must require CTA_POLICY_ZERO";
        DeleteCommWithArch(comm);
      });
}

// Complementary proof for AllGather's own CE-registered check (a separate
// hasSysmemSegment computation from rcclSelectAllReduce's).
TEST(WrapMicrotestIsolated, SelectAllGather_SysmemSegmentBlocksCeRegistered) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_SysmemSegmentBlocksCeRegistered",
      []() {
        ScopedHook ceAvailable(
            g_ceAvailable,
            [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) { return true; });
        ScopedHook hasSysmem(g_devrWindowHasSysmemSegment, [](struct ncclDevrWindow*) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// hasSysmemSegment's `||` has two disjuncts (sendWin's check, recvWin's
// check); the test above only ever drove it true via sendWin (both
// windows got the same hook return value, so recvWin's own check -- gated
// by short-circuit -- never actually ran). Distinguishing sendWin/recvWin
// via distinct sentinel addresses (never dereferenced by the fakes) proves
// recvWin's side of the `||` independently.
TEST(WrapMicrotestIsolated, SelectAllGather_RecvWinSysmemSegmentBlocksCeRegistered) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_RecvWinSysmemSegmentBlocksCeRegistered",
      []() {
        int sendSentinel = 0, recvSentinel = 0;
        ScopedHook ceAvailable(
            g_ceAvailable,
            [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) { return true; });
        ScopedHook findWindow(g_devrFindWindow, [&](struct ncclComm*, void const* ptr, struct ncclDevrWindow** window) {
          *window = reinterpret_cast<struct ncclDevrWindow*>(ptr == &recvSentinel ? 0x2 : 0x1);
          return ncclSuccess;
        });
        ScopedHook hasSysmem(g_devrWindowHasSysmemSegment, [](struct ncclDevrWindow* w) {
          return w == reinterpret_cast<struct ncclDevrWindow*>(0x2); // true only for recvWin's sentinel
        });
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, &sendSentinel, &recvSentinel, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// Force-scratch's own conjuncts (comm->ddaScratch != nullptr, totalBytes <=
// ddaScratchBytes) had only ever been proven true (CeForceScratchChosen
// above); this proves ddaScratch==nullptr correctly blocks it even though
// every other force-scratch input stays eligible.
TEST(WrapMicrotestIsolated, SelectAllGather_ForceScratchBlockedWhenScratchBufferAbsent) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_ForceScratchBlockedWhenScratchBufferAbsent",
      []() {
        ScopedHook ceScratch(g_ceScratchAvailable, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                       ncclSymRegType_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nRanks = 1;
        comm->nNodes = 1;
        // comm->ddaScratch left at its zero-init default (nullptr).
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// Complementary proof: ddaScratch is present, but totalBytes exceeds
// ddaScratchBytes -- the size guard, not buffer presence, is what blocks it
// here.
TEST(WrapMicrotestIsolated, SelectAllGather_ForceScratchBlockedWhenMessageExceedsScratchCapacity) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_ForceScratchBlockedWhenMessageExceedsScratchCapacity",
      []() {
        ScopedHook ceScratch(g_ceScratchAvailable, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                       ncclSymRegType_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nRanks = 1;
        comm->nNodes = 1;
        uint8_t scratch[64];
        comm->ddaScratch = scratch;
        comm->ddaScratchBytes = sizeof(scratch); // 64 bytes
        rcclCollDecision decision{};
        // totalBytes = nRanks(1) * sendcount(1024) * sizeof(float32)(4) = 4096, > 64.
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/1024, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// Symmetric, reported: only surfaces in query mode, and only after CE
// registered has already been ruled out (both CE seams left at their
// default false here).
TEST(WrapMicrotestIsolated, SelectAllGather_SymmetricReportedWhenQueryAndEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_SymmetricReportedWhenQueryAndEligible",
      []() {
        ScopedHook symRequested(g_isSymmetricKernelRequested, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                                  size_t, const void*, void*) { return true; });
        ScopedHook symkAvailable(g_symkAvailable,
                                 [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t) { return true; });
        ScopedHook symkPick(g_symkPickKernel, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t, size_t,
                                                  int, ncclSymRegType_t, float* estTimeUs,
                                                  ncclSymkKernelId* kernelId, int* maxChannels, int* nWarps,
                                                  bool* forced) {
          *estTimeUs = 0.0f;
          *kernelId = ncclSymkKernelId_AllGather_LL;
          *maxChannels = 4;
          *nWarps = 2;
          *forced = false;
          return ncclSuccess;
        });
        ScopedHook kernelIsLL(g_symkKernelIdIsLL, [](int) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nRanks = 1;
        comm->nNodes = 1;
        comm->symmetricSupport = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_SYMMETRIC, decision.algo);
        EXPECT_EQ(NCCL_PROTO_LL, decision.protocol);
        EXPECT_EQ(4, decision.nMaxChannels);
        DeleteCommWithArch(comm);
      });
}

// Complementary proof: symEligible is true (so this block is entered at
// all), but rcclSymkQuery itself returns false (g_symkAvailable left at its
// default) -- must fall through to whatever's next (here, the plain-kernel
// fallback) rather than reporting RCCL_SYMMETRIC. This distinguishes "block
// entered" from "block's own query succeeded", never proven separately
// before.
TEST(WrapMicrotestIsolated, SelectAllGather_SymmetricEligibleButSymkQueryFailsFallsThrough) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_SymmetricEligibleButSymkQueryFailsFallsThrough",
      []() {
        ScopedHook symRequested(g_isSymmetricKernelRequested, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                                  size_t, const void*, void*) { return true; });
        // g_symkAvailable left at its default (false) -> rcclSymkQuery returns false.
        ncclComm* comm = MakeCommWithArch("gfx90a"); // not Direct-eligible -> falls all the way to plain kernel
        comm->nRanks = 1;
        comm->nNodes = 1;
        comm->symmetricSupport = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_SYMMETRIC, decision.algo);
        EXPECT_EQ(NCCL_ALGO_RING, decision.algo); // plain-kernel placeholder, not overwritten by symk
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SelectAllGather_DirectChosenWhenEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_DirectChosenWhenEligible",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 1; // auto threshold -> 8MiB
        comm->nRanks = 8; // rankMultiple == 0
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DIRECT_ALLGATHER, decision.algo);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SelectAllGather_PlainKernelFallbackReportsGetAlgoInfoResultInQueryMode) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_PlainKernelFallbackReportsGetAlgoInfoResultInQueryMode",
      []() {
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->algorithm = NCCL_ALGO_TREE;
          task->protocol = NCCL_PROTO_LL128;
          task->nMaxChannels = 5;
          return ncclSuccess;
        });
        ncclComm* comm = MakeCommWithArch("gfx90a"); // not a DDA/Direct-eligible arch
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ(NCCL_ALGO_TREE, decision.algo);
        EXPECT_EQ(NCCL_PROTO_LL128, decision.protocol);
        EXPECT_EQ(5, decision.nMaxChannels);
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclSelectReduceScatter -- rccl_wrap.cc:1206-1317. Own priority order:
// symmetric (op sum OR avg -- distinct from AllReduce's sum-only) -> DDA
// (gfx1250 fabric may preempt symmetric eligibility, IPC stays strictly
// gated on !symEligible) -> Hierarchical -> Direct -> symmetric (reported)
// -> plain kernel. NO CE step anywhere -- confirmed empirically below, not
// just by absence of a CE call in the source.
// ===========================================================================

TEST(WrapMicrotestIsolated, SelectReduceScatter_SymmetricGatedOnSumOrAvgOpOnly) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_SymmetricGatedOnSumOrAvgOpOnly",
      []() {
        ScopedHook symRequested(g_isSymmetricKernelRequested, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                                  size_t, const void*, void*) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                        ncclProd, /*query=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_SYMMETRIC, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// Both accepted ops actually reach symmetric -- distinct from AllReduce,
// where only sum is accepted (see rcclSelectAllReduce's own gate test).
TEST(WrapMicrotestIsolated, SelectReduceScatter_SymmetricAcceptsSumOrAvg) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_SymmetricAcceptsSumOrAvg",
      []() {
        ScopedHook symRequested(g_isSymmetricKernelRequested, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                                  size_t, const void*, void*) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx942"); // not gfx1250 -- DDA can't preempt here
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision sumDecision{};
        EXPECT_EQ(ncclSuccess, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                        ncclSum, /*query=*/false, &sumDecision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_SYMMETRIC, sumDecision.algo);
        rcclCollDecision avgDecision{};
        EXPECT_EQ(ncclSuccess, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                        ncclAvg, /*query=*/false, &avgDecision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_SYMMETRIC, avgDecision.algo);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SelectReduceScatter_DdaFabricLLChosenOnGfx1250) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_DdaFabricLLChosenOnGfx1250",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_DDA_LL") == 0 ? int64_t(1) : deft;
        };
        ScopedHook llEligible(g_reduceScatterDdaFabricLLEligible,
                              [](ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) {
                                return true;
                              });
        ncclComm* comm = MakeCommWithArch("gfx1250");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                        ncclSum, /*query=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL, decision.algo);

        EXPECT_EQ(NCCL_PROTO_LL, decision.protocol);
        DeleteCommWithArch(comm);
      });
}

// The genuinely ReduceScatter-unique exception: on gfx1250, DDA fabric is
// allowed to preempt symmetric eligibility entirely (`!symEligible ||
// ddaFabricArch`) -- distinct from every other gate in this function (and
// from AllReduce/AllGather's DDA, which are strictly !symEligible-gated
// with no such override).
TEST(WrapMicrotestIsolated, SelectReduceScatter_Gfx1250DdaPreemptsSymmetricEvenWhenEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_Gfx1250DdaPreemptsSymmetricEvenWhenEligible",
      []() {
        ScopedHook symRequested(g_isSymmetricKernelRequested, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                                  size_t, const void*, void*) { return true; });
        ScopedHook vmmEligible(g_reduceScatterDdaFabricEligible,
                               [](ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) {
                                 return true;
                               });
        ncclComm* comm = MakeCommWithArch("gfx1250");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                        ncclSum, /*query=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_FABRIC_VMM, decision.algo); // DDA won, not SYMMETRIC
        DeleteCommWithArch(comm);
      });
}

// Complementary proof: on a NON-gfx1250 arch, the same symEligible=true +
// DDA-eligible=true setup must NOT let DDA win -- the preemption is
// strictly gfx1250-specific.
TEST(WrapMicrotestIsolated, SelectReduceScatter_NonGfx1250DdaStrictlyGatedOnNotSymEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_NonGfx1250DdaStrictlyGatedOnNotSymEligible",
      []() {
        ScopedHook symRequested(g_isSymmetricKernelRequested, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                                  size_t, const void*, void*) { return true; });
        ScopedHook ipcEligible(g_reduceScatterDdaIpcEligible,
                               [](ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) {
                                 return true;
                               });
        ncclComm* comm = MakeCommWithArch("gfx942"); // not gfx1250
        // 8, not 1: satisfies rcclDdaEnabled's own gfx942 rank floor so this test isolates
        // the symEligible gate specifically, rather than piggybacking on an unrelated
        // rank-count guard that would mask the same gate being broken.
        comm->nRanks = 8;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                        ncclSum, /*query=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_SYMMETRIC, decision.algo); // symmetric won, not DDA_IPC
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SelectReduceScatter_DdaFabricLL128ChosenWhenLLNotEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_DdaFabricLL128ChosenWhenLLNotEligible",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_DDA_LL128") == 0 ? int64_t(1) : deft;
        };
        ScopedHook ll128Eligible(g_reduceScatterDdaFabricLL128Eligible,
                                 [](ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) {
                                   return true;
                                 });
        ncclComm* comm = MakeCommWithArch("gfx1250");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                        ncclSum, /*query=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL128, decision.algo);

        EXPECT_EQ(NCCL_PROTO_LL128, decision.protocol);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SelectReduceScatter_DdaIpcChosenOnNonGfx1250Arch) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_DdaIpcChosenOnNonGfx1250Arch",
      []() {
        ScopedHook ipcEligible(g_reduceScatterDdaIpcEligible,
                               [](ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) {
                                 return true;
                               });
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nRanks = 8; // rcclDdaEnabled (called directly here) needs >=8 for gfx942
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                        ncclSum, /*query=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DDA_IPC, decision.algo);

        DeleteCommWithArch(comm);
      });
}

// Hierarchical, live mode: sidesteps rcclHierarchicalAlgoInfo's inter/intra
// sub-comm dereferencing (only called when query=true), same reasoning as
// the AllGather Hierarchical test.
TEST(WrapMicrotestIsolated, SelectReduceScatter_HierarchicalChosenLiveMode) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_HierarchicalChosenLiveMode",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_HIERARCHICAL_REDUCE_SCATTER") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 16; // rcclHierarchicalTempBufferSize(16, false, true) == 128MiB
        comm->nRanks = 16;
        comm->hierarchicalCommsInitialized = true;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                        ncclSum, /*query=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_HIERARCHICAL_REDUCESCATTER, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// Hierarchical only fires for op==ncclSum -- proven separately since avg is
// otherwise symmetric-eligible territory and could mask this gate.
TEST(WrapMicrotestIsolated, SelectReduceScatter_HierarchicalGatedOnSumOpOnly) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_HierarchicalGatedOnSumOpOnly",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_HIERARCHICAL_REDUCE_SCATTER") == 0 ? int64_t(1) : deft;
        };
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 16;
        comm->nRanks = 16;
        comm->hierarchicalCommsInitialized = true;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                        ncclAvg, /*query=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_HIERARCHICAL_REDUCESCATTER, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// query=true's NCCLCHECK(rcclHierarchicalAlgoInfo(...)) success continuation
// (decision->protocol/nMaxChannels assignment right after it) had never
// executed -- the other query=true test through this path
// (GetCollImplInfo_ReduceScatterHierarchicalGetAlgoInfoFailurePropagates,
// further down this file) deliberately forces failure, so the line right
// after the NCCLCHECK never
// ran. This proves the success continuation independently.
TEST(WrapMicrotestIsolated, SelectReduceScatter_HierarchicalQueryModeSucceeds) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_HierarchicalQueryModeSucceeds",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_HIERARCHICAL_REDUCE_SCATTER") == 0 ? int64_t(1) : deft;
        };
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->protocol = NCCL_PROTO_LL128;
          task->nMaxChannels = 9;
          return ncclSuccess;
        });
        ncclComm* interComm = MakeCommWithArch("gfx90a"); // not Direct-eligible -> falls to getAlgoInfo
        interComm->nRanks = 16;
        interComm->nNodes = 1;
        ncclComm* intraComm = MakeCommWithArch("gfx90a");
        intraComm->nRanks = 1;
        intraComm->nNodes = 1;
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 16;
        comm->nRanks = 16;
        comm->hierarchicalCommsInitialized = true;
        comm->hierarchicalInterComm = interComm;
        comm->hierarchicalIntraComm = intraComm;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                        ncclSum, /*query=*/true, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_HIERARCHICAL_REDUCESCATTER, decision.algo);
        EXPECT_EQ(NCCL_PROTO_LL128, decision.protocol);
        EXPECT_EQ(9, decision.nMaxChannels);
        DeleteCommWithArch(comm);
        DeleteCommWithArch(interComm);
        DeleteCommWithArch(intraComm);
      });
}

TEST(WrapMicrotestIsolated, SelectReduceScatter_DirectChosenWhenEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_DirectChosenWhenEligible",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 2;
        comm->nRanks = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess,
                  rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/1048576 / 4, ncclFloat32, ncclSum,
                                          /*query=*/false, &decision)); // 1MiB total, within [128KiB,2MiB]
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DIRECT_REDUCESCATTER, decision.algo);
        DeleteCommWithArch(comm);
      });
}

// The `op < ncclAvg` conjunct excludes ncclAvg and the PreMulSum ops from the
// Direct path (see the production comment above it). Every Direct test uses
// ncclSum, which is far below ncclAvg, so none can tell `<` from `<=` -- a
// mutation sweep confirmed the `<` -> `<=` mutant survives. ncclAvg is the
// first op the guard must reject, so it is the only value that distinguishes
// them: same comm and size as the test above, only the op differs.
TEST(WrapMicrotestIsolated, SelectReduceScatter_AvgOpIsExcludedFromDirect) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_AvgOpIsExcludedFromDirect",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 2;
        comm->nRanks = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess,
                  rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/1048576 / 4, ncclFloat32, ncclAvg,
                                          /*query=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_DIRECT_REDUCESCATTER, decision.algo)
            << "ncclAvg must not take the Direct ReduceScatter path";
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclHierarchicalAlgoInfo -- rccl_wrap.cc:394-445. Unconditionally
// dereferences comm->hierarchicalInterComm/IntraComm (no query gate, unlike
// the rcclSelectXxx callers' own inner Hierarchical branches) -- every test
// here builds real sub-comms via MakeCommWithArch. Isolated throughout:
// rcclUseAllGatherDirect/rcclUseReduceScatterDirect (called on the
// sub-comms) cache disable-flag/threshold inputs in function-local statics.
// ===========================================================================

namespace {
void DeleteHierarchicalSubComms(ncclComm* inter, ncclComm* intra) {
  DeleteCommWithArch(inter);
  DeleteCommWithArch(intra);
}
}  // namespace

// Both sub-comms are configured Direct-eligible, and the reported channel
// count is inter's p2pnChannels rather than a getAlgoInfo-derived value.
// Only inter's is observable through the return values -- intra's Direct
// eligibility affects a log line, which the sibling test below asserts.
TEST(WrapMicrotestIsolated, HierarchicalAlgoInfo_AllGatherBothPhasesDirectEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_HierarchicalAlgoInfo_AllGatherBothPhasesDirectEligible",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ncclComm* interComm = MakeCommWithArch("gfx950");
        interComm->nRanks = 8; // this IS the function's local `nNodes` (interComm->nRanks)
        interComm->nNodes = 1; // auto threshold -> 8MiB
        interComm->p2pnChannels = 11;
        ncclComm* intraComm = MakeCommWithArch("gfx950");
        intraComm->nRanks = 8;
        intraComm->nNodes = 1;
        intraComm->p2pnChannels = 22;
        ncclComm* comm = MakeZeroedComm();
        comm->hierarchicalInterComm = interComm;
        comm->hierarchicalIntraComm = intraComm;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess,
                  rcclHierarchicalAlgoInfo(comm, ncclFuncAllGather, /*count=*/8, ncclFloat32, &algo, &protocol,
                                           &maxChannels));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_HIERARCHICAL_ALLGATHER, algo);
        EXPECT_EQ(NCCL_PROTO_SIMPLE, protocol);
        EXPECT_EQ(11, maxChannels); // inter's p2pnChannels, not intra's 22 nor a getAlgoInfo value
        DeleteCommWithArch(comm);
        DeleteHierarchicalSubComms(interComm, intraComm);
      });
}

// AllGather's inter-phase Direct check has a hardcoded nNodes<=16 cap (per
// the production comment: "Direct AllGather is only tuned up to 16 nodes").
// With interComm->nRanks=17, Direct is skipped regardless of whether it
// would otherwise qualify, falling to getAlgoInfo instead.
TEST(WrapMicrotestIsolated, HierarchicalAlgoInfo_AllGatherInterAboveNodeCapFallsToGetAlgoInfo) {
  RUN_ISOLATED_TEST(
      "Wrap_HierarchicalAlgoInfo_AllGatherInterAboveNodeCapFallsToGetAlgoInfo",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->protocol = NCCL_PROTO_LL128;
          task->nMaxChannels = 42;
          return ncclSuccess;
        });
        ncclComm* interComm = MakeCommWithArch("gfx950");
        interComm->nRanks = 17; // > 16: caps out Direct regardless of otherwise-qualifying setup
        interComm->nNodes = 1;
        interComm->p2pnChannels = 11; // would be wrongly reported if the cap were broken
        ncclComm* intraComm = MakeCommWithArch("gfx90a"); // not Direct-eligible; falls to getAlgoInfo too
        intraComm->nRanks = 1;
        intraComm->nNodes = 1;
        ncclComm* comm = MakeZeroedComm();
        comm->hierarchicalInterComm = interComm;
        comm->hierarchicalIntraComm = intraComm;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess,
                  rcclHierarchicalAlgoInfo(comm, ncclFuncAllGather, /*count=*/8, ncclFloat32, &algo, &protocol,
                                           &maxChannels));
        EXPECT_EQ(NCCL_PROTO_LL128, protocol);
        EXPECT_EQ(42, maxChannels);
        DeleteCommWithArch(comm);
        DeleteHierarchicalSubComms(interComm, intraComm);
      });
}

// ReduceScatter's inter-phase Direct check has NO node cap (unlike
// AllGather's) -- rcclUseReduceScatterDirect gets called regardless of how
// large interComm->nRanks is. Also proves interDirect's own arch/PXN/size
// gates work through this call path, not just AllGather's.
TEST(WrapMicrotestIsolated, HierarchicalAlgoInfo_ReduceScatterInterDirectHasNoNodeCap) {
  RUN_ISOLATED_TEST(
      "Wrap_HierarchicalAlgoInfo_ReduceScatterInterDirectHasNoNodeCap",
      []() {
        ncclComm* interComm = MakeCommWithArch("gfx950"); // rcclUseReduceScatterDirect is gfx950-only
        interComm->nRanks = 17; // > 16 -- would fail AllGather's cap, but RS has none
        interComm->nNodes = 2;  // rcclUseReduceScatterDirect's 2-node tier: [128KiB, 2MiB]
        interComm->p2pnChannels = 33;
        ncclComm* intraComm = MakeCommWithArch("gfx90a"); // never reached for Direct (RS has no intra-Direct)
        intraComm->nRanks = 1;
        intraComm->nNodes = 1;
        ncclComm* comm = MakeZeroedComm();
        comm->hierarchicalInterComm = interComm;
        comm->hierarchicalIntraComm = intraComm;
        // count*typeSize(4)*nNodes(17) must land in [131072, 2097152].
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess,
                  rcclHierarchicalAlgoInfo(comm, ncclFuncReduceScatter, /*count=*/8192, ncclFloat32, &algo,
                                           &protocol, &maxChannels));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_HIERARCHICAL_REDUCESCATTER, algo);
        EXPECT_EQ(NCCL_PROTO_SIMPLE, protocol);
        EXPECT_EQ(33, maxChannels); // inter's p2pnChannels -- Direct fired despite nNodes=17
        DeleteCommWithArch(comm);
        DeleteHierarchicalSubComms(interComm, intraComm);
      });
}

// The production comment states ReduceScatter never runs Direct intra-node
// ("only AllGather gets the fast path here"). intraProto/intraChan aren't
// exposed as output params (logged only), so this proves it via
// CaptureLog: intraComm is set up to otherwise qualify for Direct (matching
// the AllGather-eligible setup from the first test above), but since coll
// is ReduceScatter, the logged intra values must come from getAlgoInfo's
// seam, not intraComm->p2pnChannels.
TEST(WrapMicrotestIsolated, HierarchicalAlgoInfo_ReduceScatterIntraNeverUsesDirect) {
  RUN_ISOLATED_TEST(
      "Wrap_HierarchicalAlgoInfo_ReduceScatterIntraNeverUsesDirect",
      []() {
        // rcclHierarchicalAlgoInfo's final summary line uses INFO(), not WARN() --
        // unlike every other CaptureLog-based test in this file, this one needs
        // ScopedDebugLogging to raise level+mask first, or CaptureLog observes
        // nothing at all (confirmed: both EXPECT_NE and EXPECT_EQ below "passed"
        // for the wrong reason -- an empty log matches neither substring -- before
        // this fix).
        RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_ALL);
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->protocol = NCCL_PROTO_LL128; // LL128 == 1, distinct from Direct's SIMPLE (2)
          task->nMaxChannels = 99;           // distinct from intraComm->p2pnChannels below
          return ncclSuccess;
        });
        ncclComm* interComm = MakeCommWithArch("gfx90a"); // not Direct-eligible for RS; simple getAlgoInfo path
        interComm->nRanks = 2;
        interComm->nNodes = 1;
        ncclComm* intraComm = MakeCommWithArch("gfx950"); // otherwise Direct-eligible IF this were AllGather
        intraComm->nRanks = 8;
        intraComm->nNodes = 1;
        intraComm->p2pnChannels = 55; // would wrongly appear in the log if isAllGather's gate were broken
        ncclComm* comm = MakeZeroedComm();
        comm->hierarchicalInterComm = interComm;
        comm->hierarchicalIntraComm = intraComm;
        int algo, protocol, maxChannels;
        std::string log = RcclUnitTesting::CaptureLog([&]() {
          EXPECT_EQ(ncclSuccess, rcclHierarchicalAlgoInfo(comm, ncclFuncReduceScatter, /*count=*/8, ncclFloat32,
                                                          &algo, &protocol, &maxChannels));
        });
        EXPECT_NE(std::string::npos, log.find("intra: proto=1 channels=99"));  // NCCL_PROTO_LL128 == 1
        EXPECT_EQ(std::string::npos, log.find("channels=55"));                // p2pnChannels never used
        DeleteCommWithArch(comm);
        DeleteHierarchicalSubComms(interComm, intraComm);
      });
}

// ===========================================================================
// Back to rcclSelectReduceScatter (rccl_wrap.cc:1206-1317) for three cases
// that need the sub-comm helpers defined in the section above, which is why
// they sit here rather than with the rest of that function's tests.
// ===========================================================================

// No CE step anywhere in rcclSelectReduceScatter: confirmed empirically, not
// just by absence of a call in the source. Forces every CE-related seam true
// (the exact combination that WOULD produce CE_REGISTERED in
// rcclSelectAllReduce/AllGather) and confirms the outcome is still
// something else entirely -- that function simply never calls
// ncclCeAvailable/ncclCeScratchAvailable at all.
TEST(WrapMicrotestIsolated, SelectReduceScatter_NeverChoosesCeRegardlessOfCeSeams) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_NeverChoosesCeRegardlessOfCeSeams",
      []() {
        ScopedHook ceAvailable(
            g_ceAvailable,
            [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) { return true; });
        ScopedHook ceScratch(g_ceScratchAvailable, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                       ncclSymRegType_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx90a"); // not DDA/Direct/Hierarchical eligible
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                        ncclSum, /*query=*/false, &decision));
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, decision.algo);
        EXPECT_NE((int)rcclAddonAlgos_t::RCCL_CE_2SHOT, decision.algo);
        EXPECT_EQ(NCCL_ALGO_RING, decision.algo); // fell all the way to the plain-kernel placeholder
        DeleteCommWithArch(comm);
      });
}

// symEligible's own query-mode reporting block (`if (query) { ... }` inside
// `if (symEligible)`) had never been entered for ReduceScatter -- only
// AllReduce/AllGather had a symmetric-reported query test. rcclSymkQuery
// itself stays at its default (false) here; this just proves the block is
// reached at all, matching decision->algo already being RCCL_SYMMETRIC
// either way.
TEST(WrapMicrotestIsolated, SelectReduceScatter_SymmetricReportedWhenQueryAndEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_SymmetricReportedWhenQueryAndEligible",
      []() {
        ScopedHook symRequested(g_isSymmetricKernelRequested, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                                  size_t, const void*, void*) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx90a"); // not DDA/Direct/Hierarchical eligible
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                        ncclSum, /*query=*/true, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_SYMMETRIC, decision.algo);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SelectReduceScatter_PlainKernelFallbackReportsGetAlgoInfoResultInQueryMode) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_PlainKernelFallbackReportsGetAlgoInfoResultInQueryMode",
      []() {
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->algorithm = NCCL_ALGO_TREE;
          task->protocol = NCCL_PROTO_LL128;
          task->nMaxChannels = 3;
          return ncclSuccess;
        });
        ncclComm* comm = MakeCommWithArch("gfx90a");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                        ncclSum, /*query=*/true, &decision));
        EXPECT_EQ(NCCL_ALGO_TREE, decision.algo);
        EXPECT_EQ(NCCL_PROTO_LL128, decision.protocol);
        EXPECT_EQ(3, decision.nMaxChannels);
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclGetAlgoInfo -- rccl_wrap.cc:447-478. Previously had ZERO tests
// (confirmed via grep before this work). Older, simpler size-based
// dispatcher: Hierarchical (AllGather/ReduceScatter) -> Direct
// (AllGather only) -> plain getAlgoInfo fallback. Isolated throughout for
// the same reasons as its callees.
// ===========================================================================

TEST(WrapMicrotestIsolated, GetAlgoInfo_AllGatherHierarchicalDispatchesToHierarchicalAlgoInfo) {
  RUN_ISOLATED_TEST(
      "Wrap_GetAlgoInfo_AllGatherHierarchicalDispatchesToHierarchicalAlgoInfo",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ncclComm* interComm = MakeCommWithArch("gfx950");
        interComm->nRanks = 8;
        interComm->nNodes = 1;
        interComm->p2pnChannels = 66;
        ncclComm* intraComm = MakeCommWithArch("gfx90a");
        intraComm->nRanks = 1;
        intraComm->nNodes = 1;
        ncclComm* comm = MakeZeroedComm();
        comm->nNodes = 8; // rcclUseHierarchicalAllGather needs 8+ nodes
        comm->nRanks = 8;
        comm->hierarchicalCommsInitialized = true;
        comm->hierarchicalInterComm = interComm;
        comm->hierarchicalIntraComm = intraComm;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess, rcclGetAlgoInfo(comm, ncclFuncAllGather, /*count=*/8, ncclFloat32,
                                               /*collNetSupport=*/0, /*nvlsSupport=*/0, /*numPipeOps=*/1, &algo,
                                               &protocol, &maxChannels));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_HIERARCHICAL_ALLGATHER, algo);
        EXPECT_EQ(66, maxChannels); // came from rcclHierarchicalAlgoInfo's inter-comm Direct path
        DeleteCommWithArch(comm);
        DeleteHierarchicalSubComms(interComm, intraComm);
      });
}

// Complementary proof to the AllGather-hierarchical dispatch test above:
// rcclParamHierarchicalReduceScatter's real default is 0, so this arm
// (rccl_wrap.cc:456) had never been reached via rcclGetAlgoInfo at all --
// only rcclSelectReduceScatter's own separate Hierarchical gate had a seam.
TEST(WrapMicrotestIsolated, GetAlgoInfo_ReduceScatterHierarchicalDispatchesToHierarchicalAlgoInfo) {
  RUN_ISOLATED_TEST(
      "Wrap_GetAlgoInfo_ReduceScatterHierarchicalDispatchesToHierarchicalAlgoInfo",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_HIERARCHICAL_REDUCE_SCATTER") == 0 ? int64_t(1) : deft;
        };
        ncclComm* interComm = MakeCommWithArch("gfx90a"); // not gfx950 -> not Direct-eligible, falls to getAlgoInfo
        interComm->nRanks = 8;
        interComm->nNodes = 1;
        ncclComm* intraComm = MakeCommWithArch("gfx90a");
        intraComm->nRanks = 1;
        intraComm->nNodes = 1;
        ncclComm* comm = MakeZeroedComm();
        comm->nNodes = 8; // rcclUseHierarchicalReduceScatter needs 8+ nodes
        comm->nRanks = 8;
        comm->hierarchicalCommsInitialized = true;
        comm->hierarchicalInterComm = interComm;
        comm->hierarchicalIntraComm = intraComm;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess, rcclGetAlgoInfo(comm, ncclFuncReduceScatter, /*count=*/8, ncclFloat32,
                                               /*collNetSupport=*/0, /*nvlsSupport=*/0, /*numPipeOps=*/1, &algo,
                                               &protocol, &maxChannels));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_HIERARCHICAL_REDUCESCATTER, algo);
        DeleteCommWithArch(comm);
        DeleteHierarchicalSubComms(interComm, intraComm);
      });
}

TEST(WrapMicrotestIsolated, GetAlgoInfo_AllGatherDirectChosenWhenEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_GetAlgoInfo_AllGatherDirectChosenWhenEligible",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 1; // auto threshold -> 8MiB; also < 8, so Hierarchical never qualifies first
        comm->nRanks = 8; // rankMultiple == 0
        comm->p2pnChannels = 17;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess, rcclGetAlgoInfo(comm, ncclFuncAllGather, /*count=*/8, ncclFloat32,
                                               /*collNetSupport=*/0, /*nvlsSupport=*/0, /*numPipeOps=*/1, &algo,
                                               &protocol, &maxChannels));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DIRECT_ALLGATHER, algo);
        EXPECT_EQ(NCCL_PROTO_SIMPLE, protocol);
        EXPECT_EQ(17, maxChannels);
        DeleteCommWithArch(comm);
      });
}

// The compound dispatch condition's second conjunct -- coll==ReduceScatter &&
// rcclUseHierarchicalReduceScatter(...) -- had only ever seen its own FALSE
// side via coll!=ReduceScatter (short-circuiting before the call). This
// proves the *other* way to reach false: coll==ReduceScatter but the
// eligibility check itself says no (nNodes < 8), falling through to Direct
// (which ReduceScatter doesn't have) and then the plain fallback.
TEST(WrapMicrotestIsolated, GetAlgoInfo_ReduceScatterFallsBackToPlainGetAlgoInfoWhenNotHierarchical) {
  RUN_ISOLATED_TEST(
      "Wrap_GetAlgoInfo_ReduceScatterFallsBackToPlainGetAlgoInfoWhenNotHierarchical",
      []() {
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->algorithm = NCCL_ALGO_RING;
          task->protocol = NCCL_PROTO_LL128;
          task->nMaxChannels = 5;
          return ncclSuccess;
        });
        ncclComm* comm = MakeZeroedComm();
        comm->nNodes = 1; // < 8: rcclUseHierarchicalReduceScatter's own node-count gate says no
        comm->nRanks = 1;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess, rcclGetAlgoInfo(comm, ncclFuncReduceScatter, /*count=*/8, ncclFloat32,
                                               /*collNetSupport=*/0, /*nvlsSupport=*/0, /*numPipeOps=*/1, &algo,
                                               &protocol, &maxChannels));
        EXPECT_EQ(NCCL_PROTO_LL128, protocol);
        EXPECT_EQ(5, maxChannels);
        DeleteCommWithArch(comm);
      });
}

// Same idea for AllGather's Direct dispatch (line 459): the only prior
// AllGather test either took Hierarchical or Direct; this proves the
// eligibility-says-no path to Direct's own false arm, falling through to the
// plain fallback instead.
TEST(WrapMicrotestIsolated, GetAlgoInfo_AllGatherFallsBackToPlainGetAlgoInfoWhenNotDirect) {
  RUN_ISOLATED_TEST(
      "Wrap_GetAlgoInfo_AllGatherFallsBackToPlainGetAlgoInfoWhenNotDirect",
      []() {
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->algorithm = NCCL_ALGO_RING;
          task->protocol = NCCL_PROTO_LL128;
          task->nMaxChannels = 6;
          return ncclSuccess;
        });
        ncclComm* comm = MakeCommWithArch("gfx90a"); // not gfx950/gfx942 -> rcclUseAllGatherDirect's own arch
                                                      // auto-selection never applies; nNodes=1 also < 8 keeps
                                                      // Hierarchical out of the way
        comm->nNodes = 1;
        comm->nRanks = 1;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess, rcclGetAlgoInfo(comm, ncclFuncAllGather, /*count=*/8, ncclFloat32,
                                               /*collNetSupport=*/0, /*nvlsSupport=*/0, /*numPipeOps=*/1, &algo,
                                               &protocol, &maxChannels));
        EXPECT_EQ(NCCL_PROTO_LL128, protocol);
        EXPECT_EQ(6, maxChannels);
        DeleteCommWithArch(comm);
      });
}

// ncclCommCount is unconditional (NCCLCHECK(ncclCommCount(comm, &nRanks)))
// right at the top of the function -- never made to fail before.
TEST(WrapMicrotestIsolated, GetAlgoInfo_CommCountFailurePropagates) {
  RUN_ISOLATED_TEST(
      "Wrap_GetAlgoInfo_CommCountFailurePropagates",
      []() {
        ScopedHook commCount(g_commCount, [](const ncclComm_t, int*) { return ncclInternalError; });
        ncclComm* comm = MakeZeroedComm();
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclInternalError,
                  rcclGetAlgoInfo(comm, ncclFuncAllReduce, /*count=*/8, ncclFloat32, /*collNetSupport=*/0,
                                  /*nvlsSupport=*/0, /*numPipeOps=*/1, &algo, &protocol, &maxChannels));
        DeleteCommWithArch(comm);
      });
}

// NCCLCHECK's compound guard is `RES != ncclSuccess && RES != ncclInProgress`
// -- ncclInProgress is deliberately NOT treated as a hard failure, so it
// must NOT short-circuit the early return the way a real error does. Every
// prior NCCLCHECK failure-injection test in this file used ncclInternalError,
// never ncclInProgress, so this specific arm of the compound condition (RES
// != ncclInProgress evaluating false) had never been proven.
TEST(WrapMicrotestIsolated, GetAlgoInfo_PlainFallbackGetAlgoInfoInProgressDoesNotShortCircuit) {
  RUN_ISOLATED_TEST(
      "Wrap_GetAlgoInfo_PlainFallbackGetAlgoInfoInProgressDoesNotShortCircuit",
      []() {
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->algorithm = NCCL_ALGO_TREE;
          task->protocol = NCCL_PROTO_LL;
          task->nMaxChannels = 9;
          return ncclInProgress;
        });
        ncclComm* comm = MakeZeroedComm();
        comm->nRanks = 1;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess,
                  rcclGetAlgoInfo(comm, ncclFuncAllReduce, /*count=*/8, ncclFloat32, /*collNetSupport=*/0,
                                  /*nvlsSupport=*/0, /*numPipeOps=*/1, &algo, &protocol, &maxChannels));
        EXPECT_EQ(NCCL_PROTO_LL, protocol);
        EXPECT_EQ(9, maxChannels);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, GetAlgoInfo_FallsBackToPlainGetAlgoInfoForAllReduce) {
  RUN_ISOLATED_TEST(
      "Wrap_GetAlgoInfo_FallsBackToPlainGetAlgoInfoForAllReduce",
      []() {
        // Neither the Hierarchical nor the Direct branch even checks ncclFuncAllReduce
        // (both are gated on AllGather/ReduceScatter specifically), so this reaches the
        // plain fallback regardless of comm setup.
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->algorithm = NCCL_ALGO_PAT;
          task->protocol = NCCL_PROTO_LL;
          task->nMaxChannels = 24;
          return ncclSuccess;
        });
        ncclComm* comm = MakeZeroedComm();
        comm->nRanks = 1;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess, rcclGetAlgoInfo(comm, ncclFuncAllReduce, /*count=*/8, ncclFloat32,
                                               /*collNetSupport=*/0, /*nvlsSupport=*/0, /*numPipeOps=*/1, &algo,
                                               &protocol, &maxChannels));
        EXPECT_EQ(NCCL_ALGO_PAT, algo);
        EXPECT_EQ(NCCL_PROTO_LL, protocol);
        EXPECT_EQ(24, maxChannels);
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclGetCollImplInfo -- rccl_wrap.cc:480-525. Dispatches AllReduce/
// AllGather/ReduceScatter to their respective rcclSelectXxx(query=true);
// everything else falls back to rcclGetAlgoInfo. This section proves the
// DISPATCH itself, not the full branch trees of the callees (already
// covered above/by rcclSelectXxx's own tests) -- one representative
// outcome per callee is enough to prove the right one is called.
// ===========================================================================

// The null-check is a three-disjunct `||`, and each operand has to be able to
// trip it on its own: passing only algo==nullptr short-circuits before the
// other two are ever evaluated, so it proves nothing about them. One call per
// operand, with the other two valid. SCOPED_TRACE names the failing operand,
// which is why this reads as one test rather than three near-identical ones.
TEST(WrapMicrotest, GetCollImplInfo_AnyNullOutParamReturnsInvalidArgument) {
  ncclComm* comm = MakeZeroedComm();
  int algo, protocol, maxChannels;
  const struct {
    const char* nulled;
    int* algo;
    int* protocol;
    int* maxChannels;
  } kCases[] = {
      {"algo", nullptr, &protocol, &maxChannels},
      {"protocol", &algo, nullptr, &maxChannels},
      {"maxChannels", &algo, &protocol, nullptr},
  };
  for (const auto& c : kCases) {
    SCOPED_TRACE(c.nulled);
    EXPECT_EQ(ncclInvalidArgument,
              rcclGetCollImplInfo(comm, ncclFuncAllReduce, 8, ncclFloat32, ncclSum, nullptr, nullptr,
                                  /*graphCapturing=*/0, c.algo, c.protocol, c.maxChannels));
  }
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotestIsolated, GetCollImplInfo_AllReduceDelegatesToSelectAllReduce) {
  RUN_ISOLATED_TEST(
      "Wrap_GetCollImplInfo_AllReduceDelegatesToSelectAllReduce",
      []() {
        ScopedHook symRequested(g_isSymmetricKernelRequested, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                                                  size_t, const void*, void*) { return true; });
        // Must have a real archName, not a zeroed comm: symEligible=true skips
        // rcclSelectAllReduce's CE-2-shot early return, so execution reaches its
        // unconditional IsArchMatch(comm->archName, "gfx1250") a few lines later --
        // confirmed via a real crash (IsArchMatch doesn't null-check) before this fix.
        ncclComm* comm = MakeCommWithArch("gfx90a");
        comm->nRanks = 1;
        comm->nNodes = 1;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess, rcclGetCollImplInfo(comm, ncclFuncAllReduce, 8, ncclFloat32, ncclSum, nullptr,
                                                    nullptr, /*graphCapturing=*/0, &algo, &protocol, &maxChannels));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_SYMMETRIC, algo); // proves rcclSelectAllReduce, not rcclGetAlgoInfo, ran
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, GetCollImplInfo_AllGatherDelegatesToSelectAllGather) {
  RUN_ISOLATED_TEST(
      "Wrap_GetCollImplInfo_AllGatherDelegatesToSelectAllGather",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 1;
        comm->nRanks = 8;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess, rcclGetCollImplInfo(comm, ncclFuncAllGather, 8, ncclFloat32, ncclSum, nullptr,
                                                    nullptr, /*graphCapturing=*/0, &algo, &protocol, &maxChannels));
        // Direct AllGather is rcclSelectAllGather's own answer here (query mode); a
        // dispatch to rcclGetAlgoInfo instead would give the same algo by coincidence
        // for this setup, so this alone doesn't distinguish them -- the real proof is
        // the next test, which picks a scenario only rcclSelectAllGather can produce.
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DIRECT_ALLGATHER, algo);
        DeleteCommWithArch(comm);
      });
}

// A scenario only rcclSelectAllGather's own priority chain reaches (CE
// registered has no equivalent in rcclGetAlgoInfo's older dispatcher at
// all) -- unambiguous proof this goes through rcclSelectAllGather, not
// rcclGetAlgoInfo.
TEST(WrapMicrotestIsolated, GetCollImplInfo_AllGatherReachesCeRegisteredUnlikeGetAlgoInfo) {
  RUN_ISOLATED_TEST(
      "Wrap_GetCollImplInfo_AllGatherReachesCeRegisteredUnlikeGetAlgoInfo",
      []() {
        ScopedHook ceAvailable(
            g_ceAvailable,
            [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) { return true; });
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nRanks = 1;
        comm->nNodes = 1;
        comm->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;  // required by branch #3 as of #11389
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess, rcclGetCollImplInfo(comm, ncclFuncAllGather, 8, ncclFloat32, ncclSum, nullptr,
                                                    nullptr, /*graphCapturing=*/0, &algo, &protocol, &maxChannels));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_CE_REGISTERED, algo);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, GetCollImplInfo_ReduceScatterDelegatesToSelectReduceScatter) {
  RUN_ISOLATED_TEST(
      "Wrap_GetCollImplInfo_ReduceScatterDelegatesToSelectReduceScatter",
      []() {
        ncclComm* comm = MakeCommWithArch("gfx950");
        comm->nNodes = 2;
        comm->nRanks = 1;
        int algo, protocol, maxChannels;
        // recvcount=262144 floats = 1MiB total, within rcclUseReduceScatterDirect's
        // 2-node [128KiB,2MiB] window -- a Direct outcome rcclGetAlgoInfo's older
        // dispatcher (no Direct-ReduceScatter concept at all) could never produce.
        EXPECT_EQ(ncclSuccess, rcclGetCollImplInfo(comm, ncclFuncReduceScatter, 1048576 / 4, ncclFloat32, ncclSum,
                                                    nullptr, nullptr, /*graphCapturing=*/0, &algo, &protocol,
                                                    &maxChannels));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_DIRECT_REDUCESCATTER, algo);
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, GetCollImplInfo_OtherCollFallsBackToGetAlgoInfo) {
  RUN_ISOLATED_TEST(
      "Wrap_GetCollImplInfo_OtherCollFallsBackToGetAlgoInfo",
      []() {
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->algorithm = NCCL_ALGO_TREE;
          task->protocol = NCCL_PROTO_LL;
          task->nMaxChannels = 6;
          return ncclSuccess;
        });
        ncclComm* comm = MakeZeroedComm();
        comm->nRanks = 1;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess, rcclGetCollImplInfo(comm, ncclFuncBroadcast, 8, ncclFloat32, ncclSum, nullptr,
                                                    nullptr, /*graphCapturing=*/0, &algo, &protocol, &maxChannels));
        EXPECT_EQ(NCCL_ALGO_TREE, algo);
        EXPECT_EQ(NCCL_PROTO_LL, protocol);
        EXPECT_EQ(6, maxChannels);
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// rcclSymkQuery / rcclSymKGetInfo -- rccl_wrap.cc:547-584. The deep path
// past the four already-tested early guards (null comm, no symmetric
// support, unsupported collective, unmapped red op) -- now reachable with
// the symk seams in fakes/wrap_fakes.cc. Isolated throughout, same reasoning as
// the existing guard tests (one mutation away from the abort-floor-turned-
// seam surface, though now a controlled seam rather than a crash).
// ===========================================================================

TEST(WrapMicrotestIsolated, SymkQuery_InitOnceFailureReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_SymkQuery_InitOnceFailureReturnsFalse",
      []() {
        ScopedHook initOnce(g_symkInitOnce, [](struct ncclComm*) { return ncclInternalError; });
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        int algo, protocol, maxChannels;
        EXPECT_FALSE(
            rcclSymkQuery(comm, ncclFuncAllReduce, 8, ncclFloat32, ncclSum, &algo, &protocol, &maxChannels));
        DeleteCommWithArch(comm);
      });
}

// Explicit, direct proof that reaching ncclSymkAvailable() and having it
// say "no" returns false -- distinct from every earlier guard test, all of
// which fail before ever reaching this call (their default seam value
// happens to also be false, but for a different reason each time).
TEST(WrapMicrotestIsolated, SymkQuery_NotAvailableReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_SymkQuery_NotAvailableReturnsFalse",
      []() {
        // g_symkAvailable left at its default (false) -- this test's whole point is
        // that reaching this specific call with the default value returns false.
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        int algo, protocol, maxChannels;
        EXPECT_FALSE(
            rcclSymkQuery(comm, ncclFuncAllReduce, 8, ncclFloat32, ncclSum, &algo, &protocol, &maxChannels));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SymkQuery_PickKernelFailureReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_SymkQuery_PickKernelFailureReturnsFalse",
      []() {
        ScopedHook symkAvailable(g_symkAvailable,
                                 [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t) { return true; });
        ScopedHook pickKernel(g_symkPickKernel, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t, size_t,
                                                    int, ncclSymRegType_t, float*, ncclSymkKernelId*, int*, int*,
                                                    bool*) { return ncclInternalError; });
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        int algo, protocol, maxChannels;
        EXPECT_FALSE(
            rcclSymkQuery(comm, ncclFuncAllReduce, 8, ncclFloat32, ncclSum, &algo, &protocol, &maxChannels));
        DeleteCommWithArch(comm);
      });
}

// PickKernel itself succeeds (no error), but reports the "nothing matched"
// sentinel -- distinct from the PickKernel-failure test above, which never
// reaches this specific check at all.
TEST(WrapMicrotestIsolated, SymkQuery_KernelIdCountReturnsFalse) {
  RUN_ISOLATED_TEST(
      "Wrap_SymkQuery_KernelIdCountReturnsFalse",
      []() {
        ScopedHook symkAvailable(g_symkAvailable,
                                 [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t) { return true; });
        ScopedHook pickKernel(g_symkPickKernel, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t, size_t,
                                                    int, ncclSymRegType_t, float* estTimeUs,
                                                    ncclSymkKernelId* kernelId, int* maxChannels, int* nWarps,
                                                    bool* forced) {
          *estTimeUs = 0.0f;
          *kernelId = ncclSymkKernelId_Count; // "nothing matched" -- the default already does this,
                                              // but spelled out here for this test's own clarity
          *maxChannels = 0;
          *nWarps = 0;
          *forced = false;
          return ncclSuccess;
        });
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        int algo, protocol, maxChannels;
        EXPECT_FALSE(
            rcclSymkQuery(comm, ncclFuncAllReduce, 8, ncclFloat32, ncclSum, &algo, &protocol, &maxChannels));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SymkQuery_SuccessWithLLKernelReportsLLProtocol) {
  RUN_ISOLATED_TEST(
      "Wrap_SymkQuery_SuccessWithLLKernelReportsLLProtocol",
      []() {
        ScopedHook symkAvailable(g_symkAvailable,
                                 [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t) { return true; });
        ScopedHook pickKernel(g_symkPickKernel, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t, size_t,
                                                    int, ncclSymRegType_t, float* estTimeUs,
                                                    ncclSymkKernelId* kernelId, int* maxChannels, int* nWarps,
                                                    bool* forced) {
          *estTimeUs = 0.0f;
          *kernelId = ncclSymkKernelId_AllReduce_AGxLL_R;
          *maxChannels = 9;
          *nWarps = 1;
          *forced = false;
          return ncclSuccess;
        });
        ScopedHook kernelIsLL(g_symkKernelIdIsLL, [](int) { return true; });
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        int algo, protocol, maxChannels;
        EXPECT_TRUE(
            rcclSymkQuery(comm, ncclFuncAllReduce, 8, ncclFloat32, ncclSum, &algo, &protocol, &maxChannels));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_SYMMETRIC, algo);
        EXPECT_EQ(NCCL_PROTO_LL, protocol);
        EXPECT_EQ(9, maxChannels);
        DeleteCommWithArch(comm);
      });
}

// Distinguishes the protocol ternary's other arm: same success path, but
// rcclSymkKernelIdIsLL says no -- SIMPLE, not LL.
TEST(WrapMicrotestIsolated, SymkQuery_SuccessWithNonLLKernelReportsSimpleProtocol) {
  RUN_ISOLATED_TEST(
      "Wrap_SymkQuery_SuccessWithNonLLKernelReportsSimpleProtocol",
      []() {
        ScopedHook symkAvailable(g_symkAvailable,
                                 [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t) { return true; });
        ScopedHook pickKernel(g_symkPickKernel, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t, size_t,
                                                    int, ncclSymRegType_t, float* estTimeUs,
                                                    ncclSymkKernelId* kernelId, int* maxChannels, int* nWarps,
                                                    bool* forced) {
          *estTimeUs = 0.0f;
          *kernelId = ncclSymkKernelId_AllReduce_RSxLD_AGxST;
          *maxChannels = 12;
          *nWarps = 1;
          *forced = false;
          return ncclSuccess;
        });
        // g_symkKernelIdIsLL left at its default (false).
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        int algo, protocol, maxChannels;
        EXPECT_TRUE(
            rcclSymkQuery(comm, ncclFuncAllReduce, 8, ncclFloat32, ncclSum, &algo, &protocol, &maxChannels));
        EXPECT_EQ(NCCL_PROTO_SIMPLE, protocol);
        EXPECT_EQ(12, maxChannels);
        DeleteCommWithArch(comm);
      });
}

// rcclSymKGetInfo falls through to rcclGetCollImplInfo when symk doesn't
// apply (comm->symmetricSupport left at its zero-init default -- the
// simplest way to make rcclSymkQuery return false), reporting whatever
// backend rcclSelectAllReduce actually picks instead of failing outright.
TEST(WrapMicrotestIsolated, SymKGetInfo_FallsThroughToGetCollImplInfoWhenSymkNotApplicable) {
  RUN_ISOLATED_TEST(
      "Wrap_SymKGetInfo_FallsThroughToGetCollImplInfoWhenSymkNotApplicable",
      []() {
        // Must have a real archName, not a zeroed comm: with symEligible false and
        // ceAllReduceAllowed false (both defaults here), rcclSelectAllReduce's
        // CE-2-shot early return doesn't fire, so execution reaches its
        // unconditional IsArchMatch(comm->archName, "gfx1250") a few lines later --
        // confirmed via a real crash (IsArchMatch doesn't null-check) before this fix.
        ncclComm* comm = MakeCommWithArch("gfx90a");
        comm->nRanks = 1;
        comm->nNodes = 1;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess,
                  rcclSymKGetInfo(comm, ncclFuncAllReduce, 8, ncclFloat32, ncclSum, &algo, &protocol, &maxChannels));
        // Falls all the way to rcclSelectAllReduce's plain-kernel fallback (every
        // seam at its default), reported via getAlgoInfo's own default (RING/SIMPLE).
        EXPECT_EQ(NCCL_ALGO_RING, algo);
        DeleteCommWithArch(comm);
      });
}

// Complementary proof: when symk DOES apply, rcclSymKGetInfo returns its
// result directly and does NOT fall through to rcclGetCollImplInfo at all
// -- proven by overriding getAlgoInfo (which rcclGetCollImplInfo's fallback
// would eventually reach) to a value that would only appear if the
// fall-through wrongly happened.
TEST(WrapMicrotestIsolated, SymKGetInfo_ReturnsSymkResultWithoutFallingThrough) {
  RUN_ISOLATED_TEST(
      "Wrap_SymKGetInfo_ReturnsSymkResultWithoutFallingThrough",
      []() {
        ScopedHook symkAvailable(g_symkAvailable,
                                 [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t) { return true; });
        ScopedHook pickKernel(g_symkPickKernel, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t, size_t,
                                                    int, ncclSymRegType_t, float* estTimeUs,
                                                    ncclSymkKernelId* kernelId, int* maxChannels, int* nWarps,
                                                    bool* forced) {
          *estTimeUs = 0.0f;
          *kernelId = ncclSymkKernelId_AllReduce_AGxLL_R;
          *maxChannels = 15;
          *nWarps = 1;
          *forced = false;
          return ncclSuccess;
        });
        ScopedHook kernelIsLL(g_symkKernelIdIsLL, [](int) { return true; });
        // If this wrongly gets called (fall-through bug), it would report 999 --
        // a value that never appears if rcclSymKGetInfo correctly short-circuits.
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->nMaxChannels = 999;
          return ncclSuccess;
        });
        ncclComm* comm = MakeZeroedComm();
        comm->symmetricSupport = 1;
        comm->nRanks = 1;
        comm->nNodes = 1;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclSuccess,
                  rcclSymKGetInfo(comm, ncclFuncAllReduce, 8, ncclFloat32, ncclSum, &algo, &protocol, &maxChannels));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_SYMMETRIC, algo);
        EXPECT_EQ(15, maxChannels);
        EXPECT_NE(999, maxChannels);
        DeleteCommWithArch(comm);
      });
}

// ===========================================================================
// NCCLCHECK failure-injection. Every NCCLCHECK(call) site in this file has
// only ever been exercised with `call` returning ncclSuccess -- none of the
// seams it wraps were ever told to fail, so the macro's failure/early-return
// arm (rccl_wrap.cc checks.h:144-152: `if (RES != ncclSuccess && RES !=
// ncclInProgress) { ...; return RES; }`) has zero coverage at each site. The
// underlying seams (g_cudaGetCapturingGraph/g_getSymRegType/g_getAlgoInfo)
// are already controllable, so this is pure failure injection -- no new
// seam upgrades needed. Several of these cascade: a failure inside
// rcclSelectXxx also flips the NCCLCHECK wrapping it in
// rcclGetCollImplInfo, covering both call sites in one test.
// ===========================================================================

TEST(WrapMicrotestIsolated, SelectAllReduce_LiveModeCudaGetCapturingGraphFailurePropagates) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_LiveModeCudaGetCapturingGraphFailurePropagates",
      []() {
        ScopedHook graphProbe(g_cudaGetCapturingGraph,
                               [](struct ncclCudaGraph*, hipStream_t, int) { return ncclInternalError; });
        ncclComm* comm = MakeSelectComm();
        rcclCollDecision decision{};
        EXPECT_EQ(ncclInternalError,
                  rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                      /*stream=*/nullptr, /*query=*/false, /*graphCapturingHint=*/false, &decision));
        DeleteCommWithArch(comm);
      });
}

// ncclGetSymRegType is called unconditionally in both query modes, right
// after the symmetric-window lookup -- forcing it to fail propagates through
// rcclSelectAllReduce's own NCCLCHECK *and* rcclGetCollImplInfo's NCCLCHECK
// wrapping the call to rcclSelectAllReduce, covering both in one test.
TEST(WrapMicrotestIsolated, GetCollImplInfo_AllReduceSymRegTypeFailurePropagates) {
  RUN_ISOLATED_TEST(
      "Wrap_GetCollImplInfo_AllReduceSymRegTypeFailurePropagates",
      []() {
        ScopedHook getSymRegType(g_getSymRegType, [](struct ncclDevrWindow*, struct ncclDevrWindow*,
                                                       ncclSymRegType_t*) { return ncclInternalError; });
        ncclComm* comm = MakeCommWithArch("gfx90a");
        comm->nRanks = 1;
        comm->nNodes = 1;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclInternalError,
                  rcclGetCollImplInfo(comm, ncclFuncAllReduce, 8, ncclFloat32, ncclSum, nullptr, nullptr,
                                      /*graphCapturing=*/0, &algo, &protocol, &maxChannels));
        DeleteCommWithArch(comm);
      });
}

// Same idea for AllGather: ncclGetSymRegType is called unconditionally inside
// rcclSelectAllGather's CE-check block (reached whenever DDA/Hierarchical
// don't short-circuit first), cascading through to rcclGetCollImplInfo's own
// NCCLCHECK for the same reason as the AllReduce case above.
TEST(WrapMicrotestIsolated, GetCollImplInfo_AllGatherSymRegTypeFailurePropagates) {
  RUN_ISOLATED_TEST(
      "Wrap_GetCollImplInfo_AllGatherSymRegTypeFailurePropagates",
      []() {
        ScopedHook getSymRegType(g_getSymRegType, [](struct ncclDevrWindow*, struct ncclDevrWindow*,
                                                       ncclSymRegType_t*) { return ncclInternalError; });
        ncclComm* comm = MakeCommWithArch("gfx90a"); // not DDA/Hierarchical eligible
        comm->nRanks = 1;
        comm->nNodes = 1;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclInternalError,
                  rcclGetCollImplInfo(comm, ncclFuncAllGather, 8, ncclFloat32, ncclSum, nullptr, nullptr,
                                      /*graphCapturing=*/0, &algo, &protocol, &maxChannels));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SelectAllReduce_PlainKernelFallbackGetAlgoInfoFailurePropagates) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllReduce_PlainKernelFallbackGetAlgoInfoFailurePropagates",
      []() {
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl*, int, int, int,
                                              ncclSimInfo_t*) { return ncclInternalError; });
        ncclComm* comm = MakeSelectComm();
        rcclCollDecision decision{};
        EXPECT_EQ(ncclInternalError, rcclSelectAllReduce(comm, nullptr, nullptr, /*count=*/8, ncclFloat32, ncclSum,
                                                          /*stream=*/nullptr, /*query=*/true,
                                                          /*graphCapturingHint=*/false, &decision));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SelectAllGather_PlainKernelFallbackGetAlgoInfoFailurePropagates) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_PlainKernelFallbackGetAlgoInfoFailurePropagates",
      []() {
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl*, int, int, int,
                                              ncclSimInfo_t*) { return ncclInternalError; });
        ncclComm* comm = MakeCommWithArch("gfx90a"); // not DDA/Hierarchical/CE/symmetric/Direct eligible
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclInternalError, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                          /*query=*/true, /*graphCapturingHint=*/false, &decision));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, SelectReduceScatter_PlainKernelFallbackGetAlgoInfoFailurePropagates) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectReduceScatter_PlainKernelFallbackGetAlgoInfoFailurePropagates",
      []() {
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl*, int, int, int,
                                              ncclSimInfo_t*) { return ncclInternalError; });
        ncclComm* comm = MakeCommWithArch("gfx90a");
        comm->nRanks = 1;
        comm->nNodes = 1;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclInternalError, rcclSelectReduceScatter(comm, nullptr, nullptr, /*recvcount=*/8, ncclFloat32,
                                                              ncclSum, /*query=*/true, &decision));
        DeleteCommWithArch(comm);
      });
}

TEST(WrapMicrotestIsolated, GetAlgoInfo_PlainFallbackGetAlgoInfoFailurePropagates) {
  RUN_ISOLATED_TEST(
      "Wrap_GetAlgoInfo_PlainFallbackGetAlgoInfoFailurePropagates",
      []() {
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl*, int, int, int,
                                              ncclSimInfo_t*) { return ncclInternalError; });
        ncclComm* comm = MakeZeroedComm();
        comm->nRanks = 1;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclInternalError,
                  rcclGetAlgoInfo(comm, ncclFuncAllReduce, /*count=*/8, ncclFloat32, /*collNetSupport=*/0,
                                  /*nvlsSupport=*/0, /*numPipeOps=*/1, &algo, &protocol, &maxChannels));
        DeleteCommWithArch(comm);
      });
}

// Inter-phase getAlgoInfo failure returns early, before intra's own
// NCCLCHECK(getAlgoInfo(...)) call is ever reached -- proven by using the
// exact "both not Direct-eligible" setup that (with getAlgo succeeding)
// exercises *both* calls in HierarchicalAlgoInfo_AllGatherInterAboveNodeCapFallsToGetAlgoInfo
// above; forcing failure here can only prove the first of the two fired.
TEST(WrapMicrotestIsolated, HierarchicalAlgoInfo_InterGetAlgoInfoFailureReturnsEarly) {
  RUN_ISOLATED_TEST(
      "Wrap_HierarchicalAlgoInfo_InterGetAlgoInfoFailureReturnsEarly",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl*, int, int, int,
                                              ncclSimInfo_t*) { return ncclInternalError; });
        ncclComm* interComm = MakeCommWithArch("gfx90a"); // not Direct-eligible -> falls to getAlgoInfo
        interComm->nRanks = 17;
        interComm->nNodes = 1;
        ncclComm* intraComm = MakeCommWithArch("gfx90a");
        intraComm->nRanks = 1;
        intraComm->nNodes = 1;
        ncclComm* comm = MakeZeroedComm();
        comm->hierarchicalInterComm = interComm;
        comm->hierarchicalIntraComm = intraComm;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclInternalError,
                  rcclHierarchicalAlgoInfo(comm, ncclFuncAllGather, /*count=*/8, ncclFloat32, &algo, &protocol,
                                           &maxChannels));
        DeleteCommWithArch(comm);
        DeleteHierarchicalSubComms(interComm, intraComm);
      });
}

// Complementary proof: inter phase is Direct-eligible (skips getAlgoInfo
// entirely), so its own NCCLCHECK never even runs; intra phase falls to
// getAlgoInfo and fails there instead -- the other of the two call sites.
TEST(WrapMicrotestIsolated, HierarchicalAlgoInfo_IntraGetAlgoInfoFailurePropagates) {
  RUN_ISOLATED_TEST(
      "Wrap_HierarchicalAlgoInfo_IntraGetAlgoInfoFailurePropagates",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl*, int, int, int,
                                              ncclSimInfo_t*) { return ncclInternalError; });
        ncclComm* interComm = MakeCommWithArch("gfx950"); // Direct-eligible -> never calls getAlgoInfo
        interComm->nRanks = 8;
        interComm->nNodes = 1;
        interComm->p2pnChannels = 11;
        ncclComm* intraComm = MakeCommWithArch("gfx90a"); // not Direct-eligible -> falls to getAlgoInfo, fails
        intraComm->nRanks = 1;
        intraComm->nNodes = 1;
        ncclComm* comm = MakeZeroedComm();
        comm->hierarchicalInterComm = interComm;
        comm->hierarchicalIntraComm = intraComm;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclInternalError,
                  rcclHierarchicalAlgoInfo(comm, ncclFuncAllGather, /*count=*/8, ncclFloat32, &algo, &protocol,
                                           &maxChannels));
        DeleteCommWithArch(comm);
        DeleteHierarchicalSubComms(interComm, intraComm);
      });
}

// Cascades three NCCLCHECK sites in one test: rcclHierarchicalAlgoInfo's own
// inter-phase getAlgoInfo call, rcclSelectReduceScatter's
// NCCLCHECK(rcclHierarchicalAlgoInfo(...)) wrapping it, and
// rcclGetCollImplInfo's NCCLCHECK(rcclSelectReduceScatter(...)) wrapping
// that -- all three fail together from the single root cause.
TEST(WrapMicrotestIsolated, GetCollImplInfo_ReduceScatterHierarchicalGetAlgoInfoFailurePropagates) {
  RUN_ISOLATED_TEST(
      "Wrap_GetCollImplInfo_ReduceScatterHierarchicalGetAlgoInfoFailurePropagates",
      []() {
        g_loadParam = [](const char* env, int64_t deft) {
          return std::strcmp(env, "RCCL_HIERARCHICAL_REDUCE_SCATTER") == 0 ? int64_t(1) : deft;
        };
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl*, int, int, int,
                                              ncclSimInfo_t*) { return ncclInternalError; });
        ncclComm* interComm = MakeCommWithArch("gfx90a"); // not Direct-eligible (RS has no node cap, but gfx90a
                                                           // never qualifies) -> falls to getAlgoInfo, fails
        interComm->nRanks = 16;
        interComm->nNodes = 1;
        ncclComm* intraComm = MakeCommWithArch("gfx90a");
        intraComm->nRanks = 1;
        intraComm->nNodes = 1;
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 16; // rcclUseHierarchicalReduceScatter eligible
        comm->nRanks = 16;
        comm->hierarchicalCommsInitialized = true;
        comm->hierarchicalInterComm = interComm;
        comm->hierarchicalIntraComm = intraComm;
        int algo, protocol, maxChannels;
        EXPECT_EQ(ncclInternalError,
                  rcclGetCollImplInfo(comm, ncclFuncReduceScatter, 8, ncclFloat32, ncclSum, nullptr, nullptr,
                                      /*graphCapturing=*/0, &algo, &protocol, &maxChannels));
        DeleteCommWithArch(comm);
        DeleteHierarchicalSubComms(interComm, intraComm);
      });
}

// ===========================================================================
// rcclOverrideChannels -- two remaining branch gaps not hit by the tests
// above: an unsupported collective (rcclGetTunableIndex's `default`
// arm) and a threshold table left at its zero-init default (which collides
// with CHAN_THRESHOLDS_UNDEFINED, breaking the loop on its first iteration).
// ===========================================================================

TEST(WrapMicrotest, OverrideChannels_UnsupportedCollLeavesNcUntouched) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 4;
  comm->nRanks = 8;
  int nc = -100;
  // ncclFuncSendRecv isn't one of rcclGetTunableIndex's mapped collectives ->
  // RCCL_UNSUPPORTED_TUNABLE.
  EXPECT_EQ(ncclSuccess, rcclOverrideChannels(comm, ncclFuncSendRecv, /*nBytes=*/1024, nc));
  EXPECT_EQ(-100, nc);
  DeleteCommWithArch(comm);
}

TEST(WrapMicrotest, OverrideChannels_UndefinedThresholdBreaksLoop) {
  ncclComm* comm = MakeCommWithArch("gfx942");
  comm->nNodes = 4;
  comm->nRanks = 8;
  // minMaxChannelThresholds left at its zero-init default: 0 ==
  // CHAN_THRESHOLDS_UNDEFINED, so the very first loop iteration breaks.
  int nc = -100;
  EXPECT_EQ(ncclSuccess, rcclOverrideChannels(comm, ncclFuncAllReduce, /*nBytes=*/1024, nc));
  EXPECT_EQ(-100, nc);
  DeleteCommWithArch(comm);
}

// ===========================================================================
// INFO-macro branch sweep. INFO(FLAGS, ...) (debug.h:50-55) expands to a
// 3-term compound condition -- (level >= NCCL_LOG_INFO && (FLAGS &
// ncclDebugMask)) || (level < 0) -- so full branch coverage per call site
// needs: the default-suppressed state (already exercised by every existing
// test), a mask that does NOT match FLAGS at the default level (the "B"
// operand false), and a negative debug level (the "C" operand true,
// short-circuiting the whole condition regardless of A/B). A mask of 0
// guarantees B is false for every FLAGS value in one shot; a level of -1
// guarantees C is true for every call site in one shot -- so two broad
// sweeps (the zero-mask one directly below, the negative-level one further
// down past the rcclSelectAllGather section), each replaying already-proven
// call configurations under a different ScopedDebugLogging context, close
// every in-scope INFO call site's remaining arm(s) at once instead of one bespoke
// test per site. (ENABLE_WARP_SPEED's own INFO call sites are out of scope,
// same as the rest of that cluster; rcclUseAllGatherDirect's AINIC INFO call
// site at rccl_wrap.cc:724 is reachable -- g_useAinic is a seam, driven by
// UseAllGatherDirect_AinicDisablesDirect -- but is not replayed by this
// sweep, so its remaining logging arms stay a documented residual.)
// ===========================================================================

namespace {
// Replays a broad, already-proven cross-section of INFO(...) call sites
// under whatever ScopedDebugLogging context the caller has set up. Shared by
// both sweep tests below so the same configurations aren't duplicated twice.
void ExerciseInfoCallSites() {
  // rcclOverrideChannels: nNodes<2 (216), single-GPU-per-node (221),
  // matched-threshold-in-bounds (246+262), matched-threshold-out-of-bounds
  // (246+267).
  {
    ncclComm* comm = MakeZeroedComm();
    comm->nNodes = 1;
    int nc = -100;
    rcclOverrideChannels(comm, ncclFuncAllReduce, /*nBytes=*/1024, nc);
    DeleteCommWithArch(comm);
  }
  {
    ncclComm* comm = MakeCommWithArch("gfx942");
    comm->nNodes = 4;
    comm->nRanks = 4;
    int nc = -100;
    rcclOverrideChannels(comm, ncclFuncAllReduce, /*nBytes=*/1024, nc);
    DeleteCommWithArch(comm);
  }
  {
    ncclComm* comm = MakeCommWithArch("gfx942");
    comm->nNodes = 4;
    comm->nRanks = 8;
    comm->nChannels = 32;
    comm->config.minCTAs = 1;
    comm->config.maxCTAs = 64;
    comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][0] = 1;
    comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][1] = 2048;
    comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][2] = 8;
    int nc = -100;
    rcclOverrideChannels(comm, ncclFuncAllReduce, /*nBytes=*/16384, nc);
    DeleteCommWithArch(comm);
  }
  {
    ncclComm* comm = MakeCommWithArch("gfx942");
    comm->nNodes = 4;
    comm->nRanks = 8;
    comm->nChannels = 32;
    comm->config.minCTAs = 1;
    comm->config.maxCTAs = 4;
    comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][0] = 1;
    comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][1] = 2048;
    comm->minMaxChannelThresholds[RCCL_AR_TUNABLE][0][2] = 8;
    int nc = -100;
    rcclOverrideChannels(comm, ncclFuncAllReduce, /*nBytes=*/8192, nc);
    DeleteCommWithArch(comm);
  }
  // rcclHierarchicalAlgoInfo's summary line (442).
  {
    SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
    ncclComm* interComm = MakeCommWithArch("gfx950");
    interComm->nRanks = 8;
    interComm->nNodes = 1;
    interComm->p2pnChannels = 11;
    ncclComm* intraComm = MakeCommWithArch("gfx950");
    intraComm->nRanks = 8;
    intraComm->nNodes = 1;
    intraComm->p2pnChannels = 22;
    ncclComm* comm = MakeZeroedComm();
    comm->hierarchicalInterComm = interComm;
    comm->hierarchicalIntraComm = intraComm;
    int algo, protocol, maxChannels;
    rcclHierarchicalAlgoInfo(comm, ncclFuncAllGather, /*count=*/8, ncclFloat32, &algo, &protocol, &maxChannels);
    DeleteCommWithArch(comm);
    DeleteHierarchicalSubComms(interComm, intraComm);
    ClearMicroEnv();
  }
  // rcclUseAllGatherDirect: param-disabled (719), CTA-policy-ZERO (742).
  {
    g_loadParam = [](const char* env, int64_t deft) {
      return std::strcmp(env, "RCCL_DIRECT_ALLGATHER_DISABLE") == 0 ? int64_t(1) : deft;
    };
    SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
    ncclComm* comm = MakeCommWithArch("gfx950");
    comm->nRanks = 8;
    size_t msgSize = 1024;
    rcclUseAllGatherDirect(comm, msgSize);
    DeleteCommWithArch(comm);
    ClearMicroEnv();
  }
  {
    g_loadParam = [](const char*, int64_t deft) { return deft; };
    SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
    ncclComm* comm = MakeCommWithArch("gfx950");
    comm->nNodes = 1;
    comm->nRanks = 8;
    comm->symmetricSupport = 1;
    comm->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
    size_t msgSize = 1024;
    rcclUseAllGatherDirect(comm, msgSize);
    DeleteCommWithArch(comm);
    ClearMicroEnv();
  }
  // rcclUseCeAllReduce: disabled-by-default (774).
  {
    g_loadParam = [](const char* env, int64_t deft) {
      return std::strcmp(env, "RCCL_CE_ALLREDUCE") == 0 ? int64_t(0) : deft;
    };
    ncclComm* comm = MakeCommWithArch("gfx942");
    rcclUseCeAllReduce(comm, /*count=*/8, ncclFloat32, ncclSum, /*acc=*/nullptr);
    DeleteCommWithArch(comm);
  }
  // rcclCeAllReduceGraphLatchTick: latch-set (837), latch-cleared (850).
  {
    ncclComm* comm = MakeZeroedComm();
    rcclCeAllReduceGraphLatchTick(comm, /*ceCapturing=*/true);
    DeleteCommWithArch(comm);
  }
  {
    ncclComm* comm = MakeZeroedComm();
    comm->ceColl.graphModeSeen = true;
    comm->localPersistentRefs = 0;
    rcclCeAllReduceGraphLatchTick(comm, /*ceCapturing=*/false);
    DeleteCommWithArch(comm);
  }
  // rcclSetPxn (1386) / rcclSetP2pNetChunkSize (1412): gfx942 above threshold.
  {
    ncclComm* comm = MakeCommWithArch("gfx942");
    comm->nRanks = 64;
    SetMicroEnvAbsent("NCCL_PXN_DISABLE");
    int rcclPxnDisable = -100;
    rcclSetPxn(comm, rcclPxnDisable);
    ClearMicroEnv();
    SetMicroEnvAbsent("NCCL_P2P_NET_CHUNKSIZE");
    int rcclP2pNetChunkSize = -100;
    rcclSetP2pNetChunkSize(comm, rcclP2pNetChunkSize);
    ClearMicroEnv();
    DeleteCommWithArch(comm);
  }
  // rcclOptThreadBlockSize: rounded-up-to-warp-multiple (1623), clamped-to-
  // max (1627), bumped-up-to-minimum (1630).
  {
    g_loadParam = [](const char* env, int64_t deft) {
      return std::strcmp(env, "RCCL_THREADS_PER_BLOCK") == 0 ? int64_t(200) : deft;
    };
    ncclComm* comm = MakeCommWithArch("gfx942");
    comm->WarpSize = 64;
    ncclTaskColl info{};
    int nThreads = -1;
    rcclOptThreadBlockSize(comm, &info, /*nBytes=*/1024, nThreads);
    DeleteCommWithArch(comm);
  }
  {
    g_loadParam = [](const char* env, int64_t deft) {
      return std::strcmp(env, "RCCL_THREADS_PER_BLOCK") == 0 ? int64_t(1024) : deft;
    };
    ncclComm* comm = MakeCommWithArch("gfx942");
    comm->WarpSize = 64;
    ncclTaskColl info{};
    int nThreads = -1;
    rcclOptThreadBlockSize(comm, &info, /*nBytes=*/1024, nThreads);
    DeleteCommWithArch(comm);
  }
  {
    g_loadParam = [](const char* env, int64_t deft) {
      return std::strcmp(env, "RCCL_THREADS_PER_BLOCK") == 0 ? int64_t(64) : deft;
    };
    ncclComm* comm = MakeCommWithArch("gfx942");
    comm->WarpSize = 64;
    ncclTaskColl info{};
    int nThreads = -1;
    rcclOptThreadBlockSize(comm, &info, /*nBytes=*/1024, nThreads);
    DeleteCommWithArch(comm);
  }
  // commSetUnrollFactor: user-set (1678), gfx950-single-node pre-set default
  // (1709).
  {
    g_loadParam = [](const char* env, int64_t deft) {
      return std::strcmp(env, "RCCL_UNROLL_FACTOR") == 0 ? int64_t(NCCL_UNROLL_2) : deft;
    };
    ncclComm* comm = MakeCommWithArch("gfx942");
    commSetUnrollFactor(comm);
    DeleteCommWithArch(comm);
  }
  {
    g_loadParam = [](const char*, int64_t deft) { return deft; };
    ncclComm* comm = MakeCommWithArch("gfx950");
    comm->nNodes = 1;
    commSetUnrollFactor(comm);
    DeleteCommWithArch(comm);
  }
}
}  // namespace

TEST(WrapMicrotestIsolated, InfoMacroSweep_AllCallSitesSuppressedByZeroMask) {
  RUN_ISOLATED_TEST(
      "Wrap_InfoMacroSweep_AllCallSitesSuppressedByZeroMask",
      []() {
        // mask=0: (FLAGS & 0) is always 0, so the macro's "B" operand is
        // false at every call site regardless of FLAGS category, and level
        // is non-negative so "C" is false too -- the whole condition is
        // false and ncclDebugLog is never invoked.
        RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, /*mask=*/0);
        ExerciseInfoCallSites();
      });
}

// ===========================================================================
// rcclSelectAllGather's OWN inline Hierarchical block (rccl_wrap.cc:1082-
// 1121) is distinct from rcclHierarchicalAlgoInfo (a separate, similarly-
// shaped implementation, reached from rcclGetAlgoInfo and from
// rcclSelectReduceScatter's query-mode hierarchical branch) -- confirmed
// entirely unexercised (llvm-cov show: every line in the `if (query)` body
// reports 0 hits) because the only existing Hierarchical test for this
// function uses query=false (live mode), which skips the block outright.
// ===========================================================================

TEST(WrapMicrotestIsolated, SelectAllGather_HierarchicalQueryModeInterAndIntraDirectEligible) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_HierarchicalQueryModeInterAndIntraDirectEligible",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ncclComm* interComm = MakeCommWithArch("gfx950");
        interComm->nRanks = 8; // <=16 node cap, Direct-eligible
        interComm->nNodes = 1;
        interComm->p2pnChannels = 11;
        ncclComm* intraComm = MakeCommWithArch("gfx950");
        intraComm->nRanks = 8;
        intraComm->nNodes = 1;
        intraComm->p2pnChannels = 22;
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 8; // rcclUseHierarchicalAllGather eligible
        comm->nRanks = 8;
        comm->hierarchicalCommsInitialized = true;
        comm->hierarchicalInterComm = interComm;
        comm->hierarchicalIntraComm = intraComm;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ((int)rcclAddonAlgos_t::RCCL_HIERARCHICAL_ALLGATHER, decision.algo);
        EXPECT_EQ(NCCL_PROTO_SIMPLE, decision.protocol);
        EXPECT_EQ(11, decision.nMaxChannels); // inter's p2pnChannels, not intra's 22 nor a getAlgoInfo value
        DeleteCommWithArch(comm);
        DeleteHierarchicalSubComms(interComm, intraComm);
      });
}

// rcclSelectAllGather's own inline Hierarchical block has its own `nNodes <=
// 16` node cap (a separate copy of the same rule rcclHierarchicalAlgoInfo
// enforces) -- its false side had never fired here: interComm stays
// otherwise Direct-eligible (gfx950, small message), but nRanks=17 trips
// the cap specifically, distinct from Direct-ineligibility.
TEST(WrapMicrotestIsolated, SelectAllGather_HierarchicalQueryModeInterAboveNodeCapFallsToGetAlgoInfo) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_HierarchicalQueryModeInterAboveNodeCapFallsToGetAlgoInfo",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->protocol = NCCL_PROTO_LL128;
          task->nMaxChannels = 42;
          return ncclSuccess;
        });
        ncclComm* interComm = MakeCommWithArch("gfx950");
        interComm->nRanks = 17; // > 16: caps out Direct regardless of otherwise-qualifying setup
        interComm->nNodes = 1;
        interComm->p2pnChannels = 11; // would be wrongly reported if the cap were broken
        ncclComm* intraComm = MakeCommWithArch("gfx90a"); // not Direct-eligible; falls to getAlgoInfo too
        intraComm->nRanks = 1;
        intraComm->nNodes = 1;
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 8;
        comm->nRanks = 8;
        comm->hierarchicalCommsInitialized = true;
        comm->hierarchicalInterComm = interComm;
        comm->hierarchicalIntraComm = intraComm;
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ(NCCL_PROTO_LL128, decision.protocol);
        EXPECT_EQ(42, decision.nMaxChannels);
        DeleteCommWithArch(comm);
        DeleteHierarchicalSubComms(interComm, intraComm);
      });
}

TEST(WrapMicrotestIsolated, SelectAllGather_HierarchicalQueryModeFallsToGetAlgoInfo) {
  RUN_ISOLATED_TEST(
      "Wrap_SelectAllGather_HierarchicalQueryModeFallsToGetAlgoInfo",
      []() {
        SetMicroEnvAbsent("RCCL_DIRECT_ALLGATHER_THRESHOLD");
        ScopedHook getAlgo(g_getAlgoInfo, [](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                              ncclSimInfo_t*) {
          task->protocol = NCCL_PROTO_LL128;
          task->nMaxChannels = 42;
          return ncclSuccess;
        });
        ncclComm* interComm = MakeCommWithArch("gfx90a"); // not Direct-eligible -> falls to getAlgoInfo
        interComm->nRanks = 8;
        interComm->nNodes = 1;
        ncclComm* intraComm = MakeCommWithArch("gfx90a"); // not Direct-eligible -> falls to getAlgoInfo
        intraComm->nRanks = 1;
        intraComm->nNodes = 1;
        ncclComm* comm = MakeCommWithArch("gfx942");
        comm->nNodes = 8;
        comm->nRanks = 8;
        comm->hierarchicalCommsInitialized = true;
        comm->hierarchicalInterComm = interComm;
        comm->hierarchicalIntraComm = intraComm;
        RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_ALL); // exercises the summary INFO too
        rcclCollDecision decision{};
        EXPECT_EQ(ncclSuccess, rcclSelectAllGather(comm, nullptr, nullptr, /*sendcount=*/8, ncclFloat32,
                                                    /*query=*/true, /*graphCapturingHint=*/false, &decision));
        EXPECT_EQ(NCCL_PROTO_LL128, decision.protocol);
        EXPECT_EQ(42, decision.nMaxChannels);
        DeleteCommWithArch(comm);
        DeleteHierarchicalSubComms(interComm, intraComm);
      });
}

TEST(WrapMicrotestIsolated, InfoMacroSweep_AllCallSitesForcedOpenByNegativeDebugLevel) {
  RUN_ISOLATED_TEST(
      "Wrap_InfoMacroSweep_AllCallSitesForcedOpenByNegativeDebugLevel",
      []() {
        // level=-1: "C" (level < 0) is unconditionally true, short-circuiting
        // the whole condition to true regardless of A/B -- ncclDebugLog
        // fires at every call site reached below.
        RcclUnitTesting::ScopedDebugLogging debugLogging(/*level=*/-1, /*mask=*/0);
        ExerciseInfoCallSites();
      });
}
