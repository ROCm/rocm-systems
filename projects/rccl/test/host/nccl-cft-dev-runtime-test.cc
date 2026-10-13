/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/cft_dev_runtime.cc: the CFT (Compute Fabric
// Transport) logical-endpoint support-probe, size-computation, and
// bind/unbind helpers, plus the three public LE-info query functions. This
// build's CUDA_VERSION is below 13030, so the real CUDA logical-endpoint
// driver calls are compiled out everywhere in this file (no ROCm
// equivalent exists); what remains compiled is exercised here: the
// zero-everything-and-succeed body of ncclGpuCftSupport, the host-side
// branch logic of computeCftSize/computeCftMcSize, the condition-evaluation
// (but now-empty) bodies of symBindTeamLe/symUnbindTeamLe, the
// trivial-failure stand-ins for symTeamObtainUcLe/symTeamObtainMcLe, and the
// three unconditional ncclGet*DeviceLeInfo query functions.
//
// core.cc is compiled for real alongside cft_dev_runtime.cc (not faked):
// ncclGetMultimemDeviceLeInfo/ncclGetCftDeviceLeInfo/ncclGetPeerDeviceLeInfo
// call ncclTeamCftMultimem/ncclTeamCft, and faking those would just be
// re-deriving core.cc's own already-tested logic (AICOMRCCL-2825 PR3) rather
// than reusing it.

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

#include "fakes/cft_dev_runtime_fakes.h"
#include "fakes/dev_runtime_fakes.h"

#include CORE_CC_PATH
#include CFT_DEV_RUNTIME_CC_PATH

namespace {

// Heap-allocates a ncclDevrTeam sized and populated for its flexible-array
// worldRankList, same idiom dev_runtime.cc's own symTeamObtain uses. An
// automatic (stack) ncclDevrTeam's worldRankList has zero elements; a fake
// that hands one to code indexing it at team.nRanks-1 is out of bounds.
struct DevrTeamDeleter {
  void operator()(ncclDevrTeam* t) const { std::free(t); }
};
using DevrTeamPtr = std::unique_ptr<ncclDevrTeam, DevrTeamDeleter>;

DevrTeamPtr MakeDevrTeam(ncclTeam_t team, ncclCftLeId mcLeId0, ncclCftLeId mcLeId1) {
  auto* t = static_cast<ncclDevrTeam*>(std::malloc(sizeof(ncclDevrTeam) + team.nRanks * sizeof(int)));
  t->team = team;
  t->ucLeId[0] = t->ucLeId[1] = NCCL_LE_ID_INVALID;
  t->mcLeId[0] = mcLeId0;
  t->mcLeId[1] = mcLeId1;
  for (int i = 0; i < team.nRanks; i++) t->worldRankList[i] = i;
  return DevrTeamPtr(t);
}

class NcclCftDevRuntimeMicrotest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>(); // value-initialised: POD members zeroed
    comm = commStorage.get();
    // Load-bearing only for computeCftSize/computeCftMcSize, which check
    // devrState.bigSize themselves: ncclDevrInitOnce in this binary is the
    // always-succeeds fake seam at fakes/dev_runtime_fakes.cc, which never
    // reads devrState, so bigSize is inert on the ncclTeamCft/
    // ncclTeamCftMultimem path.
    comm->devrState.bigSize = 1;
  }

  void TearDown() override {
    ResetDevRuntimeFakes();
    ResetCftDevRuntimeFakes();
  }
};

} // namespace

// ---------------------------------------------------------------------------
// ncclGpuCftSupport: CUDA_VERSION < 13030 on this build, so the entire
// capability-probe body is compiled out; only the zero-and-succeed prologue
// remains.
// ---------------------------------------------------------------------------

TEST_F(NcclCftDevRuntimeMicrotest, GpuCftSupport_ZeroesAllThreeOutputsAndSucceeds) {
  int gpuCftSupport = 99;
  bool gpuCftMulticastSupport = true, gpuCftCountedSupport = true;
  EXPECT_EQ(ncclGpuCftSupport(comm, &gpuCftSupport, &gpuCftMulticastSupport, &gpuCftCountedSupport), ncclSuccess);
  EXPECT_EQ(gpuCftSupport, 0);
  EXPECT_FALSE(gpuCftMulticastSupport);
  EXPECT_FALSE(gpuCftCountedSupport);
}

// ---------------------------------------------------------------------------
// computeCftSize / computeCftMcSize
// ---------------------------------------------------------------------------

TEST_F(NcclCftDevRuntimeMicrotest, ComputeCftSize_AlreadySized_ReturnsCftSizeWithoutCallingComputeLsaSize) {
  comm->devrState.bigSize = 7;
  comm->devrState.cftSize = 42;
  bool called = false;
  g_computeLsaSize = [&](ncclComm*) { called = true; return 1; };
  EXPECT_EQ(computeCftSize(comm), 42);
  EXPECT_FALSE(called);
}

TEST_F(NcclCftDevRuntimeMicrotest, ComputeCftSize_NotSizedNoGpuCftSupport_ReturnsOneWithoutCallingComputeLsaSize) {
  comm->devrState.bigSize = 0;
  comm->gpuCftSupport = 0;
  bool called = false;
  g_computeLsaSize = [&](ncclComm*) { called = true; return 99; };
  EXPECT_EQ(computeCftSize(comm), 1);
  EXPECT_FALSE(called);
}

TEST_F(NcclCftDevRuntimeMicrotest, ComputeCftSize_NotSizedWithGpuCftSupport_DelegatesToComputeLsaSize) {
  comm->devrState.bigSize = 0;
  comm->gpuCftSupport = 13030;
  g_computeLsaSize = [](ncclComm*) { return 8; };
  EXPECT_EQ(computeCftSize(comm), 8);
}

TEST_F(NcclCftDevRuntimeMicrotest, ComputeCftSizeAndMcSize_GpuCftSupportJustBelow13030_ReturnsOne) {
  // Every other probe uses 0 or exactly 13030, far from the `>= 13030` edge at
  // cft_dev_runtime.cc:54 and 61; 13029 is the first value it must reject.
  comm->devrState.bigSize = 0;
  comm->gpuCftSupport = 13029;
  bool called = false;
  g_computeLsaSize = [&](ncclComm*) { called = true; return 99; };
  EXPECT_EQ(computeCftSize(comm), 1);
  EXPECT_EQ(computeCftMcSize(comm), 1);
  EXPECT_FALSE(called);
}

TEST_F(NcclCftDevRuntimeMicrotest, ComputeCftMcSize_AlreadySized_ReturnsCftMcSizeWithoutCallingComputeLsaSize) {
  comm->devrState.bigSize = 7;
  comm->devrState.cftMcSize = 24;
  bool called = false;
  g_computeLsaSize = [&](ncclComm*) { called = true; return 1; };
  EXPECT_EQ(computeCftMcSize(comm), 24);
  EXPECT_FALSE(called);
}

TEST_F(NcclCftDevRuntimeMicrotest, ComputeCftMcSize_NotSizedNoGpuCftSupport_ReturnsOneWithoutCallingComputeLsaSize) {
  comm->devrState.bigSize = 0;
  comm->gpuCftSupport = 0;
  bool called = false;
  g_computeLsaSize = [&](ncclComm*) { called = true; return 99; };
  EXPECT_EQ(computeCftMcSize(comm), 1);
  EXPECT_FALSE(called);
}

TEST_F(NcclCftDevRuntimeMicrotest, ComputeCftMcSize_NotSizedWithGpuCftSupport_DelegatesToComputeLsaSize) {
  comm->devrState.bigSize = 0;
  comm->gpuCftSupport = 13030;
  g_computeLsaSize = [](ncclComm*) { return 6; };
  EXPECT_EQ(computeCftMcSize(comm), 6);
}

// ---------------------------------------------------------------------------
// symBindTeamLe / symUnbindTeamLe: on this build the CUDA bind/unbind calls
// inside the condition are compiled out, so the only externally observable
// contract left is "always succeeds" and "always consults
// ncclDevrWinRegEnabled with this memory's winFlags and ncclDevrRegisterCft".
// ---------------------------------------------------------------------------

TEST_F(NcclCftDevRuntimeMicrotest, SymBindTeamLe_AlwaysSucceedsAndConsultsWinRegEnabledWithCftCapability) {
  auto mem = std::make_unique<ncclDevrMemory>();
  mem->winFlags = 0x5a;
  int sawWinFlags = -1;
  ncclDevrRegisterCapability sawCapability = ncclDevrRegisterAll;
  g_devrWinRegEnabled = [&](int winFlags, ncclDevrRegisterCapability capability) {
    sawWinFlags = winFlags;
    sawCapability = capability;
    return true;
  };
  EXPECT_EQ(symBindTeamLe(comm, mem.get(), /*le=*/7), ncclSuccess);
  EXPECT_EQ(sawWinFlags, 0x5a);
  EXPECT_EQ(sawCapability, ncclDevrRegisterCft);
}

TEST_F(NcclCftDevRuntimeMicrotest, SymBindTeamLe_WinRegDisabled_StillSucceeds) {
  auto mem = std::make_unique<ncclDevrMemory>();
  g_devrWinRegEnabled = [](int, ncclDevrRegisterCapability) { return false; };
  EXPECT_EQ(symBindTeamLe(comm, mem.get(), /*le=*/7), ncclSuccess);
}

TEST_F(NcclCftDevRuntimeMicrotest, SymBindTeamLe_InvalidLe_StillSucceeds) {
  auto mem = std::make_unique<ncclDevrMemory>();
  g_devrWinRegEnabled = [](int, ncclDevrRegisterCapability) { return true; };
  EXPECT_EQ(symBindTeamLe(comm, mem.get(), /*le=*/NCCL_LE_ID_INVALID), ncclSuccess);
}

TEST_F(NcclCftDevRuntimeMicrotest, SymBindTeamLe_GlobalHasSysmemSegment_StillSucceeds) {
  auto mem = std::make_unique<ncclDevrMemory>();
  mem->globalHasSysmemSegment = true;
  g_devrWinRegEnabled = [](int, ncclDevrRegisterCapability) { return true; };
  EXPECT_EQ(symBindTeamLe(comm, mem.get(), /*le=*/7), ncclSuccess);
}

TEST_F(NcclCftDevRuntimeMicrotest, SymUnbindTeamLe_AlwaysSucceedsAndConsultsWinRegEnabledWithCftCapability) {
  auto mem = std::make_unique<ncclDevrMemory>();
  mem->winFlags = 0x3c;
  int sawWinFlags = -1;
  ncclDevrRegisterCapability sawCapability = ncclDevrRegisterAll;
  g_devrWinRegEnabled = [&](int winFlags, ncclDevrRegisterCapability capability) {
    sawWinFlags = winFlags;
    sawCapability = capability;
    return true;
  };
  EXPECT_EQ(symUnbindTeamLe(comm, mem.get(), /*le=*/3), ncclSuccess);
  EXPECT_EQ(sawWinFlags, 0x3c);
  EXPECT_EQ(sawCapability, ncclDevrRegisterCft);
}

TEST_F(NcclCftDevRuntimeMicrotest, SymUnbindTeamLe_WinRegDisabled_StillSucceeds) {
  auto mem = std::make_unique<ncclDevrMemory>();
  g_devrWinRegEnabled = [](int, ncclDevrRegisterCapability) { return false; };
  EXPECT_EQ(symUnbindTeamLe(comm, mem.get(), /*le=*/3), ncclSuccess);
}

TEST_F(NcclCftDevRuntimeMicrotest, SymUnbindTeamLe_InvalidLe_StillSucceeds) {
  auto mem = std::make_unique<ncclDevrMemory>();
  g_devrWinRegEnabled = [](int, ncclDevrRegisterCapability) { return true; };
  EXPECT_EQ(symUnbindTeamLe(comm, mem.get(), /*le=*/NCCL_LE_ID_INVALID), ncclSuccess);
}

TEST_F(NcclCftDevRuntimeMicrotest, SymUnbindTeamLe_GlobalHasSysmemSegment_StillSucceeds) {
  auto mem = std::make_unique<ncclDevrMemory>();
  mem->globalHasSysmemSegment = true;
  g_devrWinRegEnabled = [](int, ncclDevrRegisterCapability) { return true; };
  EXPECT_EQ(symUnbindTeamLe(comm, mem.get(), /*le=*/3), ncclSuccess);
}

// ---------------------------------------------------------------------------
// symTeamObtainUcLe / symTeamObtainMcLe: this build compiles the
// CUDA_VERSION < 13030 stand-ins, which ignore every argument and fail.
// ---------------------------------------------------------------------------

TEST_F(NcclCftDevRuntimeMicrotest, SymTeamObtainUcLe_UnsupportedOnThisPlatform_ReturnsInvalidUsage) {
  bool needBarrier = false;
  EXPECT_EQ(symTeamObtainUcLe(comm, /*t=*/nullptr, /*devr=*/nullptr, &needBarrier, /*counted=*/false),
            ncclInvalidUsage);
}

TEST_F(NcclCftDevRuntimeMicrotest, SymTeamObtainMcLe_UnsupportedOnThisPlatform_ReturnsInvalidUsage) {
  bool needBarrier = false;
  EXPECT_EQ(symTeamObtainMcLe(comm, /*t=*/nullptr, /*devr=*/nullptr, &needBarrier, /*counted=*/true),
            ncclInvalidUsage);
}

// Shared window/comm-resolution setup for all three ncclGet*DeviceLeInfo
// query functions; BigOffset() keeps each derived suite's winHost.bigOffset
// distinct, same reasoning as the handle.* field checks elsewhere in this
// file: a test mixing up which window it is reading would get a visibly
// wrong offset back instead of an accidental match.
class NcclGetDeviceLeInfoTestBase : public NcclCftDevRuntimeMicrotest {
protected:
  ncclDevrWindow winHost{};
  int dummyWindowTarget = 0;
  ncclWindow_t window = reinterpret_cast<ncclWindow_t>(&dummyWindowTarget);
  // What the ncclDevrWinRegEnabled seam was last asked, and what it answers.
  int sawWinFlags = -1;
  ncclDevrRegisterCapability sawCapability = ncclDevrRegisterAll;
  bool winRegEnabled = true;

  virtual size_t BigOffset() const = 0;
  // Calls this suite's query function with its own extra arguments held fixed,
  // so the shared prologue checks below run against all three functions.
  virtual ncclResult_t Invoke(ncclWindow_t w, ncclCftLeId* leId, size_t* leOffset) = 0;

  void SetUp() override {
    NcclCftDevRuntimeMicrotest::SetUp();
    comm->gpuCftSupport = 13030;
    // Non-degenerate flat CFT team: at cftSize 0 the peer query's range check rejects every
    // peer first, so the shared prologue checks below would never reach the check they name.
    comm->devrState.cftSize = 4;
    winHost.bigOffset = BigOffset();
    g_foundComm = comm;
    g_foundWinHost = &winHost;
    UseFoundCommAndWindow();
    g_devrWinRegEnabled = [this](int winFlags, ncclDevrRegisterCapability capability) {
      sawWinFlags = winFlags;
      sawCapability = capability;
      return winRegEnabled;
    };
  }

  // The prologue below is identical across the three query functions
  // (cft_dev_runtime.cc:295-309 and its two twins).
  void ExpectNullArgsRejectedByPtrCheck() {
    ncclCftLeId leId;
    size_t leOffset;
    EXPECT_EQ(Invoke(nullptr, &leId, &leOffset), ncclInvalidArgument);
    EXPECT_EQ(Invoke(window, nullptr, &leOffset), ncclInvalidArgument);
    EXPECT_EQ(Invoke(window, &leId, nullptr), ncclInvalidArgument);
  }

  void ExpectWindowNotCftRegisteredReturnsInvalidUsage() {
    // Also pins what the registration check asks: this window's own flags,
    // against the CFT capability rather than e.g. ncclDevrRegisterAll.
    winHost.winFlags = 0x5a;
    winRegEnabled = false;
    ncclCftLeId leId;
    size_t leOffset;
    EXPECT_EQ(Invoke(window, &leId, &leOffset), ncclInvalidUsage);
    EXPECT_EQ(sawWinFlags, 0x5a);
    EXPECT_EQ(sawCapability, ncclDevrRegisterCft);
  }

  void ExpectNoGpuCftSupportReturnsInvalidArgument() {
    comm->gpuCftSupport = 0;
    ncclCftLeId leId;
    size_t leOffset;
    EXPECT_EQ(Invoke(window, &leId, &leOffset), ncclInvalidArgument);
  }

  void ExpectFindCommAndHostWindowFailurePropagates() {
    g_findCommAndHostWindowFromDeviceWindow = [](ncclWindow_t, ncclComm_t*, ncclDevrWindow**) {
      return ncclSystemError;
    };
    ncclCftLeId leId;
    size_t leOffset;
    EXPECT_EQ(Invoke(window, &leId, &leOffset), ncclSystemError);
  }
};

// ---------------------------------------------------------------------------
// ncclGetMultimemDeviceLeInfo
// ---------------------------------------------------------------------------

class NcclGetMultimemDeviceLeInfoTest : public NcclGetDeviceLeInfoTestBase {
protected:
  size_t BigOffset() const override { return 0x1000; }
  ncclResult_t Invoke(ncclWindow_t w, ncclCftLeId* leId, size_t* leOffset) override {
    return ncclGetMultimemDeviceLeInfo(w, 0, leId, leOffset);
  }
};

TEST_F(NcclGetMultimemDeviceLeInfoTest, NullArgs_RejectedByPtrCheckBeforeAnyLookup) {
  ExpectNullArgsRejectedByPtrCheck();
}
TEST_F(NcclGetMultimemDeviceLeInfoTest, WindowNotCftRegistered_ReturnsInvalidUsage) {
  ExpectWindowNotCftRegisteredReturnsInvalidUsage();
}
TEST_F(NcclGetMultimemDeviceLeInfoTest, CommHasNoGpuCftSupport_ReturnsInvalidArgument) {
  ExpectNoGpuCftSupportReturnsInvalidArgument();
}
TEST_F(NcclGetMultimemDeviceLeInfoTest, FindCommAndHostWindowFails_PropagatesErrorViaNcclcheck) {
  ExpectFindCommAndHostWindowFailurePropagates();
}

TEST_F(NcclGetMultimemDeviceLeInfoTest, SymTeamObtainFails_PropagatesErrorViaNcclcheck) {
  g_symTeamObtain = [](ncclComm*, ncclTeam, bool, bool, bool, bool, ncclDevrTeam**, bool*) {
    return ncclInvalidUsage;
  };
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetMultimemDeviceLeInfo(window, 0, &leId, &leOffset), ncclInvalidUsage);
}

// NOTE: the NCCLCHECK(bootstrapIntraNodeBarrier(...)) propagation branch (needBarrier
// path) is not independently exercised here: fakes/bootstrap_stubs.cc's
// bootstrapIntraNodeBarrier is a parameterless-result `{ return ncclSuccess; }` stub
// shared by every other suite that links it, with no failure hook. Giving it one is out
// of scope for this file's tests.

TEST_F(NcclGetMultimemDeviceLeInfoTest, SuccessPath_ReadsMcLeIdByCountedBitAndOffsetsByBigOffsetPlusOffset) {
  comm->devrState.cftMcSize = 4;
  comm->devrState.cftMcSelf = 1;
  // NCCL_WIN_COLL_SYMMETRIC is ORed in alongside NCCL_WIN_CFT_COUNTED so this
  // exercises `winFlags & NCCL_WIN_CFT_COUNTED` as a real bit test: every
  // other winFlags value elsewhere in this file is exactly
  // NCCL_WIN_CFT_COUNTED or exactly 0, which `winFlags == NCCL_WIN_CFT_COUNTED`
  // would also satisfy.
  winHost.winFlags = NCCL_WIN_CFT_COUNTED | NCCL_WIN_COLL_SYMMETRIC;

  DevrTeamPtr team = MakeDevrTeam(ncclTeam_t{4, 1, 1}, /*mcLeId0=*/111, /*mcLeId1=*/222); // counted==true selects mcLeId1
  ncclDevrTeam* teamPtr = team.get();
  g_symTeamObtain = [teamPtr](ncclComm*, ncclTeam team, bool multimem, bool counted, bool wantsLeUc,
                             bool wantsLeMc, ncclDevrTeam** outTeam, bool* needBarrier) {
    EXPECT_FALSE(multimem);
    EXPECT_TRUE(counted); // from winHost.winFlags & NCCL_WIN_CFT_COUNTED
    EXPECT_FALSE(wantsLeUc);
    EXPECT_TRUE(wantsLeMc);
    EXPECT_EQ(team.nRanks, 4);
    EXPECT_EQ(team.rank, 1);
    *outTeam = teamPtr;
    *needBarrier = true;
    return ncclSuccess;
  };

  ncclCftLeId leId = 0;
  size_t leOffset = 0;
  ASSERT_EQ(ncclGetMultimemDeviceLeInfo(window, /*offset=*/0x20, &leId, &leOffset), ncclSuccess);
  EXPECT_EQ(leId, 222u);
  EXPECT_EQ(leOffset, winHost.bigOffset + 0x20);
}

TEST_F(NcclGetMultimemDeviceLeInfoTest, UncountedWindow_ReadsMcLeIdZeroSlot) {
  comm->devrState.cftMcSize = 4;
  comm->devrState.cftMcSelf = 0;
  winHost.winFlags = 0; // not counted

  DevrTeamPtr team = MakeDevrTeam(ncclTeam_t{4, 0, 1}, /*mcLeId0=*/55, /*mcLeId1=*/66);
  ncclDevrTeam* teamPtr = team.get();
  g_symTeamObtain = [teamPtr](ncclComm*, ncclTeam, bool, bool counted, bool, bool, ncclDevrTeam** outTeam,
                             bool* needBarrier) {
    EXPECT_FALSE(counted);
    *outTeam = teamPtr;
    *needBarrier = false; // no barrier requested this time
    return ncclSuccess;
  };

  ncclCftLeId leId = 0;
  size_t leOffset = 0;
  ASSERT_EQ(ncclGetMultimemDeviceLeInfo(window, /*offset=*/0, &leId, &leOffset), ncclSuccess);
  EXPECT_EQ(leId, 55u);
}

TEST_F(NcclGetMultimemDeviceLeInfoTest, SymmetricOnlyWindow_IsUncountedAndReadsMcLeIdZeroSlot) {
  // NCCL_WIN_COLL_SYMMETRIC without NCCL_WIN_CFT_COUNTED: a counted-bit mask
  // that tested the companion bit instead would read the counted slot here.
  comm->devrState.cftMcSize = 4;
  comm->devrState.cftMcSelf = 0;
  winHost.winFlags = NCCL_WIN_COLL_SYMMETRIC;

  DevrTeamPtr team = MakeDevrTeam(ncclTeam_t{4, 0, 1}, /*mcLeId0=*/77, /*mcLeId1=*/88);
  ncclDevrTeam* teamPtr = team.get();
  g_symTeamObtain = [teamPtr](ncclComm*, ncclTeam, bool, bool counted, bool, bool, ncclDevrTeam** outTeam,
                             bool* needBarrier) {
    EXPECT_FALSE(counted);
    *outTeam = teamPtr;
    *needBarrier = false;
    return ncclSuccess;
  };

  ncclCftLeId leId = 0;
  size_t leOffset = 0;
  ASSERT_EQ(ncclGetMultimemDeviceLeInfo(window, /*offset=*/0, &leId, &leOffset), ncclSuccess);
  EXPECT_EQ(leId, 77u);
}

// ---------------------------------------------------------------------------
// ncclGetCftDeviceLeInfo
// ---------------------------------------------------------------------------

class NcclGetCftDeviceLeInfoTest : public NcclGetDeviceLeInfoTestBase {
protected:
  size_t BigOffset() const override { return 0x2000; }
  ncclResult_t Invoke(ncclWindow_t w, ncclCftLeId* leId, size_t* leOffset) override {
    return ncclGetCftDeviceLeInfo(w, 0, /*peerCft=*/0, ncclTeam_t{4, 0, 1}, leId, leOffset);
  }
};

TEST_F(NcclGetCftDeviceLeInfoTest, NullArgs_RejectedByPtrCheck) {
  ExpectNullArgsRejectedByPtrCheck();
}
TEST_F(NcclGetCftDeviceLeInfoTest, WindowNotCftRegistered_ReturnsInvalidUsage) {
  ExpectWindowNotCftRegisteredReturnsInvalidUsage();
}
TEST_F(NcclGetCftDeviceLeInfoTest, CommHasNoGpuCftSupport_ReturnsInvalidArgument) {
  ExpectNoGpuCftSupportReturnsInvalidArgument();
}
TEST_F(NcclGetCftDeviceLeInfoTest, FindCommAndHostWindowFails_PropagatesErrorViaNcclcheck) {
  ExpectFindCommAndHostWindowFailurePropagates();
}

TEST_F(NcclGetCftDeviceLeInfoTest, BaseLeIdNotYetCreated_ReturnsInvalidUsage) {
  comm->devrState.le[0].baseId = NCCL_LE_ID_INVALID;
  ncclTeam_t cftTeam{4, 0, 1};
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetCftDeviceLeInfo(window, 0, 0, cftTeam, &leId, &leOffset), ncclInvalidUsage);
}

TEST_F(NcclGetCftDeviceLeInfoTest, SuccessPath_AddsFlatTeamRankAndPeerOffsetToBaseLeId) {
  // ncclTeamCft(comm, FLAT) reads straight from devrState.cftSize/cftSelf; it never
  // consults devrState.bigSize itself (that's load-bearing only for computeCftSize/
  // computeCftMcSize -- see NcclCftDevRuntimeMicrotest::SetUp's comment).
  comm->devrState.cftSize = 8;
  comm->devrState.cftSelf = 3;
  comm->devrState.le[0].baseId = 1000;
  winHost.winFlags = 0; // uncounted -> le[0]

  ncclTeam_t cftTeam{4, /*rank=*/1, /*stride=*/2};
  ncclCftLeId leId = 0;
  size_t leOffset = 0;
  // *leId = baseLeId + flatTeam.rank + (peerCft - cftTeam.rank) * cftTeam.stride
  //       = 1000 + 3 + (5 - 1) * 2 = 1011
  ASSERT_EQ(ncclGetCftDeviceLeInfo(window, /*offset=*/0x10, /*peerCft=*/5, cftTeam, &leId, &leOffset), ncclSuccess);
  EXPECT_EQ(leId, 1011u);
  EXPECT_EQ(leOffset, winHost.bigOffset + 0x10);
}

TEST_F(NcclGetCftDeviceLeInfoTest, CountedWindow_UsesCountedBaseLeIdSlot) {
  comm->devrState.cftSize = 4;
  comm->devrState.cftSelf = 0;
  comm->devrState.le[0].baseId = NCCL_LE_ID_INVALID; // uncounted slot deliberately unset
  comm->devrState.le[1].baseId = 500;                // counted slot is what must be read
  winHost.winFlags = NCCL_WIN_CFT_COUNTED;

  ncclTeam_t cftTeam{4, 0, 1};
  ncclCftLeId leId = 0;
  size_t leOffset = 0;
  ASSERT_EQ(ncclGetCftDeviceLeInfo(window, 0, /*peerCft=*/0, cftTeam, &leId, &leOffset), ncclSuccess);
  EXPECT_EQ(leId, 500u); // 500 + flatTeam.rank(0) + (0 - 0) * 1
}

// ---------------------------------------------------------------------------
// ncclGetPeerDeviceLeInfo
// ---------------------------------------------------------------------------

class NcclGetPeerDeviceLeInfoTest : public NcclGetDeviceLeInfoTestBase {
protected:
  size_t BigOffset() const override { return 0x3000; }
  ncclResult_t Invoke(ncclWindow_t w, ncclCftLeId* leId, size_t* leOffset) override {
    return ncclGetPeerDeviceLeInfo(w, 0, /*peerWorld=*/0, leId, leOffset);
  }
};

TEST_F(NcclGetPeerDeviceLeInfoTest, NullArgs_RejectedByPtrCheck) {
  ExpectNullArgsRejectedByPtrCheck();
}
TEST_F(NcclGetPeerDeviceLeInfoTest, WindowNotCftRegistered_ReturnsInvalidUsage) {
  ExpectWindowNotCftRegisteredReturnsInvalidUsage();
}
TEST_F(NcclGetPeerDeviceLeInfoTest, CommHasNoGpuCftSupport_ReturnsInvalidArgument) {
  ExpectNoGpuCftSupportReturnsInvalidArgument();
}
TEST_F(NcclGetPeerDeviceLeInfoTest, FindCommAndHostWindowFails_PropagatesErrorViaNcclcheck) {
  ExpectFindCommAndHostWindowFailurePropagates();
}

TEST_F(NcclGetPeerDeviceLeInfoTest, PeerOutsideFlatTeamBoundaries_ReturnsInvalidArgument) {
  // flatStart = comm->rank - flatTeam.rank; valid range is [flatStart, flatStart + nRanks).
  // The boundary check runs before baseLeId is ever read, so no valid baseId is needed for
  // the two out-of-range probes below. 20 and 7 alone don't pin either exact edge though: a
  // relaxed `>=` -> `>` survives because 12 (the first rejected value above the range) is
  // never probed, and a relaxed `<` -> `<=` survives because 8 (the first accepted value) is
  // never probed either. 12 closes the first; 8 (further down, where it can succeed) closes
  // the second.
  comm->rank = 10;
  comm->devrState.cftSize = 4;
  comm->devrState.cftSelf = 2; // flatTeam.rank == 2 -> flatStart == 8, valid peers in [8, 12)
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetPeerDeviceLeInfo(window, 0, /*peerWorld=*/20, &leId, &leOffset), ncclInvalidArgument);
  EXPECT_EQ(ncclGetPeerDeviceLeInfo(window, 0, /*peerWorld=*/7, &leId, &leOffset), ncclInvalidArgument);
  EXPECT_EQ(ncclGetPeerDeviceLeInfo(window, 0, /*peerWorld=*/12, &leId, &leOffset), ncclInvalidArgument);

  // peerWorld=8 is the lower inclusive edge: it must succeed, not reject. This path proceeds
  // past the boundary check into the baseLeId read, so (unlike the three probes above) it
  // needs a valid baseId.
  comm->devrState.le[0].baseId = 1;
  ASSERT_EQ(ncclGetPeerDeviceLeInfo(window, 0, /*peerWorld=*/8, &leId, &leOffset), ncclSuccess);
  EXPECT_EQ(leId, 1u); // baseLeId(1) + flatTeam.rank(2) + (peerWorld(8) - comm->rank(10)) == 1
}

TEST_F(NcclGetPeerDeviceLeInfoTest, CountedWindow_UsesCountedBaseLeIdSlot) {
  comm->rank = 10;
  comm->devrState.cftSize = 4;
  comm->devrState.cftSelf = 2; // flatStart = 8, valid peers in [8, 12)
  comm->devrState.le[0].baseId = NCCL_LE_ID_INVALID; // uncounted slot deliberately unset
  comm->devrState.le[1].baseId = 2000;               // counted slot is what must be read
  winHost.winFlags = NCCL_WIN_CFT_COUNTED;

  ncclCftLeId leId = 0;
  size_t leOffset = 0;
  // *leId = baseLeId + flatTeam.rank + (peerWorld - comm->rank) = 2000 + 2 + (11 - 10) = 2003
  ASSERT_EQ(ncclGetPeerDeviceLeInfo(window, 0, /*peerWorld=*/11, &leId, &leOffset), ncclSuccess);
  EXPECT_EQ(leId, 2003u);
}

TEST_F(NcclGetPeerDeviceLeInfoTest, BaseLeIdNotYetCreated_ReturnsInvalidUsage) {
  comm->rank = 10;
  comm->devrState.cftSize = 4;
  comm->devrState.cftSelf = 2;
  comm->devrState.le[0].baseId = NCCL_LE_ID_INVALID;
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetPeerDeviceLeInfo(window, 0, /*peerWorld=*/10, &leId, &leOffset), ncclInvalidUsage);
}

TEST_F(NcclGetPeerDeviceLeInfoTest, SuccessPath_AddsFlatTeamRankAndWorldOffsetToBaseLeId) {
  comm->rank = 10;
  comm->devrState.cftSize = 4;
  comm->devrState.cftSelf = 2; // flatStart = 8, valid peers in [8, 12)
  comm->devrState.le[0].baseId = 1000;
  winHost.winFlags = 0;

  ncclCftLeId leId = 0;
  size_t leOffset = 0;
  // *leId = baseLeId + flatTeam.rank + (peerWorld - comm->rank) = 1000 + 2 + (11 - 10) = 1003
  ASSERT_EQ(ncclGetPeerDeviceLeInfo(window, /*offset=*/0x30, /*peerWorld=*/11, &leId, &leOffset), ncclSuccess);
  EXPECT_EQ(leId, 1003u);
  EXPECT_EQ(leOffset, winHost.bigOffset + 0x30);
}
