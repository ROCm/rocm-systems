/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/cft_dev_runtime.cc: the CFT (confidential/
// fault-tolerant?) logical-endpoint support-probe, size-computation, and
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

#include <cstring>
#include <memory>

#include "fakes/cft_dev_runtime_fakes.h"
#include "fakes/dev_runtime_fakes.h"

#include CORE_CC_PATH
#include CFT_DEV_RUNTIME_CC_PATH

namespace {

class NcclCftDevRuntimeMicrotest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>(); // value-initialised: POD members zeroed
    comm = commStorage.get();
    // bigSize != 0 makes the real ncclDevrInitOnce (called by ncclTeamCft/
    // ncclTeamCftMultimem via core.cc) a no-op that leaves devrState as set below.
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
  g_devrWinRegEnabled = [&](int winFlags, ncclDevrRegisterCapability) {
    sawWinFlags = winFlags;
    return true;
  };
  EXPECT_EQ(symUnbindTeamLe(comm, mem.get(), /*le=*/3), ncclSuccess);
  EXPECT_EQ(sawWinFlags, 0x3c);
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

// ---------------------------------------------------------------------------
// ncclGetMultimemDeviceLeInfo
// ---------------------------------------------------------------------------

class NcclGetMultimemDeviceLeInfoTest : public NcclCftDevRuntimeMicrotest {
protected:
  ncclDevrWindow winHost{};
  int dummyWindowTarget = 0;
  ncclWindow_t window = reinterpret_cast<ncclWindow_t>(&dummyWindowTarget);

  void SetUp() override {
    NcclCftDevRuntimeMicrotest::SetUp();
    comm->gpuCftSupport = 13030;
    winHost.bigOffset = 0x1000;
    g_foundComm = comm;
    g_foundWinHost = &winHost;
    UseFoundCommAndWindow();
  }
};

TEST_F(NcclGetMultimemDeviceLeInfoTest, NullArgs_RejectedByPtrCheckBeforeAnyLookup) {
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_NE(ncclGetMultimemDeviceLeInfo(nullptr, 0, &leId, &leOffset), ncclSuccess);
  EXPECT_NE(ncclGetMultimemDeviceLeInfo(window, 0, nullptr, &leOffset), ncclSuccess);
  EXPECT_NE(ncclGetMultimemDeviceLeInfo(window, 0, &leId, nullptr), ncclSuccess);
}

TEST_F(NcclGetMultimemDeviceLeInfoTest, WindowNotCftRegistered_ReturnsInvalidUsage) {
  g_devrWinRegEnabled = [](int, ncclDevrRegisterCapability) { return false; };
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetMultimemDeviceLeInfo(window, 0, &leId, &leOffset), ncclInvalidUsage);
}

TEST_F(NcclGetMultimemDeviceLeInfoTest, CommHasNoGpuCftSupport_ReturnsInvalidArgument) {
  comm->gpuCftSupport = 0;
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetMultimemDeviceLeInfo(window, 0, &leId, &leOffset), ncclInvalidArgument);
}

TEST_F(NcclGetMultimemDeviceLeInfoTest, FindCommAndHostWindowFails_PropagatesErrorViaNcclcheck) {
  g_findCommAndHostWindowFromDeviceWindow = [](ncclWindow_t, ncclComm_t*, ncclDevrWindow**) {
    return ncclSystemError;
  };
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetMultimemDeviceLeInfo(window, 0, &leId, &leOffset), ncclSystemError);
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
// of scope for this file's tests -- see the PR description's coverage note.

TEST_F(NcclGetMultimemDeviceLeInfoTest, SuccessPath_ReadsMcLeIdByCountedBitAndOffsetsByBigOffsetPlusOffset) {
  comm->devrState.cftMcSize = 4;
  comm->devrState.cftMcSelf = 1;
  winHost.winFlags = NCCL_WIN_CFT_COUNTED;

  // ncclDevrTeam has a flexible array member (worldRankList[]), which clang
  // refuses to let a lambda capture directly (even by reference) -- capture a
  // pointer to it instead.
  ncclDevrTeam capturedTeamStorage{};
  capturedTeamStorage.mcLeId[0] = 111;
  capturedTeamStorage.mcLeId[1] = 222; // counted == true selects this one
  ncclDevrTeam* capturedTeamPtr = &capturedTeamStorage;
  bool sawNeedBarrierRequested = false;
  g_symTeamObtain = [&, capturedTeamPtr](ncclComm*, ncclTeam team, bool multimem, bool counted, bool wantsLeUc,
                                        bool wantsLeMc, ncclDevrTeam** outTeam, bool* needBarrier) {
    EXPECT_FALSE(multimem);
    EXPECT_TRUE(counted); // from winHost.winFlags & NCCL_WIN_CFT_COUNTED
    EXPECT_FALSE(wantsLeUc);
    EXPECT_TRUE(wantsLeMc);
    EXPECT_EQ(team.nRanks, 4);
    EXPECT_EQ(team.rank, 1);
    *outTeam = capturedTeamPtr;
    *needBarrier = true;
    sawNeedBarrierRequested = true;
    return ncclSuccess;
  };

  ncclCftLeId leId = 0;
  size_t leOffset = 0;
  ASSERT_EQ(ncclGetMultimemDeviceLeInfo(window, /*offset=*/0x20, &leId, &leOffset), ncclSuccess);
  EXPECT_TRUE(sawNeedBarrierRequested);
  EXPECT_EQ(leId, 222u);
  EXPECT_EQ(leOffset, winHost.bigOffset + 0x20);
}

TEST_F(NcclGetMultimemDeviceLeInfoTest, UncountedWindow_ReadsMcLeIdZeroSlot) {
  comm->devrState.cftMcSize = 4;
  comm->devrState.cftMcSelf = 0;
  winHost.winFlags = 0; // not counted

  ncclDevrTeam capturedTeamStorage{};
  capturedTeamStorage.mcLeId[0] = 55;
  capturedTeamStorage.mcLeId[1] = 66;
  ncclDevrTeam* capturedTeamPtr = &capturedTeamStorage;
  g_symTeamObtain = [capturedTeamPtr](ncclComm*, ncclTeam, bool, bool counted, bool, bool, ncclDevrTeam** outTeam,
                                     bool* needBarrier) {
    EXPECT_FALSE(counted);
    *outTeam = capturedTeamPtr;
    *needBarrier = false; // no barrier requested this time
    return ncclSuccess;
  };

  ncclCftLeId leId = 0;
  size_t leOffset = 0;
  ASSERT_EQ(ncclGetMultimemDeviceLeInfo(window, /*offset=*/0, &leId, &leOffset), ncclSuccess);
  EXPECT_EQ(leId, 55u);
}

// ---------------------------------------------------------------------------
// ncclGetCftDeviceLeInfo
// ---------------------------------------------------------------------------

class NcclGetCftDeviceLeInfoTest : public NcclCftDevRuntimeMicrotest {
protected:
  ncclDevrWindow winHost{};
  int dummyWindowTarget = 0;
  ncclWindow_t window = reinterpret_cast<ncclWindow_t>(&dummyWindowTarget);

  void SetUp() override {
    NcclCftDevRuntimeMicrotest::SetUp();
    comm->gpuCftSupport = 13030;
    winHost.bigOffset = 0x2000;
    g_foundComm = comm;
    g_foundWinHost = &winHost;
    UseFoundCommAndWindow();
  }
};

TEST_F(NcclGetCftDeviceLeInfoTest, NullArgs_RejectedByPtrCheck) {
  ncclTeam_t cftTeam{4, 0, 1};
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_NE(ncclGetCftDeviceLeInfo(nullptr, 0, 0, cftTeam, &leId, &leOffset), ncclSuccess);
}

TEST_F(NcclGetCftDeviceLeInfoTest, WindowNotCftRegistered_ReturnsInvalidUsage) {
  g_devrWinRegEnabled = [](int, ncclDevrRegisterCapability) { return false; };
  ncclTeam_t cftTeam{4, 0, 1};
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetCftDeviceLeInfo(window, 0, 0, cftTeam, &leId, &leOffset), ncclInvalidUsage);
}

TEST_F(NcclGetCftDeviceLeInfoTest, CommHasNoGpuCftSupport_ReturnsInvalidArgument) {
  comm->gpuCftSupport = 0;
  ncclTeam_t cftTeam{4, 0, 1};
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetCftDeviceLeInfo(window, 0, 0, cftTeam, &leId, &leOffset), ncclInvalidArgument);
}

TEST_F(NcclGetCftDeviceLeInfoTest, FindCommAndHostWindowFails_PropagatesErrorViaNcclcheck) {
  g_findCommAndHostWindowFromDeviceWindow = [](ncclWindow_t, ncclComm_t*, ncclDevrWindow**) {
    return ncclSystemError;
  };
  ncclTeam_t cftTeam{4, 0, 1};
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetCftDeviceLeInfo(window, 0, 0, cftTeam, &leId, &leOffset), ncclSystemError);
}

TEST_F(NcclGetCftDeviceLeInfoTest, BaseLeIdNotYetCreated_ReturnsInvalidUsage) {
  comm->devrState.le[0].baseId = NCCL_LE_ID_INVALID;
  ncclTeam_t cftTeam{4, 0, 1};
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetCftDeviceLeInfo(window, 0, 0, cftTeam, &leId, &leOffset), ncclInvalidUsage);
}

TEST_F(NcclGetCftDeviceLeInfoTest, SuccessPath_AddsFlatTeamRankAndPeerOffsetToBaseLeId) {
  // ncclTeamCft(comm, FLAT) reads straight from devrState.cftSize/cftSelf (bigSize != 0
  // from the fixture keeps ncclDevrInitOnce a no-op).
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

class NcclGetPeerDeviceLeInfoTest : public NcclCftDevRuntimeMicrotest {
protected:
  ncclDevrWindow winHost{};
  int dummyWindowTarget = 0;
  ncclWindow_t window = reinterpret_cast<ncclWindow_t>(&dummyWindowTarget);

  void SetUp() override {
    NcclCftDevRuntimeMicrotest::SetUp();
    comm->gpuCftSupport = 13030;
    winHost.bigOffset = 0x3000;
    g_foundComm = comm;
    g_foundWinHost = &winHost;
    UseFoundCommAndWindow();
  }
};

TEST_F(NcclGetPeerDeviceLeInfoTest, WindowNotCftRegistered_ReturnsInvalidUsage) {
  g_devrWinRegEnabled = [](int, ncclDevrRegisterCapability) { return false; };
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetPeerDeviceLeInfo(window, 0, 0, &leId, &leOffset), ncclInvalidUsage);
}

TEST_F(NcclGetPeerDeviceLeInfoTest, CommHasNoGpuCftSupport_ReturnsInvalidArgument) {
  comm->gpuCftSupport = 0;
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetPeerDeviceLeInfo(window, 0, 0, &leId, &leOffset), ncclInvalidArgument);
}

TEST_F(NcclGetPeerDeviceLeInfoTest, FindCommAndHostWindowFails_PropagatesErrorViaNcclcheck) {
  g_findCommAndHostWindowFromDeviceWindow = [](ncclWindow_t, ncclComm_t*, ncclDevrWindow**) {
    return ncclSystemError;
  };
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetPeerDeviceLeInfo(window, 0, 0, &leId, &leOffset), ncclSystemError);
}

TEST_F(NcclGetPeerDeviceLeInfoTest, PeerOutsideFlatTeamBoundaries_ReturnsInvalidArgument) {
  // flatStart = comm->rank - flatTeam.rank; valid range is [flatStart, flatStart + nRanks).
  comm->rank = 10;
  comm->devrState.cftSize = 4;
  comm->devrState.cftSelf = 2; // flatTeam.rank == 2 -> flatStart == 8, valid peers in [8, 12)
  comm->devrState.le[0].baseId = 1; // must be valid so we reach the boundary check
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetPeerDeviceLeInfo(window, 0, /*peerWorld=*/20, &leId, &leOffset), ncclInvalidArgument);
  EXPECT_EQ(ncclGetPeerDeviceLeInfo(window, 0, /*peerWorld=*/7, &leId, &leOffset), ncclInvalidArgument);
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
