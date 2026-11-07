/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/cft_dev_runtime.cc, with src/nccl_device/core.cc compiled for real.
// CUDA_VERSION is below 13030 here, so the CUDA logical-endpoint calls are compiled out.

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

#include "fakes/dev_runtime_fakes.h"

#include CORE_CC_PATH
#include CFT_DEV_RUNTIME_CC_PATH

namespace {

// Heap-allocated so worldRankList has team.nRanks elements, like symTeamObtain's own teams.
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

class CftDevRuntimeMicrotest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>(); // value-initialised: POD members zeroed
    comm = commStorage.get();
    // Only computeCftSize/computeCftMcSize read bigSize; the ncclDevrInitOnce fake ignores it.
    comm->devrState.bigSize = 1;
  }

  void TearDown() override {
    ResetDevRuntimeFakes();
  }
};

} // namespace

// ncclGpuCftSupport: only the zero-and-succeed prologue is compiled on this build.

TEST_F(CftDevRuntimeMicrotest, GpuCftSupport_ZeroesAllThreeOutputsAndSucceeds) {
  int gpuCftSupport = 99;
  bool gpuCftMulticastSupport = true, gpuCftCountedSupport = true;
  EXPECT_EQ(ncclGpuCftSupport(comm, &gpuCftSupport, &gpuCftMulticastSupport, &gpuCftCountedSupport), ncclSuccess);
  EXPECT_EQ(gpuCftSupport, 0);
  EXPECT_FALSE(gpuCftMulticastSupport);
  EXPECT_FALSE(gpuCftCountedSupport);
}

// computeCftSize / computeCftMcSize

TEST_F(CftDevRuntimeMicrotest, ComputeCftSize_AlreadySized_ReturnsCftSizeWithoutCallingComputeLsaSize) {
  comm->devrState.bigSize = 7;
  comm->devrState.cftSize = 42;
  bool called = false;
  g_computeLsaSize = [&](ncclComm*) { called = true; return 1; };
  EXPECT_EQ(computeCftSize(comm), 42);
  EXPECT_FALSE(called);
}

TEST_F(CftDevRuntimeMicrotest, ComputeCftSize_NotSizedNoGpuCftSupport_ReturnsOneWithoutCallingComputeLsaSize) {
  comm->devrState.bigSize = 0;
  comm->gpuCftSupport = 0;
  bool called = false;
  g_computeLsaSize = [&](ncclComm*) { called = true; return 99; };
  EXPECT_EQ(computeCftSize(comm), 1);
  EXPECT_FALSE(called);
}

TEST_F(CftDevRuntimeMicrotest, ComputeCftSize_NotSizedWithGpuCftSupport_DelegatesToComputeLsaSize) {
  comm->devrState.bigSize = 0;
  comm->gpuCftSupport = 13030;
  g_computeLsaSize = [](ncclComm*) { return 8; };
  EXPECT_EQ(computeCftSize(comm), 8);
}

TEST_F(CftDevRuntimeMicrotest, ComputeCftSizeAndMcSize_GpuCftSupportJustBelow13030_ReturnsOne) {
  // 13029 is the first value the `>= 13030` check must reject.
  comm->devrState.bigSize = 0;
  comm->gpuCftSupport = 13029;
  bool called = false;
  g_computeLsaSize = [&](ncclComm*) { called = true; return 99; };
  EXPECT_EQ(computeCftSize(comm), 1);
  EXPECT_EQ(computeCftMcSize(comm), 1);
  EXPECT_FALSE(called);
}

TEST_F(CftDevRuntimeMicrotest, ComputeCftMcSize_AlreadySized_ReturnsCftMcSizeWithoutCallingComputeLsaSize) {
  comm->devrState.bigSize = 7;
  comm->devrState.cftMcSize = 24;
  bool called = false;
  g_computeLsaSize = [&](ncclComm*) { called = true; return 1; };
  EXPECT_EQ(computeCftMcSize(comm), 24);
  EXPECT_FALSE(called);
}

TEST_F(CftDevRuntimeMicrotest, ComputeCftMcSize_NotSizedNoGpuCftSupport_ReturnsOneWithoutCallingComputeLsaSize) {
  comm->devrState.bigSize = 0;
  comm->gpuCftSupport = 0;
  bool called = false;
  g_computeLsaSize = [&](ncclComm*) { called = true; return 99; };
  EXPECT_EQ(computeCftMcSize(comm), 1);
  EXPECT_FALSE(called);
}

TEST_F(CftDevRuntimeMicrotest, ComputeCftMcSize_NotSizedWithGpuCftSupport_DelegatesToComputeLsaSize) {
  comm->devrState.bigSize = 0;
  comm->gpuCftSupport = 13030;
  g_computeLsaSize = [](ncclComm*) { return 6; };
  EXPECT_EQ(computeCftMcSize(comm), 6);
}

// symBindTeamLe / symUnbindTeamLe: the CUDA calls are compiled out, so they always succeed and
// only the ncclDevrWinRegEnabled query is observable.

TEST_F(CftDevRuntimeMicrotest, SymBindTeamLe_AlwaysSucceedsAndConsultsWinRegEnabledWithCftCapability) {
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

TEST_F(CftDevRuntimeMicrotest, SymUnbindTeamLe_AlwaysSucceedsAndConsultsWinRegEnabledWithCftCapability) {
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

// symTeamObtainUcLe / symTeamObtainMcLe: this build's stand-ins always fail.

TEST_F(CftDevRuntimeMicrotest, SymTeamObtainUcLe_UnsupportedOnThisPlatform_ReturnsInvalidUsage) {
  bool needBarrier = false;
  EXPECT_EQ(symTeamObtainUcLe(comm, /*t=*/nullptr, /*devr=*/nullptr, &needBarrier, /*counted=*/false),
            ncclInvalidUsage);
}

TEST_F(CftDevRuntimeMicrotest, SymTeamObtainMcLe_UnsupportedOnThisPlatform_ReturnsInvalidUsage) {
  bool needBarrier = false;
  EXPECT_EQ(symTeamObtainMcLe(comm, /*t=*/nullptr, /*devr=*/nullptr, &needBarrier, /*counted=*/true),
            ncclInvalidUsage);
}

// Shared setup for the three ncclGet*DeviceLeInfo queries; BigOffset() is distinct per suite.
class CftDevRuntimeLeInfoMicrotestBase : public CftDevRuntimeMicrotest {
protected:
  ncclDevrWindow winHost{};
  int dummyWindowTarget = 0;
  ncclWindow_t window = reinterpret_cast<ncclWindow_t>(&dummyWindowTarget);
  // What the ncclDevrWinRegEnabled seam was last asked, and what it answers.
  int sawWinFlags = -1;
  ncclDevrRegisterCapability sawCapability = ncclDevrRegisterAll;
  bool winRegEnabled = true;

  virtual size_t BigOffset() const = 0;
  // Calls this suite's query with its other arguments fixed.
  virtual ncclResult_t Invoke(ncclWindow_t w, ncclCftLeId* leId, size_t* leOffset) = 0;

  void SetUp() override {
    CftDevRuntimeMicrotest::SetUp();
    comm->gpuCftSupport = 13030;
    // At cftSize 0 the peer query rejects every peer before reaching the checks below.
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

  void ExpectNullArgsRejectedByPtrCheck() {
    ncclCftLeId leId;
    size_t leOffset;
    EXPECT_EQ(Invoke(nullptr, &leId, &leOffset), ncclInvalidArgument);
    EXPECT_EQ(Invoke(window, nullptr, &leOffset), ncclInvalidArgument);
    EXPECT_EQ(Invoke(window, &leId, nullptr), ncclInvalidArgument);
  }

  void ExpectWindowNotCftRegisteredReturnsInvalidUsage() {
    // Asked with this window's flags, against the CFT capability.
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

// ncclGetMultimemDeviceLeInfo

class CftDevRuntimeMultimemLeInfoMicrotest : public CftDevRuntimeLeInfoMicrotestBase {
protected:
  size_t BigOffset() const override { return 0x1000; }
  ncclResult_t Invoke(ncclWindow_t w, ncclCftLeId* leId, size_t* leOffset) override {
    return ncclGetMultimemDeviceLeInfo(w, 0, leId, leOffset);
  }
};

TEST_F(CftDevRuntimeMultimemLeInfoMicrotest, NullArgs_RejectedByPtrCheckBeforeAnyLookup) {
  ExpectNullArgsRejectedByPtrCheck();
}
TEST_F(CftDevRuntimeMultimemLeInfoMicrotest, WindowNotCftRegistered_ReturnsInvalidUsage) {
  ExpectWindowNotCftRegisteredReturnsInvalidUsage();
}
TEST_F(CftDevRuntimeMultimemLeInfoMicrotest, CommHasNoGpuCftSupport_ReturnsInvalidArgument) {
  ExpectNoGpuCftSupportReturnsInvalidArgument();
}
TEST_F(CftDevRuntimeMultimemLeInfoMicrotest, FindCommAndHostWindowFails_PropagatesErrorViaNcclcheck) {
  ExpectFindCommAndHostWindowFailurePropagates();
}

TEST_F(CftDevRuntimeMultimemLeInfoMicrotest, SymTeamObtainFails_PropagatesErrorViaNcclcheck) {
  g_symTeamObtain = [](ncclComm*, ncclTeam, bool, bool, bool, bool, ncclDevrTeam**, bool*) {
    return ncclInvalidUsage;
  };
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetMultimemDeviceLeInfo(window, 0, &leId, &leOffset), ncclInvalidUsage);
}

// The bootstrapIntraNodeBarrier failure branch isn't covered: fakes/bootstrap_stubs.cc has no
// failure hook.

TEST_F(CftDevRuntimeMultimemLeInfoMicrotest, SuccessPath_ReadsMcLeIdByCountedBitAndOffsetsByBigOffsetPlusOffset) {
  comm->devrState.cftMcSize = 4;
  comm->devrState.cftMcSelf = 1;
  // SYMMETRIC alongside CFT_COUNTED, so `== NCCL_WIN_CFT_COUNTED` would fail where `&` passes.
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

TEST_F(CftDevRuntimeMultimemLeInfoMicrotest, UncountedWindow_ReadsMcLeIdZeroSlot) {
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

TEST_F(CftDevRuntimeMultimemLeInfoMicrotest, SymmetricOnlyWindow_IsUncountedAndReadsMcLeIdZeroSlot) {
  // SYMMETRIC without CFT_COUNTED: testing the wrong bit would read the counted slot.
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

// ncclGetCftDeviceLeInfo

class CftDevRuntimeCftLeInfoMicrotest : public CftDevRuntimeLeInfoMicrotestBase {
protected:
  size_t BigOffset() const override { return 0x2000; }
  ncclResult_t Invoke(ncclWindow_t w, ncclCftLeId* leId, size_t* leOffset) override {
    return ncclGetCftDeviceLeInfo(w, 0, /*peerCft=*/0, ncclTeam_t{4, 0, 1}, leId, leOffset);
  }
};

TEST_F(CftDevRuntimeCftLeInfoMicrotest, NullArgs_RejectedByPtrCheck) {
  ExpectNullArgsRejectedByPtrCheck();
}
TEST_F(CftDevRuntimeCftLeInfoMicrotest, WindowNotCftRegistered_ReturnsInvalidUsage) {
  ExpectWindowNotCftRegisteredReturnsInvalidUsage();
}
TEST_F(CftDevRuntimeCftLeInfoMicrotest, CommHasNoGpuCftSupport_ReturnsInvalidArgument) {
  ExpectNoGpuCftSupportReturnsInvalidArgument();
}
TEST_F(CftDevRuntimeCftLeInfoMicrotest, FindCommAndHostWindowFails_PropagatesErrorViaNcclcheck) {
  ExpectFindCommAndHostWindowFailurePropagates();
}

TEST_F(CftDevRuntimeCftLeInfoMicrotest, BaseLeIdNotYetCreated_ReturnsInvalidUsage) {
  comm->devrState.le[0].baseId = NCCL_LE_ID_INVALID;
  ncclTeam_t cftTeam{4, 0, 1};
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetCftDeviceLeInfo(window, 0, 0, cftTeam, &leId, &leOffset), ncclInvalidUsage);
}

TEST_F(CftDevRuntimeCftLeInfoMicrotest, SuccessPath_AddsFlatTeamRankAndPeerOffsetToBaseLeId) {
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

TEST_F(CftDevRuntimeCftLeInfoMicrotest, CountedWindow_UsesCountedBaseLeIdSlot) {
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

// ncclGetPeerDeviceLeInfo

class CftDevRuntimePeerLeInfoMicrotest : public CftDevRuntimeLeInfoMicrotestBase {
protected:
  size_t BigOffset() const override { return 0x3000; }
  ncclResult_t Invoke(ncclWindow_t w, ncclCftLeId* leId, size_t* leOffset) override {
    return ncclGetPeerDeviceLeInfo(w, 0, /*peerWorld=*/0, leId, leOffset);
  }
};

TEST_F(CftDevRuntimePeerLeInfoMicrotest, NullArgs_RejectedByPtrCheck) {
  ExpectNullArgsRejectedByPtrCheck();
}
TEST_F(CftDevRuntimePeerLeInfoMicrotest, WindowNotCftRegistered_ReturnsInvalidUsage) {
  ExpectWindowNotCftRegisteredReturnsInvalidUsage();
}
TEST_F(CftDevRuntimePeerLeInfoMicrotest, CommHasNoGpuCftSupport_ReturnsInvalidArgument) {
  ExpectNoGpuCftSupportReturnsInvalidArgument();
}
TEST_F(CftDevRuntimePeerLeInfoMicrotest, FindCommAndHostWindowFails_PropagatesErrorViaNcclcheck) {
  ExpectFindCommAndHostWindowFailurePropagates();
}

TEST_F(CftDevRuntimePeerLeInfoMicrotest, PeerOutsideFlatTeamBoundaries_ReturnsInvalidArgument) {
  // Valid range is [flatStart, flatStart + nRanks) = [8, 12); 12 and 8 pin the two edges.
  comm->rank = 10;
  comm->devrState.cftSize = 4;
  comm->devrState.cftSelf = 2; // flatTeam.rank == 2 -> flatStart == 8, valid peers in [8, 12)
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetPeerDeviceLeInfo(window, 0, /*peerWorld=*/20, &leId, &leOffset), ncclInvalidArgument);
  EXPECT_EQ(ncclGetPeerDeviceLeInfo(window, 0, /*peerWorld=*/7, &leId, &leOffset), ncclInvalidArgument);
  EXPECT_EQ(ncclGetPeerDeviceLeInfo(window, 0, /*peerWorld=*/12, &leId, &leOffset), ncclInvalidArgument);

  // 8 is the lower inclusive edge, so it must succeed.
  comm->devrState.le[0].baseId = 1;
  ASSERT_EQ(ncclGetPeerDeviceLeInfo(window, 0, /*peerWorld=*/8, &leId, &leOffset), ncclSuccess);
  EXPECT_EQ(leId, 1u); // baseLeId(1) + flatTeam.rank(2) + (peerWorld(8) - comm->rank(10)) == 1
}

TEST_F(CftDevRuntimePeerLeInfoMicrotest, CountedWindow_UsesCountedBaseLeIdSlot) {
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

TEST_F(CftDevRuntimePeerLeInfoMicrotest, BaseLeIdNotYetCreated_ReturnsInvalidUsage) {
  comm->rank = 10;
  comm->devrState.cftSize = 4;
  comm->devrState.cftSelf = 2;
  comm->devrState.le[0].baseId = NCCL_LE_ID_INVALID;
  ncclCftLeId leId;
  size_t leOffset;
  EXPECT_EQ(ncclGetPeerDeviceLeInfo(window, 0, /*peerWorld=*/10, &leId, &leOffset), ncclInvalidUsage);
}

TEST_F(CftDevRuntimePeerLeInfoMicrotest, SuccessPath_AddsFlatTeamRankAndWorldOffsetToBaseLeId) {
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
