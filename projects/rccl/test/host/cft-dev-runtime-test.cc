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
#include <string>
#include <vector>

#include <unistd.h>

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

// computeCftSizes: starts from computeLsaTeamSize for both sizes (one rank on ROCm, where gpuCftSupport is 0);
// with gpuCftSupport >= 13040 it switches to the CUDA logical-endpoint clique sizes (gcd of the runs of equal
// peerInfo clique ids) when every rank reported an id, and rejects clique sizes that do not nest.

namespace {
// Gives the comm nRanks peerInfo entries with the given unicast/multicast clique ids.
struct CliquePeers {
  std::vector<ncclPeerInfo> peers;
  CliquePeers(ncclComm* comm, std::vector<uint32_t> const& uc, std::vector<uint32_t> const& mc)
      : peers(uc.size()) {
    for (size_t r = 0; r < uc.size(); r++) {
      peers[r].cftUnicastCliqueId = uc[r];
      peers[r].cftMulticastCliqueId = mc[r];
    }
    comm->nRanks = static_cast<int>(uc.size());
    comm->peerInfo = peers.data();
  }
};
} // namespace

// [RCCL] ROCm has no logical endpoints (gpuCftSupport stays 0), so the CFT teams keep one rank instead of taking
// the LSA team size, which would hand out multi-rank teams with no usable LE addressing.
TEST_F(CftDevRuntimeMicrotest, ComputeCftSizes_NoGpuCftSupport_BothSizesAreOne) {
  comm->gpuCftSupport = 0;
  comm->nRanks = 8;
  g_computeLsaTeamSize = [](ncclComm*) { return 8; };
  int cftSize = -1, cftMcSize = -1;
  EXPECT_EQ(computeCftSizes(comm, &cftSize, &cftMcSize), ncclSuccess);
  EXPECT_EQ(cftSize, 1);
  EXPECT_EQ(cftMcSize, 1);
}

// [RCCL] NCCL_CFT_{UC,MC}_TEAM_SIZE must not bring the multi-rank team back without logical endpoints. The params are
// cached per process, so each case runs in a fresh child: the exit code is 100 on error, else cftSize * 10 + cftMcSize.
TEST_F(CftDevRuntimeMicrotest, ComputeCftSizes_NoGpuCftSupport_IgnoresTeamSizeOverrides) {
  const std::string style = ::testing::GTEST_FLAG(death_test_style);
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  comm->gpuCftSupport = 0;
  comm->nRanks = 8;
  g_computeLsaTeamSize = [](ncclComm*) { return 8; };
  auto run = [this] {
    int cftSize = -1, cftMcSize = -1;
    if (computeCftSizes(comm, &cftSize, &cftMcSize) != ncclSuccess) _exit(100);
    _exit(cftSize * 10 + cftMcSize);
  };
  EXPECT_EXIT((setenv("NCCL_CFT_UC_TEAM_SIZE", "8", 1), unsetenv("NCCL_CFT_MC_TEAM_SIZE"), run()),
              ::testing::ExitedWithCode(11), "");
  EXPECT_EXIT((unsetenv("NCCL_CFT_UC_TEAM_SIZE"), setenv("NCCL_CFT_MC_TEAM_SIZE", "4", 1), run()),
              ::testing::ExitedWithCode(11), "");
  EXPECT_EXIT((setenv("NCCL_CFT_UC_TEAM_SIZE", "1", 1), setenv("NCCL_CFT_MC_TEAM_SIZE", "1", 1), run()),
              ::testing::ExitedWithCode(11), "");
  ::testing::GTEST_FLAG(death_test_style) = style;
}

TEST_F(CftDevRuntimeMicrotest, ComputeCftSizes_GpuCftSupportJustBelow13040_IgnoresCliqueIds) {
  // 13039 is the first value the `>= 13040` check must reject: the clique ids below would give UC 4 / MC 2.
  CliquePeers p(comm, {0, 0, 0, 0, 1, 1, 1, 1}, {0, 0, 1, 1, 2, 2, 3, 3});
  comm->gpuCftSupport = 13039;
  comm->gpuCftMulticastSupport = true;
  g_computeLsaTeamSize = [](ncclComm*) { return 2; };
  int cftSize = -1, cftMcSize = -1;
  EXPECT_EQ(computeCftSizes(comm, &cftSize, &cftMcSize), ncclSuccess);
  EXPECT_EQ(cftSize, 2);
  EXPECT_EQ(cftMcSize, 2);
}

TEST_F(CftDevRuntimeMicrotest, ComputeCftSizes_CliqueIdsWithMulticast_UsesCliqueSizes) {
  CliquePeers p(comm, {0, 0, 0, 0, 1, 1, 1, 1}, {0, 0, 1, 1, 2, 2, 3, 3});
  comm->gpuCftSupport = 13040;
  comm->gpuCftMulticastSupport = true;
  g_computeLsaTeamSize = [](ncclComm*) { return 2; };
  int cftSize = -1, cftMcSize = -1;
  EXPECT_EQ(computeCftSizes(comm, &cftSize, &cftMcSize), ncclSuccess);
  EXPECT_EQ(cftSize, 4);
  EXPECT_EQ(cftMcSize, 2);
}

TEST_F(CftDevRuntimeMicrotest, ComputeCftSizes_CliqueIdsUnevenRuns_UsesGcdOfRunLengths) {
  // Unicast runs of 4 and 2 give gcd 2.
  CliquePeers p(comm, {5, 5, 5, 5, 6, 6}, {0, 0, 0, 0, 0, 0});
  comm->gpuCftSupport = 13040;
  comm->gpuCftMulticastSupport = false;
  g_computeLsaTeamSize = [](ncclComm*) { return 1; };
  int cftSize = -1, cftMcSize = -1;
  EXPECT_EQ(computeCftSizes(comm, &cftSize, &cftMcSize), ncclSuccess);
  EXPECT_EQ(cftSize, 2);
  EXPECT_EQ(cftMcSize, 1); // no multicast support: MC clique size is 1
}

TEST_F(CftDevRuntimeMicrotest, ComputeCftSizes_RankWithoutUnicastCliqueId_FallsBackToLsaTeamSize) {
  CliquePeers p(comm, {0, 0, UINT32_MAX, 0}, {0, 0, 0, 0});
  comm->gpuCftSupport = 13040;
  comm->gpuCftMulticastSupport = true;
  g_computeLsaTeamSize = [](ncclComm*) { return 4; };
  int cftSize = -1, cftMcSize = -1;
  EXPECT_EQ(computeCftSizes(comm, &cftSize, &cftMcSize), ncclSuccess);
  EXPECT_EQ(cftSize, 4);
  EXPECT_EQ(cftMcSize, 4);
}

TEST_F(CftDevRuntimeMicrotest, ComputeCftSizes_RankWithoutMulticastCliqueId_FallsBackToLsaTeamSize) {
  CliquePeers p(comm, {0, 0, 0, 0}, {0, 0, 1, UINT32_MAX});
  comm->gpuCftSupport = 13040;
  comm->gpuCftMulticastSupport = true;
  g_computeLsaTeamSize = [](ncclComm*) { return 2; };
  int cftSize = -1, cftMcSize = -1;
  EXPECT_EQ(computeCftSizes(comm, &cftSize, &cftMcSize), ncclSuccess);
  EXPECT_EQ(cftSize, 2);
  EXPECT_EQ(cftMcSize, 2);
}

TEST_F(CftDevRuntimeMicrotest, ComputeCftSizes_UnicastCliqueNotMultipleOfLsaTeamSize_ReturnsInternalError) {
  CliquePeers p(comm, {0, 0, 1, 1}, {0, 0, 1, 1});
  comm->gpuCftSupport = 13040;
  comm->gpuCftMulticastSupport = true;
  g_computeLsaTeamSize = [](ncclComm*) { return 4; };
  int cftSize = -1, cftMcSize = -1;
  EXPECT_EQ(computeCftSizes(comm, &cftSize, &cftMcSize), ncclInternalError);
}

TEST_F(CftDevRuntimeMicrotest, ComputeCftSizes_UnicastCliqueNotMultipleOfMulticastClique_ReturnsInternalError) {
  // UC clique 2, MC clique 4: UC % MC != 0.
  CliquePeers p(comm, {0, 0, 1, 1}, {0, 0, 0, 0});
  comm->gpuCftSupport = 13040;
  comm->gpuCftMulticastSupport = true;
  g_computeLsaTeamSize = [](ncclComm*) { return 1; };
  int cftSize = -1, cftMcSize = -1;
  EXPECT_EQ(computeCftSizes(comm, &cftSize, &cftMcSize), ncclInternalError);
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
