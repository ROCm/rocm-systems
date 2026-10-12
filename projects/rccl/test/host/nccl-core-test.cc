/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/nccl_device/core.cc: the ncclTeam* family that
// maps a communicator onto its world/LSA/CFT/rail views, plus the
// world/LSA rank-translation helpers. Every function but ncclTeamRankToWorld
// gates through ncclDevrInitOnce first, so the fake's hook lets us drive both
// the success path (devrState fields read back verbatim) and the failure
// path (all five return a poisoned/zeroed answer rather than touching
// uninitialised devrState).

#include <gtest/gtest.h>

#include <memory>

#include "fakes/dev_runtime_fakes.h"

#include CORE_CC_PATH

namespace {

class NcclCoreMicrotest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>(); // value-initialised: POD members zeroed
    comm = commStorage.get();
  }

  void TearDown() override { ResetDevRuntimeFakes(); }
};

} // namespace

TEST_F(NcclCoreMicrotest, TeamWorld_ReturnsCommNRanksAndRankWithStrideOne) {
  comm->nRanks = 4;
  comm->rank = 2;
  ncclTeam_t team = ncclTeamWorld(comm);
  EXPECT_EQ(team.nRanks, 4);
  EXPECT_EQ(team.rank, 2);
  EXPECT_EQ(team.stride, 1);
}

TEST_F(NcclCoreMicrotest, TeamLsa_SuccessPath_ReadsLsaSizeAndSelfFromDevrState) {
  comm->devrState.lsaSize = 8;
  comm->devrState.lsaSelf = 3;
  ncclTeam_t team = ncclTeamLsa(comm);
  EXPECT_EQ(team.nRanks, 8);
  EXPECT_EQ(team.rank, 3);
  EXPECT_EQ(team.stride, 1);
}

TEST_F(NcclCoreMicrotest, TeamLsa_InitOnceFails_ReturnsZeroedTeamWithoutReadingDevrState) {
  // Poison lsaSize/lsaSelf so that if the function ignored the failure and
  // read them anyway, the zero-check below would fail loudly.
  comm->devrState.lsaSize = 99;
  comm->devrState.lsaSelf = 99;
  g_devrInitOnce = [](ncclComm*) { return ncclSystemError; };
  ncclTeam_t team = ncclTeamLsa(comm);
  EXPECT_EQ(team.nRanks, 0);
  EXPECT_EQ(team.rank, 0);
  EXPECT_EQ(team.stride, 0);
}

TEST_F(NcclCoreMicrotest, TeamCftMultimem_SuccessPath_ReadsCftMcSizeAndSelfFromDevrState) {
  comm->devrState.cftMcSize = 16;
  comm->devrState.cftMcSelf = 5;
  ncclTeam_t team = ncclTeamCftMultimem(comm);
  EXPECT_EQ(team.nRanks, 16);
  EXPECT_EQ(team.rank, 5);
  EXPECT_EQ(team.stride, 1);
}

TEST_F(NcclCoreMicrotest, TeamCftMultimem_InitOnceFails_ReturnsZeroedTeam) {
  comm->devrState.cftMcSize = 99;
  comm->devrState.cftMcSelf = 99;
  g_devrInitOnce = [](ncclComm*) { return ncclSystemError; };
  ncclTeam_t team = ncclTeamCftMultimem(comm);
  EXPECT_EQ(team.nRanks, 0);
  EXPECT_EQ(team.rank, 0);
  EXPECT_EQ(team.stride, 0);
}

TEST_F(NcclCoreMicrotest, TeamRail_SuccessPath_DividesCommNRanksAndRankByLsaSize) {
  comm->nRanks = 16;
  comm->rank = 10;
  comm->devrState.lsaSize = 4;
  ncclTeam_t team = ncclTeamRail(comm);
  EXPECT_EQ(team.nRanks, 4);  // 16 / 4
  EXPECT_EQ(team.rank, 2);    // 10 / 4
  EXPECT_EQ(team.stride, 4);  // == lsaSize
}

TEST_F(NcclCoreMicrotest, TeamRail_InitOnceFails_ReturnsZeroedTeamWithoutDividingByLsaSize) {
  // If the failure branch were missing, this would divide by zero instead.
  comm->nRanks = 16;
  comm->rank = 10;
  comm->devrState.lsaSize = 0;
  g_devrInitOnce = [](ncclComm*) { return ncclSystemError; };
  ncclTeam_t team = ncclTeamRail(comm);
  EXPECT_EQ(team.nRanks, 0);
  EXPECT_EQ(team.rank, 0);
  EXPECT_EQ(team.stride, 0);
}

TEST_F(NcclCoreMicrotest, TeamCft_FlatMode_ReturnsFlatTeamFromCftSizeAndSelf) {
  comm->devrState.cftSize = 12;
  comm->devrState.cftSelf = 7;
  ncclTeam_t team = ncclTeamCft(comm, NCCL_CFT_TEAM_FLAT);
  EXPECT_EQ(team.nRanks, 12);
  EXPECT_EQ(team.rank, 7);
  EXPECT_EQ(team.stride, 1);
}

TEST_F(NcclCoreMicrotest, TeamCft_HierMultimemMode_OuterFactorsTheFlatTeamByCftMcSize) {
  // ncclTeamOuterFactor(flat, innerSize): nRanks = flat.nRanks/innerSize,
  // rank = flat.rank/innerSize, stride = flat.stride*innerSize (== 1*innerSize
  // here since the flat team's stride is always 1).
  comm->devrState.cftSize = 12;
  comm->devrState.cftSelf = 7;
  comm->devrState.cftMcSize = 3;
  ncclTeam_t team = ncclTeamCft(comm, NCCL_CFT_TEAM_HIER_MULTIMEM);
  EXPECT_EQ(team.nRanks, 4);  // 12 / 3
  EXPECT_EQ(team.rank, 2);    // 7 / 3
  EXPECT_EQ(team.stride, 3);  // 1 * 3
}

TEST_F(NcclCoreMicrotest, TeamCft_HierLsaMode_OuterFactorsTheFlatTeamByLsaSize) {
  comm->devrState.cftSize = 20;
  comm->devrState.cftSelf = 9;
  comm->devrState.lsaSize = 5;
  ncclTeam_t team = ncclTeamCft(comm, NCCL_CFT_TEAM_HIER_LSA);
  EXPECT_EQ(team.nRanks, 4);  // 20 / 5
  EXPECT_EQ(team.rank, 1);    // 9 / 5
  EXPECT_EQ(team.stride, 5);  // 1 * 5
}

TEST_F(NcclCoreMicrotest, TeamCft_UnrecognisedMode_ReturnsZeroedTeam) {
  comm->devrState.cftSize = 12;
  comm->devrState.cftSelf = 7;
  ncclTeam_t team = ncclTeamCft(comm, static_cast<ncclCftTeamMode_t>(99));
  EXPECT_EQ(team.nRanks, 0);
  EXPECT_EQ(team.rank, 0);
  EXPECT_EQ(team.stride, 0);
}

TEST_F(NcclCoreMicrotest, TeamCft_InitOnceFails_ReturnsZeroedTeamRegardlessOfMode) {
  comm->devrState.cftSize = 99;
  comm->devrState.cftSelf = 99;
  g_devrInitOnce = [](ncclComm*) { return ncclSystemError; };
  for (ncclCftTeamMode_t mode : {NCCL_CFT_TEAM_FLAT, NCCL_CFT_TEAM_HIER_MULTIMEM, NCCL_CFT_TEAM_HIER_LSA}) {
    ncclTeam_t team = ncclTeamCft(comm, mode);
    EXPECT_EQ(team.nRanks, 0);
    EXPECT_EQ(team.rank, 0);
    EXPECT_EQ(team.stride, 0);
  }
}

TEST_F(NcclCoreMicrotest, TeamRankToWorld_MapsATeamLocalRankBackToItsWorldRankViaStride) {
  // Does not gate on ncclDevrInitOnce at all -- pure arithmetic over comm->rank
  // and the caller-supplied team, so no fake wiring is needed for this one.
  comm->rank = 10;
  ncclTeam_t team{/*nRanks=*/4, /*rank=*/2, /*stride=*/3};
  // comm->rank + (rank - team.rank) * team.stride
  EXPECT_EQ(ncclTeamRankToWorld(comm, team, team.rank), 10); // self: offset 0
  EXPECT_EQ(ncclTeamRankToWorld(comm, team, team.rank + 1), 13);
  EXPECT_EQ(ncclTeamRankToWorld(comm, team, team.rank - 1), 7);
}

TEST_F(NcclCoreMicrotest, TeamRankToLsa_SuccessPath_MapsATeamLocalRankToAnLsaRankViaLsaSelf) {
  comm->devrState.lsaSelf = 5;
  ncclTeam_t team{/*nRanks=*/4, /*rank=*/2, /*stride=*/3};
  // devrState.lsaSelf + (rank - team.rank) * team.stride
  EXPECT_EQ(ncclTeamRankToLsa(comm, team, team.rank), 5);
  EXPECT_EQ(ncclTeamRankToLsa(comm, team, team.rank + 1), 8);
}

TEST_F(NcclCoreMicrotest, TeamRankToLsa_InitOnceFails_ReturnsNegativeOneWithoutReadingLsaSelf) {
  comm->devrState.lsaSelf = 99;
  g_devrInitOnce = [](ncclComm*) { return ncclSystemError; };
  ncclTeam_t team{/*nRanks=*/4, /*rank=*/2, /*stride=*/3};
  EXPECT_EQ(ncclTeamRankToLsa(comm, team, team.rank), -1);
}
