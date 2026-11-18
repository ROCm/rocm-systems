/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/gin/gin_anvil_reach.cc: which communicators the
// Anvil SDMA GIN backend can serve, and when it
// registers windows through fabric memory. Built into
// rccl-UnitTestsMicroGinAnvilReach; this TU #includes the hipified source
// (GIN_ANVIL_REACH_CC_PATH). No GPU, no librccl.so.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "comm.h"
#include "graph/topo.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"

#include GIN_ANVIL_REACH_CC_PATH

namespace {

class GinAnvilReachMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_cuMemEnable = [] { return 1; };
    g_loadParam = ParamOverride({{"RCCL_GIN_ANVIL_FABRIC_MEM", 1}, {"RCCL_GIN_ANVIL_FABRIC_POD", 1}});
  }
  void TearDown() override {
    g_cuMemEnable = [] { return 0; };
    g_loadParam = [](const char*, int64_t deft) { return deft; };
  }
};

struct AnvilReachCase {
  const char* name;
  const char* arch;
  int mnnvl;
  int nHosts;
  int nRanks;
  int cliqueSize;
  int cuMem;
  GinAnvilSdmaReach expected;
};

class GinAnvilReachMicrotestPerSystem : public GinAnvilReachMicrotest,
                                         public ::testing::WithParamInterface<AnvilReachCase> {};

struct ReachComm {
  std::unique_ptr<ncclComm> comm = std::make_unique<ncclComm>();
  std::unique_ptr<ncclTopoSystem> topo = std::make_unique<ncclTopoSystem>();
  std::vector<ncclPeerInfo> peers;
  std::string arch;

  ReachComm(const char* archName, int mnnvl, int nRanks, int cliqueSize, int nHosts = 1, int cuMemSupport = 1)
    : peers(static_cast<size_t>(nRanks)), arch(archName) {
    comm->archName = &arch[0];
    comm->MNNVL = mnnvl;
    comm->nRanks = nRanks;
    comm->clique.size = cliqueSize;
    comm->cuMemSupport = cuMemSupport;
    topo->nHosts = nHosts;
    comm->topo = topo.get();
    for (ncclPeerInfo& p : peers) p.crossNicSupport = true;
    comm->peerInfo = peers.data();
  }
};

}  // namespace

TEST_F(GinAnvilReachMicrotest, NullComm_ReachesNothing) {
  EXPECT_EQ(ginAnvilSdmaReachCapable(nullptr), GinAnvilSdmaReach::None);
  EXPECT_FALSE(ginAnvilUseFabricPod(nullptr));
  EXPECT_FALSE(ginAnvilCrossNic(nullptr));
  EXPECT_FALSE(ginAnvilFabricMemCapable(nullptr));
  EXPECT_FALSE(ginAnvilUseFabricMem(nullptr));
}

// Fabric registration needs MNNVL on an MI450, one clique covering the
// communicator, and cuMem; MNNVL on any other arch does not qualify.
TEST_F(GinAnvilReachMicrotest, UseFabricMem_OnlyForMi450SingleCliqueWithCuMem) {
  EXPECT_TRUE(ginAnvilUseFabricMem(ReachComm("gfx1250", 1, 8, 8).comm.get()));
  EXPECT_FALSE(ginAnvilUseFabricMem(ReachComm("gfx1250", 1, 8, 4).comm.get()));
  EXPECT_FALSE(ginAnvilUseFabricMem(ReachComm("gfx1250", 0, 8, 8).comm.get()));
  EXPECT_FALSE(ginAnvilUseFabricMem(ReachComm("gfx950", 1, 8, 8).comm.get()));
  EXPECT_FALSE(ginAnvilUseFabricMem(ReachComm("gfx1250", 1, 8, 8, /*nHosts=*/1, /*cuMemSupport=*/0).comm.get()));
}

// A caller may run a collective only where ginAnvilFabricMemCapable holds, so a
// rank must not decide from its own cuMem setting: with cuMem on here and off on
// a peer, every rank, this one included, must read false.
TEST_F(GinAnvilReachMicrotest, FabricMemCapable_NeedsCuMemOnEveryRank) {
  g_cuMemEnable = [] { return 1; };
  EXPECT_FALSE(ginAnvilFabricMemCapable(ReachComm("gfx1250", 1, 8, 8, /*nHosts=*/1, /*cuMemSupport=*/0).comm.get()));
  g_cuMemEnable = [] { return 0; };
  EXPECT_TRUE(ginAnvilFabricMemCapable(ReachComm("gfx1250", 1, 8, 8, /*nHosts=*/1, /*cuMemSupport=*/1).comm.get()));
}

// Off unless RCCL_GIN_ANVIL_FABRIC_MEM=1. Capability ignores the opt-in, so a
// rank that did not opt in still joins the plugin's agreement vote.
TEST_F(GinAnvilReachMicrotest, UseFabricMem_NeedsOptIn) {
  ReachComm rc("gfx1250", 1, 8, 8);
  g_loadParam = [](const char*, int64_t deft) { return deft; };
  EXPECT_FALSE(ginAnvilUseFabricMem(rc.comm.get()));
  EXPECT_TRUE(ginAnvilFabricMemCapable(rc.comm.get()));
  EXPECT_FALSE(ginAnvilFabricMemCapable(ReachComm("gfx942", 1, 8, 8).comm.get()));
}

TEST_P(GinAnvilReachMicrotestPerSystem, ByArchAndTopology_CrossesHostsOnlyInMi450Pod) {
  const AnvilReachCase c = GetParam();
  ReachComm rc(c.arch, c.mnnvl, c.nRanks, c.cliqueSize, c.nHosts, c.cuMem);

  EXPECT_EQ(ginAnvilSdmaReachCapable(rc.comm.get()), c.expected);
  EXPECT_EQ(ginAnvilUseFabricPod(rc.comm.get()), c.expected == GinAnvilSdmaReach::FabricPod);
}

INSTANTIATE_TEST_SUITE_P(
    PerSystem, GinAnvilReachMicrotestPerSystem,
    ::testing::Values(
        AnvilReachCase{"Mi300SingleHost", "gfx942:sramecc+:xnack-", 0, 1, 8, 8, 1, GinAnvilSdmaReach::Host},
        AnvilReachCase{"Mi300MultiHostForcedMnnvl", "gfx942", 1, 2, 16, 16, 1, GinAnvilSdmaReach::None},
        AnvilReachCase{"Mi350SingleHost", "gfx950", 0, 1, 8, 8, 0, GinAnvilSdmaReach::Host},
        AnvilReachCase{"Mi350MultiHostForcedMnnvl", "gfx950", 1, 2, 16, 16, 1, GinAnvilSdmaReach::None},
        AnvilReachCase{"Mi450SingleHost", "gfx1250", 0, 1, 4, 4, 0, GinAnvilSdmaReach::Host},
        AnvilReachCase{"Mi450FabricPod", "gfx1250", 1, 2, 8, 8, 1, GinAnvilSdmaReach::FabricPod},
        AnvilReachCase{"Mi450TwoCliques", "gfx1250", 1, 2, 8, 4, 1, GinAnvilSdmaReach::None},
        AnvilReachCase{"Mi450NoCuMem", "gfx1250", 1, 2, 8, 8, 0, GinAnvilSdmaReach::None},
        AnvilReachCase{"Mi450NoMnnvl", "gfx1250", 0, 2, 8, 8, 1, GinAnvilSdmaReach::None},
        AnvilReachCase{"Mi450PodAboveRankCap", "gfx1250", 1, 4, kGinAnvilIpcMaxRanks + 1,
                       kGinAnvilIpcMaxRanks + 1, 1, GinAnvilSdmaReach::None},
        AnvilReachCase{"OtherArchMultiHost", "gfx90a", 1, 2, 8, 8, 1, GinAnvilSdmaReach::None}),
    [](const ::testing::TestParamInfo<AnvilReachCase>& info) { return std::string(info.param.name); });

// Capability is topology only. This rank's pod opt-in also needs both
// RCCL_GIN_ANVIL_FABRIC_POD=1 and RCCL_GIN_ANVIL_FABRIC_MEM=1.
TEST_F(GinAnvilReachMicrotest, UseFabricPod_NeedsBothOptIns) {
  ReachComm rc("gfx1250", 1, 8, 8, 2);
  EXPECT_TRUE(ginAnvilUseFabricPod(rc.comm.get()));
  g_loadParam = ParamOverride({{"RCCL_GIN_ANVIL_FABRIC_MEM", 1}});
  EXPECT_FALSE(ginAnvilUseFabricPod(rc.comm.get()));
  g_loadParam = ParamOverride({{"RCCL_GIN_ANVIL_FABRIC_POD", 1}});
  EXPECT_FALSE(ginAnvilUseFabricPod(rc.comm.get()));
  EXPECT_EQ(ginAnvilSdmaReachCapable(rc.comm.get()), GinAnvilSdmaReach::FabricPod);
}

// NCCL_CROSS_NIC=0 on any rank makes GIN rail-connected, which Anvil's signal
// table cannot map across hosts, so a pod is not formed; a single host is
// unaffected.
TEST_F(GinAnvilReachMicrotest, FabricPod_NeedsCrossNicOnEveryRank) {
  ReachComm pod("gfx1250", 1, 8, 8, 2);
  EXPECT_TRUE(ginAnvilCrossNic(pod.comm.get()));
  pod.peers[5].crossNicSupport = false;
  EXPECT_FALSE(ginAnvilCrossNic(pod.comm.get()));
  EXPECT_EQ(ginAnvilSdmaReachCapable(pod.comm.get()), GinAnvilSdmaReach::None);
  EXPECT_FALSE(ginAnvilUseFabricPod(pod.comm.get()));

  ReachComm host("gfx1250", 1, 4, 4, 1);
  for (ncclPeerInfo& p : host.peers) p.crossNicSupport = false;
  EXPECT_EQ(ginAnvilSdmaReachCapable(host.comm.get()), GinAnvilSdmaReach::Host);
}

// comm->MNNVL follows this rank's NCCL_MNNVL_ENABLE, so a rank without it is not
// in the pod by its own reach, yet it must take the pod vote its peers take and
// vote no.
TEST_F(GinAnvilReachMicrotest, FabricPodVote_IgnoresThisRanksMnnvl) {
  ReachComm rc("gfx1250", /*mnnvl=*/0, 8, 8, /*nHosts=*/2);
  EXPECT_EQ(ginAnvilSdmaReachCapable(rc.comm.get()), GinAnvilSdmaReach::None);
  EXPECT_FALSE(ginAnvilUseFabricPod(rc.comm.get()));
  EXPECT_TRUE(ginAnvilFabricPodVotePredicate(rc.topo->nHosts, rc.comm->nRanks, ginAnvilCrossNic(rc.comm.get())));
}

TEST(GinAnvilReachMicrotestPredicate, FabricPodVoteOnlyAcrossHostsWithinPeerTableOverAFullConnection) {
  EXPECT_FALSE(ginAnvilFabricPodVotePredicate(1, 8, true));
  EXPECT_TRUE(ginAnvilFabricPodVotePredicate(2, 8, true));
  EXPECT_TRUE(ginAnvilFabricPodVotePredicate(4, kGinAnvilIpcMaxRanks, true));
  EXPECT_FALSE(ginAnvilFabricPodVotePredicate(4, kGinAnvilIpcMaxRanks + 1, true));
  EXPECT_FALSE(ginAnvilFabricPodVotePredicate(2, 8, false));
}

TEST(GinAnvilReachMicrotestPredicate, SingleHostAlwaysReached) {
  EXPECT_EQ(ginAnvilSdmaReachPredicate(1, 8, false, false), GinAnvilSdmaReach::Host);
  EXPECT_EQ(ginAnvilSdmaReachPredicate(1, 64, true, true), GinAnvilSdmaReach::Host);
}

TEST(GinAnvilReachMicrotestPredicate, CrossesHostsOnlyInCrossNicFabricPodWithinPeerTable) {
  EXPECT_EQ(ginAnvilSdmaReachPredicate(2, 8, true, true), GinAnvilSdmaReach::FabricPod);
  EXPECT_EQ(ginAnvilSdmaReachPredicate(4, kGinAnvilIpcMaxRanks, true, true), GinAnvilSdmaReach::FabricPod);
  EXPECT_EQ(ginAnvilSdmaReachPredicate(2, 8, false, true), GinAnvilSdmaReach::None);
  EXPECT_EQ(ginAnvilSdmaReachPredicate(2, 8, true, false), GinAnvilSdmaReach::None);
  EXPECT_EQ(ginAnvilSdmaReachPredicate(4, kGinAnvilIpcMaxRanks + 1, true, true), GinAnvilSdmaReach::None);
}

// Each reason the reach is None logs a different line, and a pod the topology
// allows but a rank did not opt in to names the variables to set.
TEST(GinAnvilReachMicrotestPredicate, NoReachReasonNamesTheCause) {
  const std::string noPod = ginAnvilSdmaNoReachReason(false, true, 8);
  const std::string rail = ginAnvilSdmaNoReachReason(true, false, 8);
  const std::string tooMany = ginAnvilSdmaNoReachReason(true, true, kGinAnvilIpcMaxRanks + 1);
  const std::string noOptIn = ginAnvilSdmaNoReachReason(true, true, 8);
  for (const std::string& other : {rail, tooMany, noOptIn}) EXPECT_NE(noPod, other);
  EXPECT_NE(rail, tooMany);
  EXPECT_NE(rail, noOptIn);
  EXPECT_NE(tooMany, noOptIn);
  EXPECT_NE(rail.find("NCCL_CROSS_NIC=0"), std::string::npos);
  EXPECT_NE(noOptIn.find("RCCL_GIN_ANVIL_FABRIC_POD=1"), std::string::npos);
  EXPECT_NE(noOptIn.find("RCCL_GIN_ANVIL_FABRIC_MEM=1"), std::string::npos);
}
