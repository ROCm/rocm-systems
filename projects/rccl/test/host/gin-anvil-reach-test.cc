/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/gin/gin_anvil_reach.cc: which communicators the
// Anvil SDMA GIN backend can serve on MI300, MI350 and MI450, and when it
// registers windows through fabric memory. Built into
// rccl-UnitTestsMicroGinHost; this TU #includes the hipified source
// (GIN_ANVIL_REACH_CC_PATH). No GPU, no librccl.so.

#include <gtest/gtest.h>

#include <memory>
#include <string>

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
    ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;
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
  GinAnvilArch expectedArch;
  GinAnvilSdmaReach expected;
};

class GinAnvilReachMicrotestPerSystem : public GinAnvilReachMicrotest,
                                         public ::testing::WithParamInterface<AnvilReachCase> {};

struct ReachComm {
  std::unique_ptr<ncclComm> comm = std::make_unique<ncclComm>();
  std::unique_ptr<ncclTopoSystem> topo = std::make_unique<ncclTopoSystem>();
  std::string arch;

  ReachComm(const char* archName, int mnnvl, int nRanks, int cliqueSize, int nHosts = 1) : arch(archName) {
    comm->archName = &arch[0];
    comm->MNNVL = mnnvl;
    comm->nRanks = nRanks;
    comm->clique.size = cliqueSize;
    topo->nHosts = nHosts;
    comm->topo = topo.get();
  }
};

// hipMemHandleTypeFabric exists only with HIP_FABRIC_API (absent on the ROCm
// 7.2 host-test image); without it, p2p.h supplies CU_MEM_HANDLE_TYPE_FABRIC.
void useFabricHandles() {
#ifdef HIP_FABRIC_API
  ncclCuMemHandleType = hipMemHandleTypeFabric;
#else
  ncclCuMemHandleType = CU_MEM_HANDLE_TYPE_FABRIC;
#endif
}

}  // namespace

TEST_F(GinAnvilReachMicrotest, NullComm_ReachesNothing) {
  EXPECT_EQ(ginAnvilArch(nullptr), GinAnvilArch::Other);
  EXPECT_EQ(ginAnvilSdmaReachCapable(nullptr), GinAnvilSdmaReach::None);
  EXPECT_FALSE(ginAnvilUseFabricPod(nullptr));
  EXPECT_FALSE(ginAnvilFabricMemCapable(nullptr));
  EXPECT_FALSE(ginAnvilUseFabricMem(nullptr));
  EXPECT_FALSE(ginFabricLsaA2ACapable(nullptr, 4));
}

// Fabric registration needs MNNVL on an MI450, one clique covering the
// communicator, and cuMem; MNNVL on any other arch does not qualify.
TEST_F(GinAnvilReachMicrotest, UseFabricMem_OnlyForMi450SingleCliqueWithCuMem) {
  EXPECT_TRUE(ginAnvilUseFabricMem(ReachComm("gfx1250", 1, 8, 8).comm.get()));
  EXPECT_FALSE(ginAnvilUseFabricMem(ReachComm("gfx1250", 1, 8, 4).comm.get()));
  EXPECT_FALSE(ginAnvilUseFabricMem(ReachComm("gfx1250", 0, 8, 8).comm.get()));
  EXPECT_FALSE(ginAnvilUseFabricMem(ReachComm("gfx950", 1, 8, 8).comm.get()));
  g_cuMemEnable = [] { return 0; };
  EXPECT_FALSE(ginAnvilUseFabricMem(ReachComm("gfx1250", 1, 8, 8).comm.get()));
}

// Off unless RCCL_GIN_ANVIL_FABRIC_MEM=1. Capability ignores the opt-in, so a
// rank that did not opt in still joins the plugin's agreement vote.
TEST_F(GinAnvilReachMicrotest, UseFabricMem_NeedsOptIn) {
  ReachComm rc("gfx1250", 1, 8, 8);
  // Everything else the lane needs is in place, so only the opt-in keeps it off.
  rc.comm->devrState.lsaSize = 8;
  useFabricHandles();
  g_loadParam = [](const char*, int64_t deft) { return deft; };
  EXPECT_FALSE(ginAnvilUseFabricMem(rc.comm.get()));
  EXPECT_TRUE(ginAnvilFabricMemCapable(rc.comm.get()));
  EXPECT_FALSE(ginFabricLsaA2ACapable(rc.comm.get(), 8));
  EXPECT_FALSE(ginAnvilFabricMemCapable(ReachComm("gfx942", 1, 8, 8).comm.get()));
}

// The LSA AllToAll lane also needs fabric handles and a GIN and LSA team that
// both cover the communicator.
TEST_F(GinAnvilReachMicrotest, FabricLsaA2ACapable_NeedsFabricHandlesAndFullTeams) {
  ReachComm rc("gfx1250", 1, 8, 8);
  rc.comm->devrState.lsaSize = 8;
  EXPECT_FALSE(ginFabricLsaA2ACapable(rc.comm.get(), 8));
  useFabricHandles();
  EXPECT_TRUE(ginFabricLsaA2ACapable(rc.comm.get(), 8));
  EXPECT_FALSE(ginFabricLsaA2ACapable(rc.comm.get(), 4));
  rc.comm->devrState.lsaSize = 4;
  EXPECT_FALSE(ginFabricLsaA2ACapable(rc.comm.get(), 8));
}

// An unset or negative threshold reads as -1.
TEST_F(GinAnvilReachMicrotest, FabricLsaEnv_OptInAndThreshold) {
  EXPECT_FALSE(ginAnvilFabricLsaA2AOptIn());
  EXPECT_EQ(ginAnvilFabricLsaThresholdEnv(), -1);
  g_loadParam = ParamOverride({{"RCCL_GIN_ANVIL_FABRIC_LSA_A2A", 1},
                               {"RCCL_GIN_ANVIL_FABRIC_LSA_THRESHOLD_ALLTOALL", 65536}});
  EXPECT_TRUE(ginAnvilFabricLsaA2AOptIn());
  EXPECT_EQ(ginAnvilFabricLsaThresholdEnv(), 65536);
  g_loadParam = ParamOverride({{"RCCL_GIN_ANVIL_FABRIC_LSA_THRESHOLD_ALLTOALL", -5}});
  EXPECT_EQ(ginAnvilFabricLsaThresholdEnv(), -1);
}

TEST_P(GinAnvilReachMicrotestPerSystem, ByArchAndTopology_CrossesHostsOnlyInMi450Pod) {
  const AnvilReachCase c = GetParam();
  ReachComm rc(c.arch, c.mnnvl, c.nRanks, c.cliqueSize, c.nHosts);
  const int cuMem = c.cuMem;
  g_cuMemEnable = [cuMem] { return cuMem; };

  EXPECT_EQ(ginAnvilArch(rc.comm.get()), c.expectedArch);
  EXPECT_EQ(ginAnvilSdmaReachCapable(rc.comm.get()), c.expected);
  EXPECT_EQ(ginAnvilUseFabricPod(rc.comm.get()), c.expected == GinAnvilSdmaReach::FabricPod);
}

INSTANTIATE_TEST_SUITE_P(
    PerSystem, GinAnvilReachMicrotestPerSystem,
    ::testing::Values(
        AnvilReachCase{"Mi300SingleHost", "gfx942:sramecc+:xnack-", 0, 1, 8, 8, 1, GinAnvilArch::Mi300,
                       GinAnvilSdmaReach::Host},
        AnvilReachCase{"Mi300MultiHostForcedMnnvl", "gfx942", 1, 2, 16, 16, 1, GinAnvilArch::Mi300,
                       GinAnvilSdmaReach::None},
        AnvilReachCase{"Mi350SingleHost", "gfx950", 0, 1, 8, 8, 0, GinAnvilArch::Mi350, GinAnvilSdmaReach::Host},
        AnvilReachCase{"Mi350MultiHostForcedMnnvl", "gfx950", 1, 2, 16, 16, 1, GinAnvilArch::Mi350,
                       GinAnvilSdmaReach::None},
        AnvilReachCase{"Mi450SingleHost", "gfx1250", 0, 1, 4, 4, 0, GinAnvilArch::Mi450, GinAnvilSdmaReach::Host},
        AnvilReachCase{"Mi450FabricPod", "gfx1250", 1, 2, 8, 8, 1, GinAnvilArch::Mi450,
                       GinAnvilSdmaReach::FabricPod},
        AnvilReachCase{"Mi450TwoCliques", "gfx1250", 1, 2, 8, 4, 1, GinAnvilArch::Mi450, GinAnvilSdmaReach::None},
        AnvilReachCase{"Mi450NoCuMem", "gfx1250", 1, 2, 8, 8, 0, GinAnvilArch::Mi450, GinAnvilSdmaReach::None},
        AnvilReachCase{"Mi450NoMnnvl", "gfx1250", 0, 2, 8, 8, 1, GinAnvilArch::Mi450, GinAnvilSdmaReach::None},
        AnvilReachCase{"Mi450PodAboveRankCap", "gfx1250", 1, 4, kGinAnvilPodMaxRanks + 1,
                       kGinAnvilPodMaxRanks + 1, 1, GinAnvilArch::Mi450, GinAnvilSdmaReach::None},
        AnvilReachCase{"OtherArchMultiHost", "gfx90a", 1, 2, 8, 8, 1, GinAnvilArch::Other,
                       GinAnvilSdmaReach::None}),
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
