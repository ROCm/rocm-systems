/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/gin/gin_anvil_reach.cc: when the Anvil SDMA GIN
// backend registers windows through fabric memory. Built into
// rccl-UnitTestsMicroGinHost; this TU #includes the hipified source
// (GIN_ANVIL_REACH_CC_PATH). No GPU, no librccl.so.

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "comm.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"

#include GIN_ANVIL_REACH_CC_PATH

namespace {

class GinAnvilReachMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_cuMemEnable = [] { return 1; };
    g_loadParam = ParamOverride({{"RCCL_GIN_ANVIL_FABRIC_MEM", 1}});
  }
  void TearDown() override {
    g_cuMemEnable = [] { return 0; };
    g_loadParam = [](const char*, int64_t deft) { return deft; };
    ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;
  }
};

struct ReachComm {
  std::unique_ptr<ncclComm> comm = std::make_unique<ncclComm>();
  std::string arch;

  ReachComm(const char* archName, int mnnvl, int nRanks, int cliqueSize) : arch(archName) {
    comm->archName = &arch[0];
    comm->MNNVL = mnnvl;
    comm->nRanks = nRanks;
    comm->clique.size = cliqueSize;
  }
};

}  // namespace

TEST_F(GinAnvilReachMicrotest, NullComm_NoFabricMem) {
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
  // hipMemHandleTypeFabric exists only with HIP_FABRIC_API (absent on the ROCm
  // 7.2 host-test image); without it, p2p.h supplies CU_MEM_HANDLE_TYPE_FABRIC.
#ifdef HIP_FABRIC_API
  ncclCuMemHandleType = hipMemHandleTypeFabric;
#else
  ncclCuMemHandleType = CU_MEM_HANDLE_TYPE_FABRIC;
#endif
  EXPECT_TRUE(ginFabricLsaA2ACapable(rc.comm.get(), 8));
  EXPECT_FALSE(ginFabricLsaA2ACapable(rc.comm.get(), 4));
  rc.comm->devrState.lsaSize = 4;
  EXPECT_FALSE(ginFabricLsaA2ACapable(rc.comm.get(), 8));
}
