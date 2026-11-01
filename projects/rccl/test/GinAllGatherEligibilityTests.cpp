/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cstring>

#include "algorithms/gin/gin_all_gather.h"
#include "comm.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

#if defined(ENABLE_ROCSHMEM_GIN)
namespace {

// This fixture deliberately stops at the window lookup: each test changes one
// earlier eligibility gate and does not need a GPU/device-runtime allocation.
struct GinAllGatherMockComm {
  ncclComm comm{};
  ncclSharedResources sharedRes{};
  char bootstrapPlaceholder{0};

  GinAllGatherMockComm() {
    std::memset(&comm, 0, sizeof(comm));
    comm.bootstrap = &bootstrapPlaceholder;
    comm.nNodes = 1;
    comm.nRanks = 8;
    comm.rank = 0;
    comm.symmetricSupport = true;
    comm.globalGinSupport = NCCL_GIN_CONNECTION_FULL;
    comm.devrState.bigSize = 1;
    comm.devrState.lsaSize = comm.nRanks;
    comm.devrState.lsaSelf = comm.rank;
    sharedRes.ginState.numActiveBackends = 1;
    sharedRes.ginState.backends[0].ginType = (ncclGinType_t)NCCL_NET_DEVICE_GIN_ANVIL_SDMA;
    comm.sharedRes = &sharedRes;
  }
};

class GinAllGatherEligibilityTest : public ::testing::Test {
 protected:
  GinAllGatherMockComm mock_;
  void* send_{reinterpret_cast<void*>(0x1000)};
  void* recv_{reinterpret_cast<void*>(0x2000)};
  static constexpr size_t kCount = (8ull * 1024 * 1024) / sizeof(float);

  bool eligible(size_t count = kCount) {
    return ncclAllGatherGinSdmaEligible(&mock_.comm, send_, recv_, count, ncclFloat32);
  }
};
} // namespace

TEST_F(GinAllGatherEligibilityTest, RejectsNullOrZeroInputs) {
  EXPECT_FALSE(ncclAllGatherGinSdmaEligible(nullptr, send_, recv_, kCount, ncclFloat32));
  EXPECT_FALSE(eligible(0));
}

TEST_F(GinAllGatherEligibilityTest, RejectsMissingBootstrapAndMultiNode) {
  mock_.comm.bootstrap = nullptr;
  EXPECT_FALSE(eligible());
  mock_.comm.bootstrap = &mock_.bootstrapPlaceholder;
  mock_.comm.nNodes = 2;
  EXPECT_FALSE(eligible());
}

TEST_F(GinAllGatherEligibilityTest, RejectsUnsupportedTopologyAndBackend) {
  mock_.comm.nRanks = 32;
  mock_.comm.devrState.lsaSize = 32;
  EXPECT_FALSE(eligible());
  mock_.comm.nRanks = 8;
  mock_.comm.devrState.lsaSize = 8;
  mock_.sharedRes.ginState.backends[0].ginType = (ncclGinType_t)NCCL_NET_DEVICE_GIN_PROXY;
  EXPECT_FALSE(eligible());
}

TEST_F(GinAllGatherEligibilityTest, RejectsNonSymmetricAndPartialLsa) {
  mock_.comm.symmetricSupport = false;
  EXPECT_FALSE(eligible());
  mock_.comm.symmetricSupport = true;
  mock_.comm.devrState.lsaSize = 4;
  EXPECT_FALSE(eligible());
}

TEST_F(GinAllGatherEligibilityTest, RequiresRegisteredBuffers) {
  ASSERT_EQ(mock_.comm.devrState.winSortedCount, 0);
  EXPECT_FALSE(eligible());
}
#else
TEST(GinAllGatherEligibilityTest, SkippedWithoutGinBuild) {
  GTEST_SKIP() << "GIN AllGather eligibility tests require ENABLE_ROCSHMEM_GIN";
}
#endif

} // namespace RcclUnitTesting
