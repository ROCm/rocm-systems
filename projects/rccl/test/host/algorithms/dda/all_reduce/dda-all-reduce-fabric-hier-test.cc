/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the eligibility check of src/algorithms/dda/all_reduce/dda_all_reduce_fabric_hier.cu, the
// fused host-hierarchical AllReduce kernel.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include DDA_ALL_REDUCE_FABRIC_HIER_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {
int64_t g_ddaEnable = 1;
}  // namespace

int64_t rcclParamDdaEnable() { return g_ddaEnable; }

namespace {

using RcclUnitTesting::DdaFabricMockComm;

// One 16-byte vector per local rank: 8 ranks per host take fp32 counts in steps of 32 and 16-bit ones in steps of 64.
constexpr size_t kFp32Granule = 32;
constexpr size_t kHalfGranule = 64;

class DdaAllReduceFabricHierMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { comm().ddaScratchBytes = size_t{1} << 30; }

  void TearDown() override {
    g_ddaEnable = 1;
    ResetDdaCollSeams();
    ResetHipFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(size_t count, ncclDataType_t datatype = ncclFloat32, int nHosts = 2, int ranksPerHost = 8,
                ncclRedOp_t op = ncclSum) {
    return ncclAllReduceDdaFabricHierEligible(mock_.get(), count, datatype, op, nHosts, ranksPerHost);
  }

  DdaFabricMockComm mock_;
};

TEST_F(DdaAllReduceFabricHierMicrotest, Eligible_SumOnFp32Fp16Bf16) {
  EXPECT_TRUE(Eligible(kFp32Granule));
  EXPECT_TRUE(Eligible(kHalfGranule, ncclFloat16));
  EXPECT_TRUE(Eligible(kHalfGranule, ncclBfloat16));
  for (ncclRedOp_t op : {ncclProd, ncclMax, ncclMin, ncclAvg}) {
    EXPECT_FALSE(Eligible(kFp32Granule, ncclFloat32, 2, 8, op)) << "op " << op;
  }
  for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64,
                                  ncclFloat8e4m3, ncclFloat8e5m2}) {
    EXPECT_FALSE(Eligible(1024, datatype)) << "datatype " << datatype;
  }
}

TEST_F(DdaAllReduceFabricHierMicrotest, Eligible_NeedsCommAndCount) {
  EXPECT_FALSE(ncclAllReduceDdaFabricHierEligible(nullptr, kFp32Granule, ncclFloat32, ncclSum, 2, 8));
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaAllReduceFabricHierMicrotest, Eligible_DdaGates) {
  g_ddaEnable = 0;
  EXPECT_FALSE(Eligible(kFp32Granule)) << "RCCL_DDA_ENABLE=0";
  g_ddaEnable = 1;
  comm().config.launchOrderImplicit = 1;
  EXPECT_FALSE(Eligible(kFp32Granule)) << "implicit launch order";
  comm().config.launchOrderImplicit = 0;
  comm().ddaDisabledAfterAbort = true;
  EXPECT_FALSE(Eligible(kFp32Granule)) << "DDA disabled after an abort";
  comm().ddaDisabledAfterAbort = false;
  EXPECT_TRUE(Eligible(kFp32Granule));
}

TEST_F(DdaAllReduceFabricHierMicrotest, Eligible_EachMissingFabricResourceRejects) {
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setFabricResourcesPresent(true);
    if (missing == 0) comm().ddaFabricMemHandler = nullptr;
    if (missing == 1) comm().ddaFabricBarrierState = nullptr;
    if (missing == 2) comm().ddaScratch = nullptr;
    if (missing == 3) comm().ddaPeerPtrsDev = nullptr;
    EXPECT_FALSE(Eligible(kFp32Granule)) << "missing resource " << missing;
  }
}

TEST_F(DdaAllReduceFabricHierMicrotest, Eligible_HostAndRankCaps) {
  constexpr size_t count = 16 * 16 * 4;
  EXPECT_FALSE(Eligible(count, ncclFloat32, 0, 8));
  EXPECT_TRUE(Eligible(count, ncclFloat32, 1, 8));
  EXPECT_TRUE(Eligible(count, ncclFloat32, 16, 8));
  EXPECT_FALSE(Eligible(count, ncclFloat32, 17, 8));
  EXPECT_FALSE(Eligible(count, ncclFloat32, 2, 1));
  EXPECT_TRUE(Eligible(count, ncclFloat32, 2, 2));
  EXPECT_TRUE(Eligible(count, ncclFloat32, 2, 16));
  EXPECT_FALSE(Eligible(17 * count, ncclFloat32, 2, 17));
}

TEST_F(DdaAllReduceFabricHierMicrotest, Eligible_CountInWholeVectorsPerLocalRank) {
  EXPECT_FALSE(Eligible(kFp32Granule / 2));
  EXPECT_FALSE(Eligible(kFp32Granule + 4));
  EXPECT_TRUE(Eligible(3 * kFp32Granule));
  EXPECT_FALSE(Eligible(kFp32Granule, ncclFloat16));
  EXPECT_FALSE(Eligible(kHalfGranule + 8, ncclBfloat16));
  EXPECT_TRUE(Eligible(kFp32Granule, ncclFloat32, 2, 8));
  EXPECT_TRUE(Eligible(16, ncclFloat32, 2, 4)) << "4 ranks per host take fp32 counts in steps of 16";
}

// Two banks of [8 host | 2 inter-host | 8 gathered] slots must hold at least one granule per round: 2 * 32 * 18 / 8
// fp32 elements, 576 bytes.
TEST_F(DdaAllReduceFabricHierMicrotest, Eligible_ScratchHoldsOneGranulePerRound) {
  comm().ddaScratchBytes = 576;
  EXPECT_TRUE(Eligible(1024));
  comm().ddaScratchBytes = 575;
  EXPECT_FALSE(Eligible(1024));
}

}  // namespace
