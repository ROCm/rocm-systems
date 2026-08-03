/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/DdaFabricTestHelpers.hpp"

#include "dda_all_gather.h"
#include "dda_all_reduce.h"
#include "dda_alltoall.h"
#include "dda_reduce_scatter.h"
#include "fabric_gpu_barrier.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{

class DdaFabricEligibilityTest : public ::testing::Test
{
protected:
    DdaFabricMockComm mockComm_;
    void*             sendbuff_{reinterpret_cast<void*>(0x1000)};
    void*             recvbuff_{reinterpret_cast<void*>(0x2000)};
};

// ---------------------------------------------------------------------------
// Scratch sizing
// ---------------------------------------------------------------------------

TEST(DdaFabricScratchSizingTest, ExplicitOverrideTakesPrecedence)
{
    const auto forced = nccl_dda_detail::ddaFabricScratchSizing(8, 4096, 0, 0, 0, 0);
    EXPECT_EQ(forced.bytes, 4096u);

    const auto disabled =
        nccl_dda_detail::ddaFabricScratchSizing(8, 0, 1, 128 * 1024 * 1024, 1, 32 * 1024 * 1024);
    EXPECT_EQ(disabled.bytes, 0u);
}

TEST(DdaFabricScratchSizingTest, DisabledDdaHasNoDerivedAllocation)
{
    const auto sizing =
        nccl_dda_detail::ddaFabricScratchSizing(8, -1, 0, 128 * 1024 * 1024, 1, 32 * 1024 * 1024);
    EXPECT_EQ(sizing.bytes, 0u);
    EXPECT_EQ(sizing.effectiveLL128Threshold, 0u);
}

TEST(DdaFabricScratchSizingTest, ZeroOverallThresholdHasNoDerivedAllocation)
{
    const auto sizing = nccl_dda_detail::ddaFabricScratchSizing(8, -1, 1, 0, 1, 32 * 1024 * 1024);
    EXPECT_EQ(sizing.bytes, 0u);
    EXPECT_EQ(sizing.effectiveLL128Threshold, 0u);
}

TEST(DdaFabricScratchSizingTest, DisabledLL128UsesSimpleCapacity)
{
    constexpr int64_t simpleThreshold = 128 * 1024 * 1024;
    const auto sizing = nccl_dda_detail::ddaFabricScratchSizing(8, -1, 1, simpleThreshold, 0, 32 * 1024 * 1024);
    EXPECT_EQ(sizing.bytes, (size_t)simpleThreshold + (size_t)simpleThreshold / 8);
    EXPECT_EQ(sizing.effectiveLL128Threshold, 0u);
}

TEST(DdaFabricScratchSizingTest, LL128ThresholdIsCappedByOverallThreshold)
{
    constexpr int64_t overallThreshold = 64 * 1024 * 1024;
    const auto oversized =
        nccl_dda_detail::ddaFabricScratchSizing(8, -1, 1, overallThreshold, 1, 128 * 1024 * 1024);
    const auto atCap =
        nccl_dda_detail::ddaFabricScratchSizing(8, -1, 1, overallThreshold, 1, overallThreshold);
    EXPECT_EQ(oversized.effectiveLL128Threshold, (size_t)overallThreshold);
    EXPECT_EQ(oversized.bytes, atCap.bytes);
}

TEST(DdaFabricScratchSizingTest, LL128ThresholdIsCappedByPathLimit)
{
    constexpr int64_t twoGiB = 2LL * 1024 * 1024 * 1024;
    const auto oversized = nccl_dda_detail::ddaFabricScratchSizing(8, -1, 1, twoGiB, 1, twoGiB);
    const auto atCap = nccl_dda_detail::ddaFabricScratchSizing(
        8, -1, 1, twoGiB, 1, nccl_dda_detail::kDdaLL128ArMaxBytes);
    EXPECT_EQ(oversized.effectiveLL128Threshold, nccl_dda_detail::kDdaLL128ArMaxBytes);
    EXPECT_EQ(oversized.bytes, atCap.bytes);
}

// ---------------------------------------------------------------------------
// AllGather
// ---------------------------------------------------------------------------

TEST_F(DdaFabricEligibilityTest, AllGather_NullComm)
{
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        nullptr, sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_MissingBootstrap)
{
    mockComm_.comm.bootstrap = nullptr;
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_MissingFabricResources)
{
    mockComm_.setFabricResourcesPresent(false);
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_MissingBarrierState)
{
    mockComm_.comm.ddaFabricBarrierState = nullptr;
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_MissingScratch)
{
    mockComm_.comm.ddaScratch = nullptr;
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_ZeroCount)
{
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_TooFewRanks)
{
    mockComm_.comm.nRanks = 1;
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_TooManyRanks)
{
    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks + 1;
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_MaxRanksEligible)
{
    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks;
    EXPECT_TRUE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclInt32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_ScratchTooSmall)
{
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_UnalignedCount)
{
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_EligibleFloat16)
{
    EXPECT_TRUE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8, ncclFloat16));
}

TEST_F(DdaFabricEligibilityTest, AllGather_EligibleBfloat16)
{
    EXPECT_TRUE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8, ncclBfloat16));
}

// Unlike the IPC path, the fabric path spans an MNNVL clique, so multi-node
// comms stay eligible (nNodes is not part of the eligibility check).
TEST_F(DdaFabricEligibilityTest, AllGather_MultiNodeStillEligible)
{
    mockComm_.comm.nNodes = 2;
    EXPECT_TRUE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclAllGatherDdaFabric(
                  sendbuff_, recvbuff_, 4, ncclInt32, mockComm_.get(), nullptr),
              ncclInvalidArgument);
}

// ---------------------------------------------------------------------------
// AllReduce
// ---------------------------------------------------------------------------

TEST_F(DdaFabricEligibilityTest, AllReduce_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_EligibleBfloat16)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8, ncclBfloat16, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_UnsupportedOp)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclProd));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclInt32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_ZeroCount)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_UnalignedCount)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_ScratchTooSmall)
{
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_MinRanksEligible)
{
    mockComm_.comm.nRanks = 2;
    EXPECT_TRUE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_TooManyRanks)
{
    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks + 1;
    EXPECT_FALSE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

// Above the flat/tree threshold (256KB) the two-shot path requires count to be
// divisible by nRanks with a 16-byte-aligned per-rank slice. count=131072
// (512KB f32, nRanks=8) satisfies both.
TEST_F(DdaFabricEligibilityTest, AllReduce_TreePathEligible)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 131072, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclAllReduceDdaFabric(sendbuff_,
                                     recvbuff_,
                                     4,
                                     ncclInt32,
                                     ncclSum,
                                     mockComm_.get(),
                                     nullptr),
              ncclInvalidArgument);
}

// ---------------------------------------------------------------------------
// AllToAll
// ---------------------------------------------------------------------------

TEST_F(DdaFabricEligibilityTest, AllToAll_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllToAllDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllToAll_ZeroCount)
{
    EXPECT_FALSE(ncclAllToAllDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllToAll_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclAllToAllDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclInt32));
}

// Scratch must hold count*nRanks elements (the full exchange), not just count.
TEST_F(DdaFabricEligibilityTest, AllToAll_ScratchTooSmallForTotal)
{
    mockComm_.comm.ddaScratchBytes = 64;
    EXPECT_FALSE(ncclAllToAllDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllToAll_UnalignedCount)
{
    EXPECT_FALSE(ncclAllToAllDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllToAll_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclAllToAllDdaFabric(
                  sendbuff_, recvbuff_, 4, ncclInt32, mockComm_.get(), nullptr),
              ncclInvalidArgument);
}

// ---------------------------------------------------------------------------
// ReduceScatter
// ---------------------------------------------------------------------------

TEST_F(DdaFabricEligibilityTest, ReduceScatter_EligibleFloat32)
{
    EXPECT_TRUE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, ReduceScatter_UnsupportedOp)
{
    EXPECT_FALSE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclMax));
}

TEST_F(DdaFabricEligibilityTest, ReduceScatter_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclInt32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, ReduceScatter_ZeroCount)
{
    EXPECT_FALSE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32, ncclSum));
}

// recvcount=3 keeps the total (recvcount*nRanks) 16-byte aligned but leaves the
// per-rank slice unaligned, exercising the per-rank alignment guard.
TEST_F(DdaFabricEligibilityTest, ReduceScatter_PerRankUnaligned)
{
    EXPECT_FALSE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, ReduceScatter_ScratchTooSmall)
{
    mockComm_.comm.ddaScratchBytes = 64;
    EXPECT_FALSE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, ReduceScatter_MaxRanksEligible)
{
    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks;
    EXPECT_TRUE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, ReduceScatter_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclReduceScatterDdaFabric(sendbuff_,
                                         recvbuff_,
                                         4,
                                         ncclInt32,
                                         ncclSum,
                                         mockComm_.get(),
                                         nullptr),
              ncclInvalidArgument);
}

} // namespace RcclUnitTesting
