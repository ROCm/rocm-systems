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
    const size_t forced = nccl_dda_detail::ddaFabricScratchSizing(8, 4096, 0, 0, 0, 0);
    EXPECT_EQ(forced, 4096u);

    const size_t disabled =
        nccl_dda_detail::ddaFabricScratchSizing(8, 0, 1, 128 * 1024 * 1024, 1, 1);
    EXPECT_EQ(disabled, 0u);
}

TEST(DdaFabricScratchSizingTest, DisabledDdaHasNoDerivedAllocation)
{
    const size_t sizing =
        nccl_dda_detail::ddaFabricScratchSizing(8, -1, 0, 128 * 1024 * 1024, 1, 1);
    EXPECT_EQ(sizing, 0u);
}

TEST(DdaFabricScratchSizingTest, ZeroOverallThresholdHasNoDerivedAllocation)
{
    const size_t sizing = nccl_dda_detail::ddaFabricScratchSizing(8, -1, 1, 0, 1, 1);
    EXPECT_EQ(sizing, 0u);
}

TEST(DdaFabricScratchSizingTest, DisabledLL128UsesSimpleCapacity)
{
    constexpr int64_t simpleThreshold = 128 * 1024 * 1024;
    // With LL and LL128 both disabled, scratch = simpleCap (the threshold).
    const size_t sizing = nccl_dda_detail::ddaFabricScratchSizing(8, -1, 1, simpleThreshold, 0, 0);
    EXPECT_EQ(sizing, (size_t)simpleThreshold);
}

TEST(DdaFabricScratchSizingTest, LL128FloorDominatesWhenLargerThanSimpleCap)
{
    // LL128 floor = 2 * 8 * 4370 * 128 = 8,949,760 bytes (~8.5 MiB)
    // If simpleCap is smaller than the LL128 floor, scratch = LL128 floor.
    constexpr int64_t smallSimpleCap = 4 * 1024 * 1024;  // 4 MiB
    const size_t sizing = nccl_dda_detail::ddaFabricScratchSizing(8, -1, 1, smallSimpleCap, 0, 1);
    constexpr size_t ll128Floor = 2 * 8 * nccl_dda_detail::kDdaLL128SlotStrideLines * 128;
    EXPECT_EQ(sizing, ll128Floor);
}

TEST(DdaFabricScratchSizingTest, LargeSimpleCapDominatesWhenEnabled)
{
    // If simpleCap > LL/LL128 floors, scratch = simpleCap.
    constexpr int64_t largeSimpleCap = 512 * 1024 * 1024;  // 512 MiB
    const size_t sizing = nccl_dda_detail::ddaFabricScratchSizing(8, -1, 1, largeSimpleCap, 1, 1);
    EXPECT_EQ(sizing, (size_t)largeSimpleCap);
}

TEST(DdaFabricScratchSizingTest, LLFloorDominatesAtHighRankCount)
{
    // At 72 ranks with small threshold, LL floor should dominate.
    // LL floor = 2 * 72 * 16384 * 16 = 37,748,736 (~36 MiB)
    constexpr int nRanks = 72;
    constexpr int64_t smallSimpleCap = 8 * 1024 * 1024;  // 8 MiB
    const size_t sizing = nccl_dda_detail::ddaFabricScratchSizing(nRanks, -1, 1, smallSimpleCap, 1, 0);
    constexpr size_t llFloor = 2 * nRanks * nccl_dda_detail::kDdaLLSlotStridePkts * 16;
    EXPECT_EQ(sizing, llFloor);
    EXPECT_GT(sizing, (size_t)smallSimpleCap);
}

TEST(DdaFabricScratchSizingTest, LL128FloorDominatesAtHighRankCount)
{
    // At 72 ranks with small threshold, LL128 floor should dominate.
    // LL128 floor = 2 * 72 * 4370 * 128 = 80,609,280 (~77 MiB)
    constexpr int nRanks = 72;
    constexpr int64_t smallSimpleCap = 8 * 1024 * 1024;  // 8 MiB
    const size_t sizing = nccl_dda_detail::ddaFabricScratchSizing(nRanks, -1, 1, smallSimpleCap, 0, 1);
    constexpr size_t ll128Floor = 2 * nRanks * nccl_dda_detail::kDdaLL128SlotStrideLines * 128;
    EXPECT_EQ(sizing, ll128Floor);
    EXPECT_GT(sizing, (size_t)smallSimpleCap);
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

TEST_F(DdaFabricEligibilityTest, ReduceScatter_DerivedScratchAcceptsSmallMessage)
{
    // Use derived scratch sizing with default params (8 ranks, 128 MiB threshold, LL+LL128 enabled).
    constexpr int64_t defaultThreshold = 128 * 1024 * 1024;
    mockComm_.comm.ddaScratchBytes = nccl_dda_detail::ddaFabricScratchSizing(
        mockComm_.comm.nRanks, -1, 1, defaultThreshold, 1, 1);
    EXPECT_TRUE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllToAll_DerivedScratchRejectsOversizedMessage)
{
    // Use derived scratch with small threshold so LL128 floor dominates.
    constexpr int64_t smallThreshold = 4 * 1024 * 1024;  // 4 MiB
    const size_t derivedScratch = nccl_dda_detail::ddaFabricScratchSizing(
        mockComm_.comm.nRanks, -1, 1, smallThreshold, 0, 1);
    mockComm_.comm.ddaScratchBytes = derivedScratch;
    // LL128 floor at 8 ranks = ~8.5 MiB. Large message exceeds derived scratch.
    const size_t largeCount = 2 * 1024 * 1024;  // 2M elements
    EXPECT_FALSE(ncclAllToAllDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, largeCount, ncclFloat32));
}

// LL128 AllReduce uses compact layout (scratch scales with message size).
// Small message needs less scratch than the fixed floor.
TEST_F(DdaFabricEligibilityTest, AllReduce_LL128CompactLayoutSmallMessageFits)
{
    // Give scratch smaller than LL128 fixed floor but enough for a small message.
    // LL128 floor at 8 ranks = 2 * 8 * 4370 * 128 = ~8.5 MiB.
    // LL128 AR compact: 2 * nRanks * numLines * 128 for the message.
    // Small message (1024 floats = 4KB) needs much less.
    mockComm_.comm.ddaScratchBytes = 1 * 1024 * 1024;  // 1 MiB - less than fixed floor
    EXPECT_TRUE(ncclAllReduceDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

// LL128 AllGather uses fixed layout (needs full floor regardless of message size).
// Same small scratch that works for AR should fail for AG.
TEST_F(DdaFabricEligibilityTest, AllGather_LL128FixedLayoutNeedsFullFloor)
{
    // Same 1 MiB scratch - smaller than the 8.5 MiB LL128 fixed floor.
    // Even a tiny message should fail because AG needs the full slot array.
    mockComm_.comm.ddaScratchBytes = 1 * 1024 * 1024;  // 1 MiB
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
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
