/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/DdaFabricTestHelpers.hpp"

#include "algorithms/dda/device/CollCommon_ll128.h"
#include "algorithms/dda/all_gather/dda_all_gather.h"
#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/alltoall/dda_alltoall.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
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
    // The LL128 floor carries the threshold in whole slices: at 8 ranks with a
    // 32 MiB threshold that is 2 * 8 * ceil(4 MiB / 1920) * 2048 = ~68 MiB.
    constexpr int nRanks = 8;
    constexpr int64_t smallSimpleCap = 4 * 1024 * 1024;   // 4 MiB
    constexpr int64_t ll128Threshold = 32 * 1024 * 1024;  // 32 MiB
    const size_t sizing =
        nccl_dda_detail::ddaFabricScratchSizing(nRanks, -1, 1, smallSimpleCap, 0, 1, ll128Threshold);
    const size_t ll128Floor = (size_t)2 * nRanks *
        nccl_dda_detail::ddaLL128Slices((size_t)ll128Threshold / nRanks) *
        nccl_dda_detail::kDdaLL128WireBytesPerSlice;
    EXPECT_EQ(sizing, ll128Floor);
    EXPECT_GT(sizing, (size_t)smallSimpleCap);
}

// AllGather LL128 gates on RCCL_DDA_LL, not RCCL_DDA_LL128, so the LL128 floor
// has to apply with only the LL flag set. Two ranks keeps the LL floor at 64 MiB
// so the threshold-derived floor is what shows through.
TEST(DdaFabricScratchSizingTest, LL128FloorArmedByLLFlagAlone)
{
    constexpr int nRanks = 2;
    constexpr int64_t smallSimpleCap = 8 * 1024 * 1024;   // 8 MiB
    constexpr int64_t ll128Threshold = 64 * 1024 * 1024;  // 64 MiB
    const size_t sizing =
        nccl_dda_detail::ddaFabricScratchSizing(nRanks, -1, 1, smallSimpleCap, 1, 0, ll128Threshold);
    const size_t ll128Floor = (size_t)2 * nRanks *
        nccl_dda_detail::ddaLL128Slices((size_t)ll128Threshold / nRanks) *
        nccl_dda_detail::kDdaLL128WireBytesPerSlice;
    EXPECT_EQ(sizing, ll128Floor);
    EXPECT_GT(sizing, (size_t)2 * nRanks * dda::common::kDdaLLMaxBytes);
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
    constexpr size_t llFloor = 2 * nRanks * dda::common::kDdaLLMaxBytes;
    EXPECT_EQ(sizing, llFloor);
    EXPECT_GT(sizing, (size_t)smallSimpleCap);
}

TEST(DdaFabricScratchSizingTest, LL128FloorDominatesAtHighRankCount)
{
    // At 72 ranks the per-rank share of the threshold is small, but 2 banks of 72
    // slots still outweigh a small simpleCap. The per-rank share rounds up, since
    // 72 does not divide the threshold evenly.
    constexpr int nRanks = 72;
    constexpr int64_t smallSimpleCap = 8 * 1024 * 1024;   // 8 MiB
    constexpr int64_t ll128Threshold = 32 * 1024 * 1024;  // 32 MiB
    const size_t sizing =
        nccl_dda_detail::ddaFabricScratchSizing(nRanks, -1, 1, smallSimpleCap, 0, 1, ll128Threshold);
    const size_t perRank = ((size_t)ll128Threshold + nRanks - 1) / nRanks;
    const size_t ll128Floor = (size_t)2 * nRanks * nccl_dda_detail::ddaLL128Slices(perRank) *
        nccl_dda_detail::kDdaLL128WireBytesPerSlice;
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
    mockComm_.comm.nRanks = dda::common::kDdaMaxNranks + 1;
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_MaxRanksEligible)
{
    mockComm_.comm.nRanks = dda::common::kDdaMaxNranks;
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
    mockComm_.comm.nRanks = dda::common::kDdaMaxNranks + 1;
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
// AllReduce LL
//
// One gate fronts both LL tiers: ncclAllReduceDdaFabricLLEligible reports
// whether the one-shot or the two-shot variant claims the message. Both are
// enabled by DDA_LL and each carries its own threshold on the total message
// size, with one-shot tested first.
//
// These run under the default parameter values -- DDA_LL=1, a 1 MiB
// DDA_LL_ONESHOT_THRESHOLD and a 16 MiB DDA_LL_TWOSHOT_THRESHOLD -- so sizes
// above 1 MiB and up to 16 MiB are claimed by the two-shot tier, which is what
// makes the tier split observable from here. They deliberately do not setenv:
// RCCL_PARAM caches on first read, and the NCCL_NO_CACHE opt-out is resolved once
// per process, so a test that changed a threshold would depend on being the first
// in the binary to touch it. Two guards stay out of reach as a result:
// kDdaLLMaxBytes sits above the one-shot threshold, and the two-shot half-slot
// bound needs a shard past kDdaLLMaxBytes / 2, so both are masked by the
// thresholds and belong to a multi-rank run.
// ---------------------------------------------------------------------------

TEST_F(DdaFabricEligibilityTest, AllReduceLL_NullComm)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        nullptr, sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_MissingBootstrap)
{
    mockComm_.comm.bootstrap = nullptr;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_MissingFabricResources)
{
    mockComm_.setFabricResourcesPresent(false);
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_MissingScratch)
{
    mockComm_.comm.ddaScratch = nullptr;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_ZeroCount)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_UnsupportedOp)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclProd));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclInt32, ncclSum));
}

// Both tiers store a 16-byte LL line at a time, so one-shot requires the message
// to be a multiple of 16 bytes and two-shot requires that of each shard. One float
// is 4 bytes, so a single element satisfies neither.
TEST_F(DdaFabricEligibilityTest, AllReduceLL_UnalignedBytes)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1, ncclFloat32, ncclSum));
}

// 8 bytes is a whole number of 8-byte payload packets but not a whole number of
// 16-byte lines, so it is the size that separates the line rule from the packet
// rule: one-shot rejects it on bytes % 16, and two-shot on a 1-byte shard.
TEST_F(DdaFabricEligibilityTest, AllReduceLL_PacketButNotLineMultiple)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 2, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_TooFewRanks)
{
    mockComm_.comm.nRanks = 1;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_TooManyRanks)
{
    mockComm_.comm.nRanks = dda::common::kDdaMaxNranks + 1;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_MinRanksEligible)
{
    mockComm_.comm.nRanks = 2;
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

// The staging footprint scales with nRanks, so the widest comm is also the one
// that has to still fit the scratch.
TEST_F(DdaFabricEligibilityTest, AllReduceLL_MaxRanksEligible)
{
    mockComm_.comm.nRanks = dda::common::kDdaMaxNranks;
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_ScratchTooSmall)
{
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_EligibleFloat16)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8, ncclFloat16, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_EligibleBfloat16)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8, ncclBfloat16, ncclSum));
}

// 262144 floats is exactly the default 1 MiB DDA_LL_ONESHOT_THRESHOLD, which the
// tier takes with <=.
TEST_F(DdaFabricEligibilityTest, AllReduceLL_AtOneShotThresholdEligible)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 262144, ncclFloat32, ncclSum));
}

// 262176 floats is 1 MiB + 128 bytes: past the one-shot threshold, so one-shot
// cannot be what accepts it, and a multiple of 16 * nRanks so the shard satisfies
// the two-shot shape rules. Accepting it is the two-shot tier serving the range
// above the one-shot threshold.
TEST_F(DdaFabricEligibilityTest, AllReduceLL_TwoShotClaimsPastOneShotThreshold)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 262176, ncclFloat32, ncclSum));
}

// 4194304 floats is exactly the default 16 MiB DDA_LL_TWOSHOT_THRESHOLD, the top
// of the LL range, which the tier takes with <=.
TEST_F(DdaFabricEligibilityTest, AllReduceLL_AtTwoShotThresholdEligible)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4194304, ncclFloat32, ncclSum));
}

// 128 bytes past the two-shot threshold: neither tier claims it and the message
// falls through to LL128 / Simple.
TEST_F(DdaFabricEligibilityTest, AllReduceLL_PastBothThresholds)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4194336, ncclFloat32, ncclSum));
}

// The gate above only reports the disjunction of the two tiers. These call each
// per-variant predicate directly, so a tier can be pinned without the other
// masking it, and assert the complementary tier to show which one owns the size.

// 16 bytes is inside the one-shot threshold and a whole line, so one-shot takes
// it; a quarter of it is a 4-byte shard, which is not, so two-shot cannot.
TEST_F(DdaFabricEligibilityTest, AllReduceLL_OneShotEligible)
{
    mockComm_.comm.nRanks = 4;
    EXPECT_TRUE(ddaLLArOneShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
    EXPECT_FALSE(ddaLLArTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

// 262160 floats is 1 MiB + 64 bytes: past the one-shot threshold, so that tier
// declines, while the 262160-byte shard is a whole number of lines and within the
// slot's payload capacity, so two-shot takes it.
TEST_F(DdaFabricEligibilityTest, AllReduceLL_TwoShotEligible)
{
    mockComm_.comm.nRanks = 4;
    EXPECT_TRUE(ddaLLArTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 262160, ncclFloat32, ncclSum));
    EXPECT_FALSE(ddaLLArOneShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 262160, ncclFloat32, ncclSum));
}

// The shard is carved out of the byte count, so 32 halves (64 bytes) divide evenly
// across 4 ranks but 33 (66 bytes) do not. The pair differs only in that, which is
// what makes the rejection attributable to the divisibility guard rather than to
// the 16-byte rule that follows it.
TEST_F(DdaFabricEligibilityTest, AllReduceLL_TwoShotRejectsUnevenShard)
{
    mockComm_.comm.nRanks = 4;
    EXPECT_TRUE(ddaLLArTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 32, ncclFloat16, ncclSum));
    EXPECT_FALSE(ddaLLArTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 33, ncclFloat16, ncclSum));
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
    mockComm_.comm.nRanks = dda::common::kDdaMaxNranks;
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

// ---------------------------------------------------------------------------
// AllGather LL128
// ---------------------------------------------------------------------------

// Mirrors ddaLL128AGMaxPerRankBytes in dda_all_gather_fabric_ll128.cu. The scratch
// holds 2 banks of nRanks slots, so this is the payload of one rank's slot.
static size_t ddaLL128AGPerRankCapBytes(int nRanks, size_t scratchBytes)
{
    const size_t slices =
        scratchBytes / ((size_t)2 * (size_t)nRanks * (size_t)dda::common::kDdaLL128WireBytesPerSlice);
    return slices * (size_t)dda::common::kDdaLL128DataBytesPerSlice;
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_EligibleFloat16)
{
    EXPECT_TRUE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8, ncclFloat16));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_EligibleBfloat16)
{
    EXPECT_TRUE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8, ncclBfloat16));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclInt32));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_NullComm)
{
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        nullptr, sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_MissingBootstrap)
{
    mockComm_.comm.bootstrap = nullptr;
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_MissingFabricResources)
{
    mockComm_.setFabricResourcesPresent(false);
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

// The kernel derives its flag from a per-block epoch cell, so the path is only
// reachable once fabric init has allocated the array.
TEST_F(DdaFabricEligibilityTest, AllGatherLL128_MissingEpochCells)
{
    mockComm_.comm.ddaLLEpochDev = nullptr;
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_EmptyEpochArray)
{
    mockComm_.comm.ddaLLEpochLen = 0;
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_ZeroCount)
{
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_TooFewRanks)
{
    mockComm_.comm.nRanks = 1;
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_TooManyRanks)
{
    mockComm_.comm.nRanks = dda::common::kDdaMaxNranks + 1;
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_MaxRanksEligible)
{
    mockComm_.comm.nRanks = dda::common::kDdaMaxNranks;
    EXPECT_TRUE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

// 3 floats is 12B, so the per-rank payload is not a whole number of 16B chunks.
TEST_F(DdaFabricEligibilityTest, AllGatherLL128_UnalignedCount)
{
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_UnalignedSendbuff)
{
    void* unaligned = reinterpret_cast<void*>(0x1008);
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), unaligned, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_UnalignedRecvbuff)
{
    void* unaligned = reinterpret_cast<void*>(0x2008);
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, unaligned, 4, ncclFloat32));
}

// The cap is derived from the scratch rather than fixed, so shrinking the
// allocation to 1 MiB still leaves a 32-slice slot that a small message fits in.
TEST_F(DdaFabricEligibilityTest, AllGatherLL128_PerRankSlotWithSmallScratch)
{
    mockComm_.comm.ddaScratchBytes = 1 * 1024 * 1024;  // 32 slices at 8 ranks
    ASSERT_GT(ddaLL128AGPerRankCapBytes(mockComm_.comm.nRanks, mockComm_.comm.ddaScratchBytes),
              4u * sizeof(float));
    EXPECT_TRUE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

// One 16B chunk past the cap: still alignment-legal, but a partial trailing
// slice would run into the next rank's slot.
TEST_F(DdaFabricEligibilityTest, AllGatherLL128_PerRankSlotOverflow)
{
    mockComm_.comm.ddaScratchBytes = 1 * 1024 * 1024;
    const size_t capBytes =
        ddaLL128AGPerRankCapBytes(mockComm_.comm.nRanks, mockComm_.comm.ddaScratchBytes);
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, (capBytes + 16) / sizeof(float), ncclFloat32));
}

// Below 2 banks * nRanks * 2 KiB the slot holds zero whole slices, so the cap is
// zero and nothing fits.
TEST_F(DdaFabricEligibilityTest, AllGatherLL128_ScratchTooSmallForOneSlice)
{
    mockComm_.comm.ddaScratchBytes = 16 * 1024;
    ASSERT_EQ(ddaLL128AGPerRankCapBytes(mockComm_.comm.nRanks, mockComm_.comm.ddaScratchBytes), 0u);
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_AtThresholdEligible)
{
    constexpr size_t totalCap = 64u * 1024u * 1024u;  // DDA_LL128_THRESHOLD default
    const size_t perRankBytes = totalCap / (size_t)mockComm_.comm.nRanks;
    EXPECT_TRUE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, perRankBytes / sizeof(float), ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGatherLL128_PastThresholdRejected)
{
    constexpr size_t totalCap = 64u * 1024u * 1024u;
    const size_t perRankBytes = totalCap / (size_t)mockComm_.comm.nRanks + 16;
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, perRankBytes / sizeof(float), ncclFloat32));
}

} // namespace RcclUnitTesting
