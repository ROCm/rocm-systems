/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <climits>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include "../src/transport/net_ib/gin.h"

TEST(RmaSegmentMathTest, SplitsAtVerbsLengthLimitWithoutLargeAllocation)
{
    ASSERT_GT(SIZE_MAX, static_cast<size_t>(UINT32_MAX));
    size_t remaining = static_cast<size_t>(UINT32_MAX) + 1;

    const size_t first =
        ncclRmaSegmentSliceBytes(remaining, remaining, remaining);
    EXPECT_EQ(first, static_cast<size_t>(UINT32_MAX));
    remaining -= first;
    EXPECT_EQ(ncclRmaSegmentSliceBytes(remaining, remaining, remaining),
              size_t{1});
}

TEST(RmaSegmentMathTest, FixedDataWrBudgetRejectsUnrepresentableChain)
{
    ASSERT_GT(SIZE_MAX, static_cast<size_t>(UINT32_MAX));
    size_t remaining =
        static_cast<size_t>(UINT32_MAX) * (NCCL_RMA_MAX_DATA_WRS + 1ULL);
    int slices = 0;
    while (remaining != 0 && slices < NCCL_RMA_MAX_DATA_WRS)
    {
        const size_t chunk =
            ncclRmaSegmentSliceBytes(remaining, remaining, remaining);
        remaining -= chunk;
        ++slices;
    }

    EXPECT_EQ(slices, NCCL_RMA_MAX_DATA_WRS);
    EXPECT_NE(remaining, size_t{0});
}

TEST(RmaSegmentMathTest, OnlyFinalWrIsSignaled)
{
    constexpr int kWrs = 7;
    for (int i = 0; i < kWrs; ++i)
        EXPECT_EQ(ncclRmaWrIsSignaled(i, kWrs), i == kWrs - 1);
    EXPECT_FALSE(ncclRmaWrIsSignaled(0, 0));
}

TEST(RmaSegmentMathTest, SignalAtomicMustBeAlignedWithinSegment)
{
    EXPECT_TRUE(ncclRmaSignalOffsetValid(/*signalOff=*/8, /*segmentEnd=*/16));
    EXPECT_FALSE(ncclRmaSignalOffsetValid(/*signalOff=*/4, /*segmentEnd=*/16));
    EXPECT_FALSE(ncclRmaSignalOffsetValid(/*signalOff=*/8, /*segmentEnd=*/12));
}

TEST(RmaSegmentMathTest, WorkRequestDepthsCoverMaximumChains)
{
    EXPECT_EQ(NCCL_RMA_MAX_DATA_WRS, 2 * NCCL_RMA_MAX_SEGMENTS);
    EXPECT_EQ(NCCL_RMA_MAX_SIGNAL_WRS, NCCL_RMA_MAX_DATA_WRS + 1);
    EXPECT_EQ(NCCL_RMA_MAX_FLUSH_WRS, NCCL_RMA_MAX_SEGMENTS);
}

namespace {

struct FakeWr
{
    int id;
    FakeWr* next;
};

int Posted(FakeWr* wr, int nWr, FakeWr* badWr)
{
    return ncclRmaPostedWrCount(wr, nWr, badWr, offsetof(FakeWr, next));
}

} // namespace

// wrap_ibv_post_send fails even after a prefix of the chain is accepted. Only
// the last WR is signaled, so posted==0 frees the request and posted>0 keeps it.
TEST(RmaSegmentMathTest, PrefixPostCountsWrsBeforeBadWr)
{
    FakeWr wr[3];
    wr[0] = {0, &wr[1]};
    wr[1] = {1, &wr[2]};
    wr[2] = {2, nullptr};

    EXPECT_EQ(Posted(&wr[0], 3, nullptr), 3);
    EXPECT_EQ(Posted(&wr[0], 3, &wr[0]), 0);
    EXPECT_EQ(Posted(&wr[0], 3, &wr[1]), 1);
    EXPECT_EQ(Posted(&wr[0], 3, &wr[2]), 2);
    EXPECT_EQ(Posted(&wr[0], 0, &wr[0]), 0);
    EXPECT_EQ(Posted(nullptr, 3, &wr[0]), 0);
}

TEST(RmaSegmentMathTest, PrefixPostKeepsRequestOnlyWhenSomethingPosted)
{
    EXPECT_TRUE(ncclRmaPostedWrCount(nullptr, 0, nullptr, 0) == 0);
    FakeWr wr[2];
    wr[0] = {0, &wr[1]};
    wr[1] = {1, nullptr};
    const int postedPrefix = Posted(&wr[0], 2, &wr[1]);
    const int postedNone = Posted(&wr[0], 2, &wr[0]);
    EXPECT_EQ(postedPrefix, 1);
    EXPECT_EQ(postedNone, 0);
    EXPECT_TRUE(postedPrefix > 0);
    EXPECT_FALSE(postedNone > 0);
}

// A failed handle calloc must still reach the status AllGather. memcpy of
// segOff is skipped until the handle exists.
TEST(RmaSegmentMathTest, FailedHandleCallocDoesNotCopySegmentOffsets)
{
    EXPECT_FALSE(ncclRmaRegistrationHandleReady(nullptr, 1));
    EXPECT_FALSE(ncclRmaRegistrationHandleReady(nullptr, NCCL_RMA_MAX_SEGMENTS));
    char handle{};
    EXPECT_FALSE(ncclRmaRegistrationHandleReady(&handle, 0));
    EXPECT_FALSE(ncclRmaRegistrationHandleReady(&handle, NCCL_RMA_MAX_SEGMENTS + 1));
    EXPECT_TRUE(ncclRmaRegistrationHandleReady(&handle, 1));
    EXPECT_TRUE(ncclRmaRegistrationHandleReady(&handle, NCCL_RMA_MAX_SEGMENTS));
}
