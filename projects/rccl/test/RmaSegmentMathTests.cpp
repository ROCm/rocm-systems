/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include "../src/transport/net_ib/gin.h"

#include <climits>
#include <cstddef>
#include <cstdint>

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
