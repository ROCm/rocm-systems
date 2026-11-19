/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <climits>
#include <cstdint>

#include <gtest/gtest.h>

#include "transport/rccl_mseg_rma.h"

TEST(RmaSegmentMathTest, SegCountSupportedIsTheOnlyCap)
{
    EXPECT_FALSE(rcclIbSegCountSupported(0));
    EXPECT_FALSE(rcclIbSegCountSupported(-1));
    EXPECT_TRUE(rcclIbSegCountSupported(1));
    EXPECT_TRUE(rcclIbSegCountSupported(17));
    EXPECT_TRUE(rcclIbSegCountSupported(RCCL_IB_REG_SEG_DEFAULT));
    EXPECT_FALSE(rcclIbSegCountSupported(RCCL_IB_REG_SEG_DEFAULT + 1));
}

// RMA bounds ranks x segments: the full count at small scale, fewer segments as ranks grow, one always.
TEST(RmaSegmentMathTest, SegTableFitsBoundsRanksTimesSegments)
{
    EXPECT_TRUE(rcclRmaSegTableFits(1024, RCCL_IB_REG_SEG_DEFAULT));
    EXPECT_FALSE(rcclRmaSegTableFits(1025, RCCL_IB_REG_SEG_DEFAULT));
    EXPECT_TRUE(rcclRmaSegTableFits(2048, 512));
    EXPECT_FALSE(rcclRmaSegTableFits(2049, 512));
    EXPECT_TRUE(rcclRmaSegTableFits(80000, 13));
    EXPECT_FALSE(rcclRmaSegTableFits(80000, 14));
    EXPECT_TRUE(rcclRmaSegTableFits(80000, 1));
    EXPECT_TRUE(rcclRmaSegTableFits(INT_MAX, 1));
    EXPECT_FALSE(rcclRmaSegTableFits(2, RCCL_IB_REG_SEG_DEFAULT + 1));
    EXPECT_FALSE(rcclRmaSegTableFits(0, 1));
    EXPECT_FALSE(rcclRmaSegTableFits(2, 0));
}

TEST(RmaSegmentMathTest, InflightCapAppliesOnlyToMultiSegmentChains)
{
    EXPECT_EQ(rcclRmaInflightRequestsCap(false, 256), 256u);
    EXPECT_EQ(rcclRmaInflightRequestsCap(true, 256), static_cast<uint32_t>(RCCL_RMA_MAX_INFLIGHT_REQUESTS));
    EXPECT_EQ(rcclRmaInflightRequestsCap(true, 16), 16u) << "a smaller proxy limit stays";
}
