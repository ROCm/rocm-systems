/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "algorithms/dda/device/CollCommon.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{

TEST(DdaCollCommon, divRoundUp)
{
    EXPECT_EQ(dda::common::divRoundUp(0, 8), 1u);
    EXPECT_EQ(dda::common::divRoundUp(1, 8), 1u);
    EXPECT_EQ(dda::common::divRoundUp(7, 8), 1u);
    EXPECT_EQ(dda::common::divRoundUp(8, 8), 1u);
    EXPECT_EQ(dda::common::divRoundUp(9, 8), 2u);
}

TEST(DdaCollCommon, calcBlockCount)
{
    EXPECT_EQ(dda::common::calcBlockCount(0, 512, 24), 1u);
    EXPECT_EQ(dda::common::calcBlockCount(512, 512, 24), 1u);
    EXPECT_EQ(dda::common::calcBlockCount(513, 512, 24), 2u);
    EXPECT_EQ(dda::common::calcBlockCount(12288, 512, 24), 24u);
    EXPECT_EQ(dda::common::calcBlockCount(20000, 512, 24), 24u);
}

TEST(DdaCollCommon, getGridAndBlockDims_SmallCount)
{
    constexpr int   kTypeSize  = static_cast<int>(sizeof(float));
    constexpr size_t kMaxBlocks = 24;
    constexpr size_t kCount     = 4;

    auto gridBlock =
        dda::common::getGridAndBlockDims(kCount, kTypeSize, kMaxBlocks);

    EXPECT_EQ(gridBlock.first.x, 1u);
    EXPECT_EQ(gridBlock.second.x, 64u);
}

TEST(DdaCollCommon, getGridAndBlockDims_LargeCount)
{
    constexpr int    kTypeSize  = static_cast<int>(sizeof(float));
    constexpr size_t kMaxBlocks = 24;
    constexpr size_t kCount     = 4096;

    auto gridBlock =
        dda::common::getGridAndBlockDims(kCount, kTypeSize, kMaxBlocks);

    EXPECT_EQ(gridBlock.first.x, 2u);
    EXPECT_EQ(gridBlock.second.x, 512u);
}

TEST(DdaCollCommon, getGridAndBlockDims_CapsBlocksAtMax)
{
    constexpr int    kTypeSize  = static_cast<int>(sizeof(float));
    constexpr size_t kMaxBlocks = 24;
    constexpr size_t kCount     = 100000;

    auto gridBlock =
        dda::common::getGridAndBlockDims(kCount, kTypeSize, kMaxBlocks);

    EXPECT_EQ(gridBlock.first.x, kMaxBlocks);
    EXPECT_EQ(gridBlock.second.x, 512u);
}

TEST(DdaCollCommon, getGridAndBlockDims_HalfPrecision)
{
    constexpr int    kTypeSize  = static_cast<int>(sizeof(uint16_t));
    constexpr size_t kMaxBlocks = 24;
    constexpr size_t kCount     = 1024;

    auto gridBlock =
        dda::common::getGridAndBlockDims(kCount, kTypeSize, kMaxBlocks);

    EXPECT_GE(gridBlock.first.x, 1u);
    EXPECT_LE(gridBlock.first.x, kMaxBlocks);
    EXPECT_GE(gridBlock.second.x, 64u);
    EXPECT_LE(gridBlock.second.x, 512u);
}

} // namespace RcclUnitTesting
