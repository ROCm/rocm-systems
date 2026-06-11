/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include "gin/gin_anvil_host_parse.h"

namespace RcclUnitTesting {

TEST(GinAnvilHostParse, NumSdmaChannels_DefaultWhenBothUnset) {
  EXPECT_EQ(ncclGinAnvilParseNumSdmaChannels(nullptr, nullptr), 4);
}

TEST(GinAnvilHostParse, NumSdmaChannels_PrefersNcclOverRocshmem) {
  EXPECT_EQ(ncclGinAnvilParseNumSdmaChannels("2", "7"), 2);
}

TEST(GinAnvilHostParse, NumSdmaChannels_FallsBackWhenNcclUnset) {
  EXPECT_EQ(ncclGinAnvilParseNumSdmaChannels(nullptr, "3"), 3);
}

TEST(GinAnvilHostParse, NumSdmaChannels_ClampLow) {
  EXPECT_EQ(ncclGinAnvilParseNumSdmaChannels("0", nullptr), 1);
  EXPECT_EQ(ncclGinAnvilParseNumSdmaChannels("-99", nullptr), 1);
}

TEST(GinAnvilHostParse, NumSdmaChannels_ClampHigh) {
  EXPECT_EQ(ncclGinAnvilParseNumSdmaChannels("9", nullptr), 8);
  EXPECT_EQ(ncclGinAnvilParseNumSdmaChannels("1000", nullptr), 8);
}

TEST(GinAnvilHostParse, NumSdmaChannels_HexAndDecimal) {
  EXPECT_EQ(ncclGinAnvilParseNumSdmaChannels("0x8", nullptr), 8);
  EXPECT_EQ(ncclGinAnvilParseNumSdmaChannels("1", nullptr), 1);
}

TEST(GinAnvilHostParse, SdmaChunkBytes_DefaultEmpty) {
  EXPECT_EQ(ncclGinAnvilParseSdmaChunkBytes(nullptr), 8u << 20u);
  EXPECT_EQ(ncclGinAnvilParseSdmaChunkBytes(""), 8u << 20u);
}

TEST(GinAnvilHostParse, SdmaChunkBytes_ParseMb) {
  EXPECT_EQ(ncclGinAnvilParseSdmaChunkBytes("1"), 1u << 20u);
  EXPECT_EQ(ncclGinAnvilParseSdmaChunkBytes("16"), 16u << 20u);
}

TEST(GinAnvilHostParse, SdmaChunkBytes_ClampLow) {
  EXPECT_EQ(ncclGinAnvilParseSdmaChunkBytes("0"), 1u << 20u);
}

TEST(GinAnvilHostParse, SdmaChunkBytes_ClampHigh) {
  EXPECT_EQ(ncclGinAnvilParseSdmaChunkBytes("200"), 128u << 20u);
  EXPECT_EQ(ncclGinAnvilParseSdmaChunkBytes("128"), 128u << 20u);
}

}  // namespace RcclUnitTesting
