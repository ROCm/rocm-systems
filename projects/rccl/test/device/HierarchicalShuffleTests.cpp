/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for hierarchical_shuffle.h

#include "DeviceTestBase.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#include "hierarchical_shuffle.h"

namespace RcclUnitTesting
{

// 4 x 2 tile grid
static constexpr int kCols  = 2;
static constexpr int kRows  = 4;
static constexpr int kTiles = kCols * kRows;

// Tile (i,j) at i*kCols+j moves to j*kRows+i, so slot d holds this source tile.
static constexpr int kExpectedTileOrder[kTiles] = {0, 2, 4, 6, 1, 3, 5, 7};

// Distinct contents per tile
static std::vector<uint8_t> tilePattern(size_t rankOffset) {
  std::vector<uint8_t> h(rankOffset * kTiles);
  for (int t = 0; t < kTiles; t++)
    for (size_t b = 0; b < rankOffset; b++)
      h[t * rankOffset + b] = static_cast<uint8_t>(t * 31 + b);
  return h;
}

class HierarchicalShuffleTest : public DeviceTestBase {
protected:
  void TestShuffle(size_t rankOffset, int numBlocks, size_t baseOffset) {
    const size_t bytes = rankOffset * kTiles;

    const std::vector<uint8_t> h_src = tilePattern(rankOffset);
    std::vector<uint8_t> h_in(baseOffset + bytes, 0);
    std::memcpy(h_in.data() + baseOffset, h_src.data(), bytes);

    DeviceBuffer<uint8_t> d_src(baseOffset + bytes), d_dst(baseOffset + bytes);
    d_src.copyFrom(h_in);
    d_dst.zero();

    hierarchicalShuffle<<<numBlocks, HIERARCHICAL_SHUFFLE_THREADS>>>(
      reinterpret_cast<const char*>(d_src.ptr + baseOffset),
      reinterpret_cast<char*>(d_dst.ptr + baseOffset), rankOffset, kCols, kRows);
    syncAndCheck();

    auto h_out = d_dst.copyTo();
    for (int d = 0; d < kTiles; d++) {
      const uint8_t* expect = h_src.data() + kExpectedTileOrder[d] * rankOffset;
      const uint8_t* got    = h_out.data() + baseOffset + d * rankOffset;
      for (size_t b = 0; b < rankOffset; b++) {
        ASSERT_EQ(got[b], expect[b]) << "slot " << d << " should hold tile "
                                     << kExpectedTileOrder[d] << ", differs at byte " << b;
      }
    }
  }
};

// Aligned 16-byte-multiple tile
TEST_F(HierarchicalShuffleTest, TransposesTilesAcrossStridedBlocks) {
  TestShuffle(/*rankOffset=*/64, /*numBlocks=*/2, /*baseOffset=*/0);
}

// An 18-byte tile hits all three paths in one launch
TEST_F(HierarchicalShuffleTest, MixedAlignmentUsesAllCopyPaths) {
  TestShuffle(/*rankOffset=*/18, /*numBlocks=*/32, /*baseOffset=*/0);
}

// 16-byte-multiple tile on a misaligned base
TEST_F(HierarchicalShuffleTest, MisalignedBaseFallsBackToByteCopy) {
  TestShuffle(/*rankOffset=*/64, /*numBlocks=*/2, /*baseOffset=*/1);
}

}
