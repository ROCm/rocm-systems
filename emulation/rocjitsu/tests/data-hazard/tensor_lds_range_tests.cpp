// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "rocjitsu/isa/arch/amdgpu/shared/tensor_dma.h"
#include "rocjitsu/vm/plugins/data_hazard/plugin.h"

#include <cstdint>
#include <vector>

namespace dh = rocjitsu::plugins::data_hazard;
namespace tdm = rocjitsu::amdgpu::tensor_dma_detail;

namespace {

/// A rank-`rank` dense tile of 4-byte elements at LDS offset @p lds_base, of the
/// shape the descriptor reader produces for one.
tdm::TensorDmaDescriptor make_tile(std::vector<uint32_t> tile_dims, uint32_t lds_base = 0,
                                   uint32_t elem_size = 4) {
  tdm::TensorDmaDescriptor desc;
  desc.count = 1;
  desc.elem_size = elem_size;
  desc.lds_base = lds_base;
  desc.tensor_rank = static_cast<uint32_t>(tile_dims.size());
  for (size_t dim = 0; dim < tile_dims.size(); ++dim) {
    desc.tile_dims[dim] = tile_dims[dim];
    // Bounds do not enter into which LDS a load writes, but a tensor extent of
    // zero skips layout validation, so keep the tile inside the tensor.
    desc.tensor_dims[dim] = tile_dims[dim];
  }
  return desc;
}

std::vector<std::pair<uint64_t, uint32_t>> ranges_of(const tdm::TensorDmaDescriptor &desc) {
  std::vector<std::pair<uint64_t, uint32_t>> flattened;
  for (const dh::LocalMemoryRange &range : dh::tensor_lds_write_ranges(desc))
    flattened.emplace_back(range.address, range.size);
  return flattened;
}

using Ranges = std::vector<std::pair<uint64_t, uint32_t>>;

TEST(TensorLdsWriteRangeTest, DenseTileCoversWholeTileFromDescriptorBase) {
  // 8x4 elements of 4 bytes, based 0x200 bytes into the wave's allocation.
  const auto desc = make_tile({8, 4}, /*lds_base=*/0x200);
  EXPECT_EQ(ranges_of(desc), (Ranges{{0x200, 8 * 4 * 4}}));
}

TEST(TensorLdsWriteRangeTest, ElementSizeScalesTheTileExtent) {
  const auto desc = make_tile({16}, /*lds_base=*/0, /*elem_size=*/2);
  EXPECT_EQ(ranges_of(desc), (Ranges{{0, 32}}));
}

TEST(TensorLdsWriteRangeTest, InactiveDescriptorWritesNothing) {
  auto desc = make_tile({32});
  desc.count = 0;
  EXPECT_TRUE(ranges_of(desc).empty());
}

TEST(TensorLdsWriteRangeTest, DescriptorTheExecutorRejectsWritesNothing) {
  auto desc = make_tile({8, 4});
  // A tile dimension of zero below the descriptor's rank is unimplemented, so
  // the transfer faults instead of writing LDS.
  desc.tile_dims[0] = 0;
  EXPECT_TRUE(ranges_of(desc).empty());
}

TEST(TensorLdsWriteRangeTest, OutOfBoundsTileStillCoversTheLdsItZeroFills) {
  auto desc = make_tile({8, 4}, /*lds_base=*/0x40);
  // The tile reaches past the tensor: a load writes zeros for the elements
  // outside it rather than skipping their LDS.
  desc.tensor_dims[0] = 2;
  EXPECT_EQ(ranges_of(desc), (Ranges{{0x40, 8 * 4 * 4}}));
}

TEST(TensorLdsWriteRangeTest, PaddedTileCoversTheSpanTheSkewedRowsOccupy) {
  auto desc = make_tile({8, 4});
  desc.pad = true;
  desc.pad_interval = 8; // dwords: pad after every 32 bytes, one row here.
  desc.pad_amount = 1;   // dwords: 4 bytes of skew per row.
  // Four rows of 32 bytes, three of them followed by 4 bytes of skew.
  EXPECT_EQ(ranges_of(desc), (Ranges{{0, 4 * 32 + 3 * 4}}));
}

TEST(TensorLdsWriteRangeTest, IterationStrideStackingTilesReportsOneRange) {
  auto desc = make_tile({8, 4});
  desc.iterate = true;
  desc.iteration_count = 3;
  desc.lds_increment = 8 * 4; // Elements: the next tile starts where this ends.
  EXPECT_EQ(ranges_of(desc), (Ranges{{0, 3 * 8 * 4 * 4}}));
}

TEST(TensorLdsWriteRangeTest, IterationStrideLeavingGapsReportsRangePerTile) {
  auto desc = make_tile({8, 4}, /*lds_base=*/0x100);
  desc.iterate = true;
  desc.iteration_count = 3;
  desc.lds_increment = 8 * 4 * 2; // Elements: one tile of spacing between tiles.
  EXPECT_EQ(ranges_of(desc), (Ranges{{0x100, 128}, {0x100 + 256, 128}, {0x100 + 512, 128}}));
}

TEST(TensorLdsWriteRangeTest, IterationStrideOfZeroRewritesTheSameTile) {
  auto desc = make_tile({8, 4});
  desc.iterate = true;
  desc.iteration_count = 4;
  desc.lds_increment = 0;
  EXPECT_EQ(ranges_of(desc), (Ranges{{0, 128}}));
}

TEST(TensorLdsWriteRangeTest, GatherPacksItsRowsFromTheDescriptorBase) {
  tdm::TensorDmaDescriptor desc;
  desc.count = 1;
  desc.elem_size = 4;
  desc.lds_base = 0x80;
  desc.gather = true;
  desc.tensor_rank = 2;
  desc.tile_dims[0] = 6;
  desc.tensor_dims[0] = 6;
  desc.tensor_dims[1] = 16;
  desc.valid_indices = 3;
  desc.gather_indices[0] = 4;
  desc.gather_indices[1] = 9;
  desc.gather_indices[2] = 1;
  // Three gathered rows of six elements, packed in index order.
  EXPECT_EQ(ranges_of(desc), (Ranges{{0x80, 3 * 6 * 4}}));
}

TEST(TensorLdsWriteRangeTest, ScatteredTilesBeyondTheRangeLimitCollapseToTheirSpan) {
  auto desc = make_tile({2, 2});
  desc.iterate = true;
  desc.iteration_count = 128;
  desc.lds_increment = 8; // Elements: twice the tile, so every tile is disjoint.
  // Past the tracked-range limit the transfer is reported as the span it
  // covers: 128 tiles of 16 bytes, the last one starting 127 strides in.
  EXPECT_EQ(ranges_of(desc), (Ranges{{0, 127 * 32 + 16}}));
}

TEST(TensorLdsWriteRangeTest, RangesTheEngineCannotAddressAreDropped) {
  auto desc = make_tile({8, 1});
  desc.lds_base = 0xfffffff0;
  desc.iterate = true;
  desc.iteration_count = 2;
  // The second tile begins past the 32-bit LDS address space the engine tracks.
  desc.lds_increment = 0x40000000;
  EXPECT_EQ(ranges_of(desc), (Ranges{{0xfffffff0, 32}}));
}

} // namespace
