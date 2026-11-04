// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/kernel_replay/blit-copy-kernel.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/blit-copy.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{
namespace abi = rocprofiler::kernel_replay::blit::kernel_abi;

TEST(kernel_replay_blit_copy, descriptors_use_size_source_dest_order)
{
    EXPECT_EQ(offsetof(rocprofiler::kernel_replay::blit::copy_region_t, size), 0);
    EXPECT_LT(offsetof(rocprofiler::kernel_replay::blit::copy_region_t, size),
              offsetof(rocprofiler::kernel_replay::blit::copy_region_t, source));
    EXPECT_LT(offsetof(rocprofiler::kernel_replay::blit::copy_region_t, source),
              offsetof(rocprofiler::kernel_replay::blit::copy_region_t, dest));

    EXPECT_EQ(offsetof(abi::copy_descriptor_t, size), 0);
    EXPECT_EQ(offsetof(abi::copy_descriptor_t, source_address), sizeof(std::uint64_t));
    EXPECT_EQ(offsetof(abi::copy_descriptor_t, dest_address), 2 * sizeof(std::uint64_t));
}

TEST(kernel_replay_blit_copy, regions_become_16_kib_tiles_and_one_tail)
{
    EXPECT_EQ(abi::tile_bytes, 16384);
    EXPECT_EQ(abi::tiles_for(0), 0);
    EXPECT_EQ(abi::tiles_for(1), 1);
    EXPECT_EQ(abi::tiles_for(16384), 1);
    EXPECT_EQ(abi::tiles_for(16385), 2);
    EXPECT_EQ(abi::tiles_for(65536), 4);

    // 40 KiB + 5 B: two full tiles, then 8 KiB + 5 B, each at its offset on both sides.
    constexpr auto region = abi::copy_descriptor_t{40965, 0x10000, 0x900000};
    ASSERT_EQ(abi::tiles_for(region.size), 3);
    constexpr auto expected = std::array<abi::copy_descriptor_t, 3>{{
        {16384, 0x10000, 0x900000},
        {16384, 0x14000, 0x904000},
        {8197, 0x18000, 0x908000},
    }};
    for(size_t index = 0; index < expected.size(); ++index)
    {
        const auto tile = abi::tile_of(region, index);
        EXPECT_EQ(tile.size, expected.at(index).size) << "tile " << index;
        EXPECT_EQ(tile.source_address, expected.at(index).source_address) << "tile " << index;
        EXPECT_EQ(tile.dest_address, expected.at(index).dest_address) << "tile " << index;
    }
}

TEST(kernel_replay_blit_copy, expanded_tiles_cover_every_byte_once)
{
    // The GPU's count, scan and expand passes, done on the host with the same helpers: tile t
    // belongs to the last region whose first tile is at most t. 1,025 regions cross a chunk.
    auto regions = std::vector<abi::copy_descriptor_t>{};
    auto offset  = std::uint64_t{0};
    for(std::uint64_t index = 0; index < abi::workgroup_size + 1; ++index)
    {
        const auto size = (index * 7919) % 70000 + 1;
        regions.push_back({size, offset, offset + (std::uint64_t{1} << 40)});
        offset += size;
    }
    EXPECT_EQ(abi::chunks_for(static_cast<std::uint32_t>(regions.size())), 2);

    auto first_tiles = std::vector<std::uint64_t>{};
    auto total       = std::uint64_t{0};
    for(const auto& region : regions)
    {
        first_tiles.push_back(total);
        total += abi::tiles_for(region.size);
    }

    auto covered = std::uint64_t{0};
    for(std::uint64_t tile = 0; tile < total; ++tile)
    {
        const auto owner = std::upper_bound(first_tiles.begin(), first_tiles.end(), tile) -
                           first_tiles.begin() - 1;
        const auto& region = regions.at(owner);
        const auto  copy   = abi::tile_of(region, tile - first_tiles.at(owner));
        ASSERT_GT(copy.size, 0) << "tile " << tile;
        ASSERT_LE(copy.size, abi::tile_bytes) << "tile " << tile;
        EXPECT_EQ(copy.source_address, covered) << "tile " << tile;
        EXPECT_EQ(copy.dest_address - copy.source_address, std::uint64_t{1} << 40);
        ASSERT_LE(copy.source_address + copy.size, region.source_address + region.size);
        covered += copy.size;
    }
    EXPECT_EQ(covered, offset);
}

TEST(kernel_replay_blit_copy, short_lists_split_into_pieces_down_to_4_kib)
{
    // 304 CUs x 2 workgroups x 16 waves: lists shorter than 4 pieces per wave are split.
    constexpr auto waves = std::uint32_t{304 * 2 * 16};
    EXPECT_EQ(abi::split_for(1, waves), 2);
    EXPECT_EQ(abi::split_for(waves, waves), 2);
    EXPECT_EQ(abi::split_for(2 * waves - 1, waves), 2);
    EXPECT_EQ(abi::split_for(2 * waves, waves), 1);
    EXPECT_EQ(abi::split_for(4 * waves - 1, waves), 1);
    EXPECT_EQ(abi::split_for(4 * waves, waves), 0);
    for(std::uint32_t tiles = 1; tiles < 8 * waves; tiles += 97)
        EXPECT_GE(abi::tile_bytes >> abi::split_for(tiles, waves),
                  std::uint64_t{1} << abi::min_piece_shift);
}
}  // namespace
