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

#pragma once

#include <cstdint>

namespace rocprofiler
{
namespace kernel_replay
{
namespace blit
{
namespace kernel_abi
{
inline constexpr auto bytes_per_item = std::uint64_t{16};
inline constexpr auto workgroup_size = std::uint32_t{1024};

// Regions are cut into 16 KiB tiles from their start, copied one wave per tile. A list too short
// to give every wave about four tiles is split into pieces, down to 4 KiB.
inline constexpr auto tile_shift             = std::uint32_t{14};
inline constexpr auto tile_bytes             = std::uint64_t{1} << tile_shift;
inline constexpr auto min_piece_shift        = std::uint32_t{12};
inline constexpr auto pieces_per_wave        = std::uint64_t{4};
inline constexpr auto copy_unroll            = std::uint32_t{2};
inline constexpr auto copy_workgroups_per_cu = std::uint32_t{2};

struct copy_descriptor_t
{
    std::uint64_t size           = 0;
    std::uint64_t source_address = 0;
    std::uint64_t dest_address   = 0;
};

constexpr std::uint64_t
tiles_for(std::uint64_t size)
{
    return (size + tile_bytes - 1) >> tile_shift;
}

constexpr std::uint32_t
chunks_for(std::uint32_t descriptor_count)
{
    return (descriptor_count + workgroup_size - 1) / workgroup_size;
}

// Tile `index` of `region`: 16 KiB at index * 16 KiB, or what is left of the region.
constexpr copy_descriptor_t
tile_of(const copy_descriptor_t& region, std::uint64_t index)
{
    const auto offset = index << tile_shift;
    const auto left   = region.size - offset;
    return copy_descriptor_t{left < tile_bytes ? left : tile_bytes,
                             region.source_address + offset,
                             region.dest_address + offset};
}

// Each tile becomes 2^split pieces, so that `tile_count` tiles give `waves` waves about
// pieces_per_wave pieces each.
constexpr std::uint32_t
split_for(std::uint32_t tile_count, std::uint32_t waves)
{
    auto split = std::uint32_t{0};
    while(split < tile_shift - min_piece_shift &&
          (std::uint64_t{tile_count} << split) < pieces_per_wave * waves)
        ++split;
    return split;
}

// One argument block per kernel, passed by value. The SDK dispatches raw AQL packets and does not
// fill the implicit kernel arguments, so the kernels never read gridDim or blockDim: workgroup
// counts are passed here instead.
struct count_tiles_args_t
{
    std::uint64_t descriptors      = 0;
    std::uint64_t chunk_totals     = 0;
    std::uint32_t descriptor_count = 0;
    std::uint32_t reserved         = 0;
};

struct scan_tiles_args_t
{
    std::uint64_t descriptors      = 0;
    std::uint64_t chunk_totals     = 0;
    std::uint64_t first_tiles      = 0;
    std::uint64_t tile_count       = 0;
    std::uint32_t descriptor_count = 0;
    std::uint32_t chunk_count      = 0;
};

struct expand_tiles_args_t
{
    std::uint64_t descriptors      = 0;
    std::uint64_t first_tiles      = 0;
    std::uint64_t tile_count       = 0;
    std::uint64_t tiles            = 0;
    std::uint32_t descriptor_count = 0;
    std::uint32_t workgroups       = 0;
};

struct copy_tiles_args_t
{
    std::uint64_t tiles      = 0;
    std::uint64_t tile_count = 0;
    std::uint32_t workgroups = 0;
    std::uint32_t reserved   = 0;
};

static_assert(sizeof(copy_descriptor_t) == 24);
static_assert(sizeof(count_tiles_args_t) == 24);
static_assert(sizeof(scan_tiles_args_t) == 40);
static_assert(sizeof(expand_tiles_args_t) == 40);
static_assert(sizeof(copy_tiles_args_t) == 24);
}  // namespace kernel_abi
}  // namespace blit
}  // namespace kernel_replay
}  // namespace rocprofiler
