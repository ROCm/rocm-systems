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
inline constexpr auto hash_unroll            = std::uint32_t{8};
inline constexpr auto copy_workgroups_per_cu = std::uint32_t{2};

// Counters the restore kernels keep on the GPU. Successive changed-tile lists alternate between
// two counts, so each restore can clear the count the next one uses.
inline constexpr auto changed_count_slots = std::uint32_t{2};
inline constexpr auto unlisted_counter    = std::uint32_t{2};
inline constexpr auto counter_count       = std::uint32_t{4};

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

// A tile's digest is the sum of digest_term(word, position) over its little-endian 8-byte words,
// counting positions from the tile's start and zero-padding a partial last word. A sum does not
// depend on which wave adds which word, so a tile copied in pieces is digested in the same pass.
// Each term is splitmix64's finalizer of the word keyed by its position. Every step is invertible,
// so two different words at one position never give the same term: changing any one word always
// changes the digest.
constexpr std::uint64_t
digest_term(std::uint64_t word, std::uint64_t position)
{
    auto value = word ^ (position * 0x9e3779b97f4a7c15ULL);
    value      = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value      = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
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

// With digests, the capture also zeroes each tile's digest and changed flag, and the counters.
struct expand_tiles_args_t
{
    std::uint64_t descriptors      = 0;
    std::uint64_t first_tiles      = 0;
    std::uint64_t tile_count       = 0;
    std::uint64_t tiles            = 0;
    std::uint64_t digests          = 0;
    std::uint64_t changed_flags    = 0;
    std::uint64_t counters         = 0;
    std::uint32_t descriptor_count = 0;
    std::uint32_t workgroups       = 0;
};

// With reverse set, each tile goes from its destination back to its source: a full restore from a
// capture's tile list.
struct copy_tiles_args_t
{
    std::uint64_t tiles      = 0;
    std::uint64_t tile_count = 0;
    std::uint32_t workgroups = 0;
    std::uint32_t reverse    = 0;
};

struct capture_tiles_args_t
{
    std::uint64_t tiles      = 0;
    std::uint64_t tile_count = 0;
    std::uint64_t digests    = 0;
    std::uint32_t workgroups = 0;
    std::uint32_t reserved   = 0;
};

// The restore kernels read a capture's tile list, whose tiles copy live memory to the snapshot,
// and copy the other way.
struct restore_changed_args_t
{
    std::uint64_t tiles         = 0;
    std::uint64_t tile_count    = 0;
    std::uint64_t digests       = 0;
    std::uint64_t changed_flags = 0;
    std::uint64_t changed       = 0;
    std::uint64_t counters      = 0;
    std::uint32_t slot          = 0;
    std::uint32_t workgroups    = 0;
};

struct restore_listed_args_t
{
    std::uint64_t tiles         = 0;
    std::uint64_t changed       = 0;
    std::uint64_t changed_count = 0;
    std::uint32_t workgroups    = 0;
    std::uint32_t reserved      = 0;
};

struct check_tiles_args_t
{
    std::uint64_t tiles         = 0;
    std::uint64_t tile_count    = 0;
    std::uint64_t digests       = 0;
    std::uint64_t changed_flags = 0;
    std::uint64_t counters      = 0;
    std::uint32_t workgroups    = 0;
    std::uint32_t reserved      = 0;
};

static_assert(sizeof(copy_descriptor_t) == 24);
static_assert(sizeof(count_tiles_args_t) == 24);
static_assert(sizeof(scan_tiles_args_t) == 40);
static_assert(sizeof(expand_tiles_args_t) == 64);
static_assert(sizeof(copy_tiles_args_t) == 24);
static_assert(sizeof(capture_tiles_args_t) == 32);
static_assert(sizeof(restore_changed_args_t) == 56);
static_assert(sizeof(restore_listed_args_t) == 32);
static_assert(sizeof(check_tiles_args_t) == 48);
}  // namespace kernel_abi
}  // namespace blit
}  // namespace kernel_replay
}  // namespace rocprofiler
