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

#include "blit-copy-kernel.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

using rocprofiler::kernel_replay::blit::kernel_abi::bytes_per_item;
using rocprofiler::kernel_replay::blit::kernel_abi::copy_descriptor_t;
using rocprofiler::kernel_replay::blit::kernel_abi::copy_tiles_args_t;
using rocprofiler::kernel_replay::blit::kernel_abi::copy_unroll;
using rocprofiler::kernel_replay::blit::kernel_abi::count_tiles_args_t;
using rocprofiler::kernel_replay::blit::kernel_abi::expand_tiles_args_t;
using rocprofiler::kernel_replay::blit::kernel_abi::scan_tiles_args_t;
using rocprofiler::kernel_replay::blit::kernel_abi::split_for;
using rocprofiler::kernel_replay::blit::kernel_abi::tile_bytes;
using rocprofiler::kernel_replay::blit::kernel_abi::tile_of;
using rocprofiler::kernel_replay::blit::kernel_abi::tiles_for;
using rocprofiler::kernel_replay::blit::kernel_abi::workgroup_size;

using copy_vector_t       = std::uint32_t __attribute__((ext_vector_type(4)));
using global_byte_t       = std::byte __attribute__((address_space(1)));
using global_descriptor_t = copy_descriptor_t __attribute__((address_space(1)));
using global_u32_t        = std::uint32_t __attribute__((address_space(1)));

static_assert(bytes_per_item == sizeof(copy_vector_t));

namespace
{
__device__ inline copy_descriptor_t
load_descriptor(const global_descriptor_t* descriptors, std::uint64_t index)
{
    auto copy = copy_descriptor_t{};
    __builtin_memcpy(&copy, descriptors + index, sizeof(copy));
    return copy;
}

__device__ inline std::uint32_t
tile_count_of(const global_descriptor_t* descriptors, std::uint32_t index)
{
    return static_cast<std::uint32_t>(tiles_for(load_descriptor(descriptors, index).size));
}

// copy_span loads and stores through these: clang rejects __builtin_memcpy in templated device
// code, and copying straight into an array element spilled the array to scratch.
__device__ inline copy_vector_t
load_item(const global_byte_t* from)
{
    copy_vector_t value;
    __builtin_memcpy(&value, from, sizeof(value));
    return value;
}

__device__ inline void
store_item(global_byte_t* to, const copy_vector_t& value)
{
    __builtin_memcpy(to, &value, sizeof(value));
}

// The whole wave copies `bytes` with `unroll` 16-byte loads in flight per lane. Full steps load
// without bounds checks: a guarded load becomes a branch the wave waits out before the next load
// issues.
template <std::uint32_t unroll>
__device__ void
copy_span(const global_byte_t* src, global_byte_t* dst, std::uint64_t bytes)
{
    const auto lane   = static_cast<std::uint64_t>(threadIdx.x % warpSize);
    const auto chunks = bytes / bytes_per_item;
    const auto step   = static_cast<std::uint64_t>(warpSize) * unroll;
    const auto full   = chunks / step * step;
    for(auto base = std::uint64_t{0}; base < full; base += step)
    {
        copy_vector_t values[unroll];
#pragma unroll
        for(std::uint32_t u = 0; u < unroll; ++u)
            values[u] = load_item(src + (base + u * warpSize + lane) * bytes_per_item);
#pragma unroll
        for(std::uint32_t u = 0; u < unroll; ++u)
            store_item(dst + (base + u * warpSize + lane) * bytes_per_item, values[u]);
    }
    for(auto chunk = full + lane; chunk < chunks; chunk += warpSize)
        store_item(dst + chunk * bytes_per_item, load_item(src + chunk * bytes_per_item));
    const auto tail = bytes % bytes_per_item;
    if(lane < tail) dst[chunks * bytes_per_item + lane] = src[chunks * bytes_per_item + lane];
}

// Inclusive scan of one value per thread across the workgroup; `total` is the workgroup's sum.
__device__ std::uint32_t
           block_inclusive_scan(std::uint32_t value, std::uint32_t& total)
{
    __shared__ std::uint32_t values[workgroup_size];
    const auto               thread = threadIdx.x;
    values[thread]                  = value;
    __syncthreads();
    for(std::uint32_t offset = 1; offset < workgroup_size; offset *= 2)
    {
        const auto add = thread >= offset ? values[thread - offset] : 0U;
        __syncthreads();
        values[thread] += add;
        __syncthreads();
    }
    const auto result = values[thread];
    total             = values[workgroup_size - 1];
    __syncthreads();
    return result;
}
}  // namespace

// Pass 1 of 3 of the tile list: the tiles in each chunk of workgroup_size descriptors.
extern "C" __global__ __launch_bounds__(workgroup_size) void kernel_replay_count_tiles(
    count_tiles_args_t args)
{
    const auto* descriptors  = reinterpret_cast<const global_descriptor_t*>(args.descriptors);
    auto*       chunk_totals = reinterpret_cast<global_u32_t*>(args.chunk_totals);
    const auto  index        = blockIdx.x * workgroup_size + threadIdx.x;

    auto total = std::uint32_t{0};
    block_inclusive_scan(index < args.descriptor_count ? tile_count_of(descriptors, index) : 0,
                         total);
    if(threadIdx.x == 0) chunk_totals[blockIdx.x] = total;
}

// Pass 2 of 3: each chunk adds up the chunks before it, then scans its own descriptors into their
// first tiles. The last chunk stores the total.
extern "C" __global__ __launch_bounds__(workgroup_size) void kernel_replay_scan_tiles(
    scan_tiles_args_t args)
{
    const auto* descriptors  = reinterpret_cast<const global_descriptor_t*>(args.descriptors);
    const auto* chunk_totals = reinterpret_cast<const global_u32_t*>(args.chunk_totals);
    auto*       first_tiles  = reinterpret_cast<global_u32_t*>(args.first_tiles);
    auto*       tile_count   = reinterpret_cast<global_u32_t*>(args.tile_count);

    auto before = std::uint32_t{0};
    for(auto chunk = threadIdx.x; chunk < blockIdx.x; chunk += workgroup_size)
        before += chunk_totals[chunk];
    auto base = std::uint32_t{0};
    block_inclusive_scan(before, base);

    const auto index     = blockIdx.x * workgroup_size + threadIdx.x;
    const auto tiles     = index < args.descriptor_count ? tile_count_of(descriptors, index) : 0;
    auto       total     = std::uint32_t{0};
    const auto inclusive = block_inclusive_scan(tiles, total);
    if(index < args.descriptor_count) first_tiles[index] = base + inclusive - tiles;
    if(blockIdx.x == args.chunk_count - 1 && threadIdx.x == 0) *tile_count = base + total;
}

// Pass 3 of 3, one thread per tile: a binary search over the first tiles finds its descriptor.
extern "C" __global__ __launch_bounds__(workgroup_size) void kernel_replay_expand_tiles(
    expand_tiles_args_t args)
{
    const auto* descriptors = reinterpret_cast<const global_descriptor_t*>(args.descriptors);
    const auto* first_tiles = reinterpret_cast<const global_u32_t*>(args.first_tiles);
    const auto  total       = *reinterpret_cast<const global_u32_t*>(args.tile_count);
    auto*       tiles       = reinterpret_cast<global_descriptor_t*>(args.tiles);
    const auto  stride      = std::uint64_t{args.workgroups} * workgroup_size;

    for(auto tile = std::uint64_t{blockIdx.x} * workgroup_size + threadIdx.x; tile < total;
        tile += stride)
    {
        auto low  = std::uint32_t{0};
        auto high = args.descriptor_count;
        while(high - low > 1)
        {
            const auto middle = (low + high) / 2;
            if(first_tiles[middle] <= tile)
                low = middle;
            else
                high = middle;
        }
        const auto copy = tile_of(load_descriptor(descriptors, low), tile - first_tiles[low]);
        __builtin_memcpy(tiles + tile, &copy, sizeof(copy));
    }
}

// One wave per tile. Work item u is piece u / count of tile u % count, so one tile's pieces, and
// the empty pieces of short tiles, land on different waves. Consecutive waves sit in different
// workgroups, so a short list still spreads over every CU.
extern "C" __global__ __launch_bounds__(workgroup_size) void kernel_replay_copy_tiles(
    copy_tiles_args_t args)
{
    const auto* tiles       = reinterpret_cast<const global_descriptor_t*>(args.tiles);
    const auto  count       = *reinterpret_cast<const global_u32_t*>(args.tile_count);
    const auto  waves       = args.workgroups * (workgroup_size / warpSize);
    const auto  split       = split_for(count, waves);
    const auto  piece_bytes = tile_bytes >> split;
    const auto  items       = count << split;
    const auto  wave        = static_cast<std::uint32_t>(
        __builtin_amdgcn_readfirstlane(threadIdx.x / warpSize * args.workgroups + blockIdx.x));

    for(auto item = wave; item < items; item += waves)
    {
        const auto tile   = load_descriptor(tiles, item % count);
        const auto offset = (item / count) * piece_bytes;
        if(offset >= tile.size) continue;
        const auto left = tile.size - offset;
        copy_span<copy_unroll>(reinterpret_cast<const global_byte_t*>(tile.source_address) + offset,
                               reinterpret_cast<global_byte_t*>(tile.dest_address) + offset,
                               left < piece_bytes ? left : piece_bytes);
    }
}
