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
using rocprofiler::kernel_replay::blit::kernel_abi::capture_tiles_args_t;
using rocprofiler::kernel_replay::blit::kernel_abi::changed_count_slots;
using rocprofiler::kernel_replay::blit::kernel_abi::check_tiles_args_t;
using rocprofiler::kernel_replay::blit::kernel_abi::copy_descriptor_t;
using rocprofiler::kernel_replay::blit::kernel_abi::copy_tiles_args_t;
using rocprofiler::kernel_replay::blit::kernel_abi::copy_unroll;
using rocprofiler::kernel_replay::blit::kernel_abi::count_tiles_args_t;
using rocprofiler::kernel_replay::blit::kernel_abi::counter_count;
using rocprofiler::kernel_replay::blit::kernel_abi::digest_term;
using rocprofiler::kernel_replay::blit::kernel_abi::expand_tiles_args_t;
using rocprofiler::kernel_replay::blit::kernel_abi::hash_unroll;
using rocprofiler::kernel_replay::blit::kernel_abi::restore_changed_args_t;
using rocprofiler::kernel_replay::blit::kernel_abi::restore_listed_args_t;
using rocprofiler::kernel_replay::blit::kernel_abi::scan_tiles_args_t;
using rocprofiler::kernel_replay::blit::kernel_abi::split_for;
using rocprofiler::kernel_replay::blit::kernel_abi::tile_bytes;
using rocprofiler::kernel_replay::blit::kernel_abi::tile_of;
using rocprofiler::kernel_replay::blit::kernel_abi::tiles_for;
using rocprofiler::kernel_replay::blit::kernel_abi::unlisted_counter;
using rocprofiler::kernel_replay::blit::kernel_abi::workgroup_size;

using copy_vector_t       = std::uint32_t __attribute__((ext_vector_type(4)));
using global_byte_t       = std::byte __attribute__((address_space(1)));
using global_descriptor_t = copy_descriptor_t __attribute__((address_space(1)));
using global_u8_t         = std::uint8_t __attribute__((address_space(1)));
using global_u32_t        = std::uint32_t __attribute__((address_space(1)));
using global_u64_t        = std::uint64_t __attribute__((address_space(1)));

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

// move_span loads and stores through these: clang rejects __builtin_memcpy in templated device
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

// The digest terms of the 16-byte chunk `chunk` chunks into its tile.
__device__ inline std::uint64_t
chunk_terms(const copy_vector_t& value, std::uint64_t chunk)
{
    const auto low  = (std::uint64_t{value.y} << 32) | value.x;
    const auto high = (std::uint64_t{value.w} << 32) | value.z;
    return digest_term(low, 2 * chunk) + digest_term(high, 2 * chunk + 1);
}

// The whole wave reads `bytes` from `src` with `unroll` 16-byte loads in flight per lane. With
// `copy` it stores them to `dst`; with `hash` it returns their digest terms summed over the wave,
// for bytes that start `first_chunk` chunks into their tile. Full steps load without bounds checks:
// a guarded load becomes a branch the wave waits out before the next load issues.
template <std::uint32_t unroll, bool copy, bool hash>
__device__ std::uint64_t
           move_span(const global_byte_t* src,
                     global_byte_t*       dst,
                     std::uint64_t        bytes,
                     std::uint64_t        first_chunk)
{
    const auto lane   = static_cast<std::uint64_t>(threadIdx.x % warpSize);
    const auto chunks = bytes / bytes_per_item;
    const auto step   = static_cast<std::uint64_t>(warpSize) * unroll;
    const auto full   = chunks / step * step;
    auto       sum    = std::uint64_t{0};
    for(auto base = std::uint64_t{0}; base < full; base += step)
    {
        copy_vector_t values[unroll];
#pragma unroll
        for(std::uint32_t u = 0; u < unroll; ++u)
            values[u] = load_item(src + (base + u * warpSize + lane) * bytes_per_item);
#pragma unroll
        for(std::uint32_t u = 0; u < unroll; ++u)
        {
            const auto chunk = base + u * warpSize + lane;
            if constexpr(copy) store_item(dst + chunk * bytes_per_item, values[u]);
            if constexpr(hash) sum += chunk_terms(values[u], first_chunk + chunk);
        }
    }
    for(auto chunk = full + lane; chunk < chunks; chunk += warpSize)
    {
        const auto value = load_item(src + chunk * bytes_per_item);
        if constexpr(copy) store_item(dst + chunk * bytes_per_item, value);
        if constexpr(hash) sum += chunk_terms(value, first_chunk + chunk);
    }

    const auto tail = bytes % bytes_per_item;
    if constexpr(copy)
    {
        if(lane < tail) dst[chunks * bytes_per_item + lane] = src[chunks * bytes_per_item + lane];
    }
    if constexpr(hash)
    {
        if(lane == 0 && tail != 0)
        {
            auto low  = std::uint64_t{0};
            auto high = std::uint64_t{0};
            for(auto index = std::uint64_t{0}; index < tail; ++index)
            {
                const auto byte = std::uint64_t{
                    std::to_integer<std::uint8_t>(src[chunks * bytes_per_item + index])};
                if(index < 8)
                    low |= byte << (8 * index);
                else
                    high |= byte << (8 * (index - 8));
            }
            const auto position = 2 * (first_chunk + chunks);
            sum += digest_term(low, position);
            if(tail > 8) sum += digest_term(high, position + 1);
        }
        for(auto offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_xor(static_cast<unsigned long long>(sum), offset);
    }
    return sum;
}

__device__ inline std::uint32_t
wave_index(std::uint32_t workgroups)
{
    return static_cast<std::uint32_t>(
        __builtin_amdgcn_readfirstlane(threadIdx.x / warpSize * workgroups + blockIdx.x));
}

// One wave per tile. Work item u is piece u / count of tile u % count, so one tile's pieces, and
// the empty pieces of short tiles, land on different waves. Consecutive waves sit in different
// workgroups, so a short list still spreads over every CU. With digests, each piece adds its terms
// to its tile's digest.
template <bool hash>
__device__ void
copy_tiles(const global_descriptor_t* tiles,
           std::uint32_t              count,
           std::uint32_t              workgroups,
           unsigned long long*        digests,
           bool                       reverse)
{
    const auto waves       = workgroups * (workgroup_size / warpSize);
    const auto split       = split_for(count, waves);
    const auto piece_bytes = tile_bytes >> split;
    const auto items       = count << split;

    for(auto item = wave_index(workgroups); item < items; item += waves)
    {
        const auto index  = item % count;
        const auto tile   = load_descriptor(tiles, index);
        const auto offset = (item / count) * piece_bytes;
        if(offset >= tile.size) continue;
        const auto left = tile.size - offset;
        const auto from = reverse ? tile.dest_address : tile.source_address;
        const auto to   = reverse ? tile.source_address : tile.dest_address;
        const auto sum  = move_span<copy_unroll, true, hash>(
            reinterpret_cast<const global_byte_t*>(from) + offset,
            reinterpret_cast<global_byte_t*>(to) + offset,
            left < piece_bytes ? left : piece_bytes,
            offset / bytes_per_item);
        if constexpr(hash)
        {
            if(threadIdx.x % warpSize == 0) atomicAdd(digests + index, sum);
        }
    }
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
extern "C" __global__ __launch_bounds__(workgroup_size) void __kernel_replay_count_tiles(
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
extern "C" __global__ __launch_bounds__(workgroup_size) void __kernel_replay_scan_tiles(
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
extern "C" __global__ __launch_bounds__(workgroup_size) void __kernel_replay_expand_tiles(
    expand_tiles_args_t args)
{
    const auto* descriptors = reinterpret_cast<const global_descriptor_t*>(args.descriptors);
    const auto* first_tiles = reinterpret_cast<const global_u32_t*>(args.first_tiles);
    const auto  total       = *reinterpret_cast<const global_u32_t*>(args.tile_count);
    auto*       tiles       = reinterpret_cast<global_descriptor_t*>(args.tiles);
    auto*       digests     = reinterpret_cast<global_u64_t*>(args.digests);
    auto*       flags       = reinterpret_cast<global_u8_t*>(args.changed_flags);
    auto*       counters    = reinterpret_cast<global_u32_t*>(args.counters);
    const auto  stride      = std::uint64_t{args.workgroups} * workgroup_size;

    if(args.counters != 0 && blockIdx.x == 0 && threadIdx.x < counter_count)
        counters[threadIdx.x] = 0;
    for(auto tile = std::uint64_t{blockIdx.x} * workgroup_size + threadIdx.x; tile < total;
        tile += stride)
    {
        if(args.digests != 0)
        {
            digests[tile] = 0;
            flags[tile]   = 0;
        }
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

extern "C" __global__ __launch_bounds__(workgroup_size) void __kernel_replay_copy_tiles(
    copy_tiles_args_t args)
{
    copy_tiles<false>(reinterpret_cast<const global_descriptor_t*>(args.tiles),
                      *reinterpret_cast<const global_u32_t*>(args.tile_count),
                      args.workgroups,
                      nullptr,
                      args.reverse != 0);
}

// The capture copy, which also digests every tile it reads. expand zeroed the digests.
extern "C" __global__ __launch_bounds__(workgroup_size) void __kernel_replay_capture_tiles(
    capture_tiles_args_t args)
{
    copy_tiles<true>(reinterpret_cast<const global_descriptor_t*>(args.tiles),
                     *reinterpret_cast<const global_u32_t*>(args.tile_count),
                     args.workgroups,
                     reinterpret_cast<unsigned long long*>(args.digests),
                     false);
}

// After a pass, one wave per captured tile hashes the live bytes. A tile whose digest changed is
// copied back from the snapshot, flagged, and appended to the changed list in no particular order.
// Each lane holds one pending entry, and the waves of a workgroup claim their list slots together
// at the end: contended atomics on one count cost several ns each.
extern "C" __global__ __launch_bounds__(workgroup_size) void __kernel_replay_restore_changed_tiles(
    restore_changed_args_t args)
{
    __shared__ std::uint32_t wave_pending[workgroup_size / 32];
    __shared__ std::uint32_t workgroup_first;

    const auto* tiles      = reinterpret_cast<const global_descriptor_t*>(args.tiles);
    const auto  count      = *reinterpret_cast<const global_u32_t*>(args.tile_count);
    const auto* digests    = reinterpret_cast<const global_u64_t*>(args.digests);
    auto*       flags      = reinterpret_cast<global_u8_t*>(args.changed_flags);
    auto*       changed    = reinterpret_cast<global_u32_t*>(args.changed);
    auto*       counters   = reinterpret_cast<unsigned int*>(args.counters);
    auto*       list_count = counters + args.slot;
    const auto  lane       = threadIdx.x % warpSize;
    const auto  local_wave = threadIdx.x / warpSize;
    const auto  waves      = args.workgroups * (workgroup_size / warpSize);

    if(blockIdx.x == 0 && threadIdx.x == 0) counters[(args.slot + 1) % changed_count_slots] = 0;

    auto pending = std::uint32_t{0};
    auto mine    = std::uint32_t{0};
    for(auto index = wave_index(args.workgroups); index < count; index += waves)
    {
        const auto tile = load_descriptor(tiles, index);
        auto*      live = reinterpret_cast<global_byte_t*>(tile.source_address);
        if(move_span<hash_unroll, false, true>(live, nullptr, tile.size, 0) == digests[index])
            continue;

        move_span<copy_unroll, true, false>(
            reinterpret_cast<const global_byte_t*>(tile.dest_address), live, tile.size, 0);
        if(lane == 0) flags[index] = 1;
        if(lane == pending) mine = index;
        if(++pending < static_cast<std::uint32_t>(warpSize)) continue;
        auto first = 0U;
        if(lane == 0) first = atomicAdd(list_count, pending);
        changed[__shfl(first, 0) + lane] = mine;
        pending                          = 0;
    }

    if(lane == 0) wave_pending[local_wave] = pending;
    __syncthreads();
    if(threadIdx.x == 0)
    {
        auto total = 0U;
        for(auto wave = 0U; wave < workgroup_size / warpSize; ++wave)
            total += wave_pending[wave];
        workgroup_first = total != 0 ? atomicAdd(list_count, total) : 0;
    }
    __syncthreads();
    auto first = workgroup_first;
    for(auto wave = 0U; wave < local_wave; ++wave)
        first += wave_pending[wave];
    if(lane < pending) changed[first + lane] = mine;
}

// Before each later pass, copies the tiles on a changed list back from the snapshot, in pieces
// like the capture when the list is short.
extern "C" __global__ __launch_bounds__(workgroup_size) void __kernel_replay_restore_listed_tiles(
    restore_listed_args_t args)
{
    const auto* tiles       = reinterpret_cast<const global_descriptor_t*>(args.tiles);
    const auto* changed     = reinterpret_cast<const global_u32_t*>(args.changed);
    const auto  count       = *reinterpret_cast<const global_u32_t*>(args.changed_count);
    const auto  waves       = args.workgroups * (workgroup_size / warpSize);
    const auto  split       = split_for(count, waves);
    const auto  piece_bytes = tile_bytes >> split;
    const auto  items       = count << split;

    for(auto item = wave_index(args.workgroups); item < items; item += waves)
    {
        const auto tile   = load_descriptor(tiles, changed[item % count]);
        const auto offset = (item / count) * piece_bytes;
        if(offset >= tile.size) continue;
        const auto left = tile.size - offset;
        move_span<copy_unroll, true, false>(
            reinterpret_cast<const global_byte_t*>(tile.dest_address) + offset,
            reinterpret_cast<global_byte_t*>(tile.source_address) + offset,
            left < piece_bytes ? left : piece_bytes,
            0);
    }
}

// After the last pass, counts the captured tiles off the changed list whose digest changed anyway.
// The restores from that list missed them.
extern "C" __global__ __launch_bounds__(workgroup_size) void __kernel_replay_check_tiles(
    check_tiles_args_t args)
{
    const auto* tiles    = reinterpret_cast<const global_descriptor_t*>(args.tiles);
    const auto  count    = *reinterpret_cast<const global_u32_t*>(args.tile_count);
    const auto* digests  = reinterpret_cast<const global_u64_t*>(args.digests);
    const auto* flags    = reinterpret_cast<const global_u8_t*>(args.changed_flags);
    auto*       counters = reinterpret_cast<unsigned int*>(args.counters);
    const auto  waves    = args.workgroups * (workgroup_size / warpSize);

    auto missed = 0U;
    for(auto index = wave_index(args.workgroups); index < count; index += waves)
    {
        if(flags[index] != 0) continue;
        const auto tile = load_descriptor(tiles, index);
        if(move_span<hash_unroll, false, true>(
               reinterpret_cast<const global_byte_t*>(tile.source_address),
               nullptr,
               tile.size,
               0) != digests[index])
            ++missed;
    }
    if(threadIdx.x % warpSize == 0 && missed != 0) atomicAdd(counters + unlisted_counter, missed);
}
