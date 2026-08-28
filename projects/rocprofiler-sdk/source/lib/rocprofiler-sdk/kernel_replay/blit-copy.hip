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
using rocprofiler::kernel_replay::blit::kernel_abi::bytes_per_tile;
using rocprofiler::kernel_replay::blit::kernel_abi::copy_descriptor_t;
using rocprofiler::kernel_replay::blit::kernel_abi::kernel_args_t;
using rocprofiler::kernel_replay::blit::kernel_abi::workgroup_size;

using copy_vector_t       = std::uint32_t __attribute__((ext_vector_type(4)));
using global_byte_t       = std::byte __attribute__((address_space(1)));
using global_descriptor_t = copy_descriptor_t __attribute__((address_space(1)));

static_assert(bytes_per_item == sizeof(copy_vector_t));

__device__ __forceinline__ copy_descriptor_t
find_descriptor(const global_descriptor_t* descriptors,
                std::uint64_t              descriptor_count,
                std::uint64_t              tile)
{
    auto first = std::uint64_t{0};
    auto last  = descriptor_count;
    while(first < last)
    {
        const auto middle     = first + (last - first) / 2;
        const auto descriptor = descriptors[middle];
        if(tile < descriptor.first_tile + descriptor.tile_count)
            last = middle;
        else
            first = middle + 1;
    }
    return descriptors[first];
}

__device__ __forceinline__ void
copy_tile(const kernel_args_t& args, std::uint64_t tile)
{
    const auto* descriptors =
        reinterpret_cast<const global_descriptor_t*>(args.descriptors_address);
    const auto copy        = find_descriptor(descriptors, args.descriptor_count, tile);
    const auto byte_offset = (tile - copy.first_tile) * bytes_per_tile +
                             static_cast<std::uint64_t>(threadIdx.x) * bytes_per_item;

    const auto* src = reinterpret_cast<const global_byte_t*>(copy.source_address);
    auto*       dst = reinterpret_cast<global_byte_t*>(copy.destination_address);

    if(byte_offset + sizeof(copy_vector_t) <= copy.size)
    {
        auto value = copy_vector_t{};
        __builtin_memcpy(&value, src + byte_offset, sizeof(value));
        __builtin_memcpy(dst + byte_offset, &value, sizeof(value));
        return;
    }

    if(byte_offset < copy.size)
    {
        for(auto i = byte_offset; i < copy.size; ++i)
            dst[i] = src[i];
    }
}

extern "C" __global__ __launch_bounds__(workgroup_size) void kernel_replay_blit(kernel_args_t args)
{
    const auto tile = static_cast<std::uint64_t>(blockIdx.x);
    if(tile < args.total_tiles) copy_tile(args, tile);
}

extern "C" __global__ __launch_bounds__(workgroup_size) void kernel_replay_blit_stride(
    kernel_args_t args)
{
    for(auto tile = static_cast<std::uint64_t>(blockIdx.x); tile < args.total_tiles;
        tile += args.launched_tiles)
        copy_tile(args, tile);
}
