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
using rocprofiler::kernel_replay::blit::kernel_abi::first_local_item;
using rocprofiler::kernel_replay::blit::kernel_abi::items_for_size;
using rocprofiler::kernel_replay::blit::kernel_abi::kernel_args_t;
using rocprofiler::kernel_replay::blit::kernel_abi::workgroup_size;

using copy_vector_t       = std::uint32_t __attribute__((ext_vector_type(4)));
using global_byte_t       = std::byte __attribute__((address_space(1)));
using global_descriptor_t = copy_descriptor_t __attribute__((address_space(1)));

static_assert(bytes_per_item == sizeof(copy_vector_t));

__device__ __forceinline__ void
copy_item(const copy_descriptor_t& copy, std::uint64_t item_offset)
{
    const auto  byte_offset = item_offset * bytes_per_item;
    const auto* src_bytes   = reinterpret_cast<const global_byte_t*>(copy.source_address);
    auto*       dst_bytes   = reinterpret_cast<global_byte_t*>(copy.destination_address);

    if(copy.size - byte_offset >= bytes_per_item)
    {
        auto value = copy_vector_t{};
        __builtin_memcpy(&value, src_bytes + byte_offset, sizeof(value));
        __builtin_memcpy(dst_bytes + byte_offset, &value, sizeof(value));
        return;
    }

    for(auto i = byte_offset; i < copy.size; ++i)
        dst_bytes[i] = src_bytes[i];
}

extern "C" __global__ __launch_bounds__(workgroup_size) void kernel_replay_blit(kernel_args_t args)
{
    const auto* descriptors =
        reinterpret_cast<const global_descriptor_t*>(args.descriptors_address);
    const auto thread = static_cast<std::uint64_t>(threadIdx.x) +
                        static_cast<std::uint64_t>(workgroup_size) * blockIdx.x;
    const auto stride = args.launched_items;

    auto logical_base = std::uint64_t{0};
    for(auto region = std::uint64_t{0}; region < args.descriptor_count; ++region)
    {
        auto copy = copy_descriptor_t{};
        __builtin_memcpy(&copy, descriptors + region, sizeof(copy));
        const auto item_count = items_for_size(copy.size);
        // Thread `t` owns compact logical items `t + k * stride`. Rotate that assignment by this
        // region's prefix to obtain its first physical 16-byte item in the discontiguous
        // allocation.
        auto item_offset = first_local_item(thread, stride, logical_base);

        while(item_offset < item_count)
        {
            copy_item(copy, item_offset);
            if(item_count - item_offset <= stride) break;
            item_offset += stride;
        }
        logical_base += item_count;
    }
}
