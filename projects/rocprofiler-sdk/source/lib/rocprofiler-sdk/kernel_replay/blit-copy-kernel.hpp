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
inline constexpr auto workgroup_size = std::uint16_t{1024};
inline constexpr auto bytes_per_tile = bytes_per_item * workgroup_size;

struct copy_descriptor_t
{
    std::uint64_t source_address      = 0;
    std::uint64_t destination_address = 0;
    std::uint64_t size                = 0;
    std::uint64_t first_tile          = 0;
    std::uint64_t tile_count          = 0;
};

struct kernel_args_t
{
    std::uint64_t descriptors_address = 0;
    std::uint64_t descriptor_count    = 0;
    std::uint64_t total_tiles         = 0;
    std::uint64_t launched_tiles      = 0;
};

static_assert(sizeof(copy_descriptor_t) == 40);
static_assert(sizeof(kernel_args_t) == 32);
}  // namespace kernel_abi
}  // namespace blit
}  // namespace kernel_replay
}  // namespace rocprofiler
