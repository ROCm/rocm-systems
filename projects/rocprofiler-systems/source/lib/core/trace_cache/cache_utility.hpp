// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include "sample_type.hpp"
#include <array>
#include <string>
#include <unistd.h>

namespace rocprofsys
{
namespace trace_cache
{
constexpr size_t MByte           = 1024 * 1024;
constexpr size_t GByte           = 1024 * 1024 * 1024;
constexpr size_t buffer_size     = 100 * MByte;
constexpr size_t flush_threshold = 80 * MByte;
const auto       filename = "/tmp/buffered_storage_" + std::to_string(getpid()) + ".bin";

constexpr size_t minimal_fragmented_memory_size = sizeof(entry_type) + sizeof(size_t);
using buffer_array_t                            = std::array<uint8_t, buffer_size>;

}  // namespace trace_cache
}  // namespace rocprofsys
