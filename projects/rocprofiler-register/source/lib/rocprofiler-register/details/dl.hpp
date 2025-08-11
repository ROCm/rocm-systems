// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <dlfcn.h>
#include <sys/types.h>
#include <unistd.h>

namespace rocprofiler_register
{
namespace binary
{
using open_modes_vec_t = std::vector<int>;

struct address_range
{
    uintptr_t start = 0;
    uintptr_t last  = 0;
};

struct segment_address_ranges
{
    std::string                filepath = {};
    std::vector<address_range> ranges   = {};
};

std::vector<segment_address_ranges>
get_segment_addresses(pid_t _pid = getpid());

// helper function for translating generic lib name to resolved path
std::optional<std::string>
get_linked_path(std::string_view, open_modes_vec_t&& = {});
}  // namespace binary
}  // namespace rocprofiler_register
