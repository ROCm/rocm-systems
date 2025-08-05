// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/common/defines.hpp"
#include "lib/common/mpl.hpp"

#include <array>
#include <cstdint>
#include <random>
#include <string>

namespace rocprofiler
{
namespace common
{
uint64_t
get_process_start_ticks_since_boot(pid_t pid);

// use this function to create a deterministic random number seed for the system and process
uint64_t
compute_system_seed(std::string_view machine_id, pid_t pid, pid_t ppid, uint64_t pstart_ticks);

std::string
generate_uuid_v7(uint64_t         timestamp_ns,
                 uint64_t         seed  = std::random_device{}(),
                 std::string_view delim = "-");
}  // namespace common
}  // namespace rocprofiler
