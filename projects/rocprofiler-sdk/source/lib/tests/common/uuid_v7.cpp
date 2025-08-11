// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/uuid_v7.hpp"
#include "lib/common/utility.hpp"
#include "lib/output/node_info.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

TEST(common, uuid_v7)
{
    namespace common = ::rocprofiler::common;
    namespace tool   = ::rocprofiler::tool;

    auto        _node_data = tool::read_node_info();
    const auto& _mach_id   = _node_data.machine_id;
    auto        _this_pid  = getpid();
    auto        _this_ppid = getppid();
    auto        _init_ns   = common::timestamp_ns();
    auto        _ticks     = common::get_process_start_ticks_since_boot(_this_pid);

    auto get_uuid_v7 = [&_mach_id, &_this_pid, &_this_ppid, &_ticks](auto _timestamp_ns) {
        auto _seed = common::compute_system_seed(_mach_id, _this_pid, _this_ppid, _ticks);
        return common::generate_uuid_v7(_timestamp_ns, _seed);
    };

    auto reference_uuid_v7 = get_uuid_v7(_init_ns);

    // uuid v7 has millisecond precision
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    for(size_t i = 0; i < 1000; ++i)
    {
        auto varying_ns      = common::timestamp_ns();
        auto varying_uuid_v7 = get_uuid_v7(varying_ns);

        EXPECT_NE(reference_uuid_v7, varying_uuid_v7)
            << fmt::format("machine-id={}, pid={}, ppid={}, ticks={}, init_ns={}, varying_ns={}",
                           _mach_id,
                           _this_pid,
                           _this_ppid,
                           _ticks,
                           _init_ns,
                           varying_ns);

        // UUIDv7 is supposed to be lexicographically sortable
        EXPECT_LT(reference_uuid_v7, varying_uuid_v7)
            << fmt::format("machine-id={}, pid={}, ppid={}, ticks={}, init_ns={}, varying_ns={}",
                           _mach_id,
                           _this_pid,
                           _this_ppid,
                           _ticks,
                           _init_ns,
                           varying_ns);
    }
}
