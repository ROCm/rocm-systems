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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/tracing/profiling_time.hpp"

#include <gtest/gtest.h>

TEST(rocprofiler_lib, timestamp)
{
    auto beg = rocprofiler::common::timestamp_ns();
    auto mid = rocprofiler_timestamp_t{};
    auto ret = rocprofiler_get_timestamp(&mid);
    auto end = rocprofiler::common::timestamp_ns();

    EXPECT_EQ(ret, ROCPROFILER_STATUS_SUCCESS);
    EXPECT_GT(beg, 0);
    EXPECT_GT(mid, beg);
    EXPECT_GT(end, mid);
}

TEST(rocprofiler_lib, profiling_time_bounds)
{
    using rocprofiler::tracing::adjust_profiling_time;
    using rocprofiler::tracing::profiling_time;

    // Exercise clock-skew repair even in a strict-timestamps build.
    rocprofiler::common::set_env("ROCPROFILER_CI_STRICT_TIMESTAMPS", "0", 1);
    rocprofiler::common::set_env("ROCPROFILER_CI_FREQ_SCALE_TIMESTAMPS", "0", 1);

    struct test_case
    {
        uint64_t start, end, expected_start, expected_end;
    };
    const test_case cases[] = {
        {120, 180, 120, 180},  // Already within the CPU bounds.
        {80, 140, 100, 160},   // Shift forward, preserving duration.
        {160, 220, 140, 200},  // Shift backward, preserving duration.
        {50, 150, 100, 200},   // Exact-window duration only needs a forward shift.
        {150, 250, 100, 200},  // Exact-window duration only needs a backward shift.
        {50, 250, 100, 200},   // Duration cannot fit: neither bound may be exceeded.
        {0, 250, 100, 200},    // Clamping must repair an unsigned wraparound from the shifts.
    };
    for(const auto& entry : cases)
    {
        SCOPED_TRACE(::testing::Message() << "start=" << entry.start << ", end=" << entry.end);
        auto result =
            adjust_profiling_time("dispatch",
                                  "test",
                                  profiling_time{HSA_STATUS_SUCCESS, entry.start, entry.end},
                                  profiling_time{HSA_STATUS_SUCCESS, 100, 200});
        EXPECT_EQ(result.status, HSA_STATUS_SUCCESS);
        EXPECT_EQ(result.start, entry.expected_start);
        EXPECT_EQ(result.end, entry.expected_end);
    }
}
