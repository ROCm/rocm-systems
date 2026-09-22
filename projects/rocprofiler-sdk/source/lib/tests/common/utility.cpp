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

#include "lib/common/utility.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>

namespace common = ::rocprofiler::common;

TEST(common, timestamp_ns_monotonic)
{
    auto _last = common::timestamp_ns();
    for(size_t i = 0; i < 10000; ++i)
    {
        auto _curr = common::timestamp_ns();
        ASSERT_GE(_curr, _last) << "timestamp_ns() went backwards at iteration " << i;
        _last = _curr;
    }
}

TEST(common, timestamp_ns_elapsed)
{
    constexpr uint64_t _sleep_ms = 100;

    auto _beg = common::timestamp_ns();
    std::this_thread::sleep_for(std::chrono::milliseconds{_sleep_ms});
    auto _elapsed_ms = (common::timestamp_ns() - _beg) / 1000000;

    // a wide upper bound so a loaded CI machine cannot fail this; the point is to catch a
    // unit/scale error in the conversion, not to measure scheduler latency
    EXPECT_GE(_elapsed_ms, _sleep_ms);
    EXPECT_LT(_elapsed_ms, 100 * _sleep_ms);
}

#if defined(_WIN32)
// get_ticks() converts the QPC counter to nanoseconds. The counter runs at ~1e7 Hz and counts
// from boot, so a naive `ticks * 1e9 / freq` overflows uint64_t after ~31 minutes of uptime and
// silently wraps. Cross-check against GetTickCount64(), which is also milliseconds since boot
// but is not derived from QPC.
TEST(common, timestamp_ns_matches_uptime)
{
    auto _uptime_ms    = static_cast<uint64_t>(::GetTickCount64());
    auto _timestamp_ms = common::timestamp_ns() / 1000000;

    auto _diff_ms =
        (_timestamp_ms > _uptime_ms) ? (_timestamp_ms - _uptime_ms) : (_uptime_ms - _timestamp_ms);

    // the two clocks drift, and QPC does not advance while the system is suspended, so only
    // require the same order of magnitude -- a wrapped conversion is wrong by several
    auto _tolerance_ms = std::max<uint64_t>(5000, _uptime_ms / 20);

    EXPECT_LT(_diff_ms, _tolerance_ms)
        << "timestamp_ns() reports " << _timestamp_ms << " ms since boot but GetTickCount64() "
        << "reports " << _uptime_ms << " ms";
}
#endif
