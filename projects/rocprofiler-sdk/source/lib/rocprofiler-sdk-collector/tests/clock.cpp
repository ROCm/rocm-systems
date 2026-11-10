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

#include "lib/rocprofiler-sdk-collector/clock.hpp"

#include <gtest/gtest.h>

#include <time.h>

namespace collector = rocprofiler::collector;

TEST(clock, samples_both_clocks)
{
    auto now    = timespec{};
    auto sample = collector::sample_clocks();
    ::clock_gettime(CLOCK_MONOTONIC, &now);
    auto monotonic = (static_cast<uint64_t>(now.tv_sec) * 1000000000) + now.tv_nsec;
    EXPECT_LE(sample.monotonic, monotonic);
    EXPECT_GT(sample.monotonic, 0u);
    // BOOTTIME is MONOTONIC plus the time suspended.
    EXPECT_GE(sample.boottime + 1000000, sample.monotonic);
}

TEST(clock, converts_monotonic_to_boottime)
{
    auto start      = collector::clock_sample{1000, 5000};
    auto stop       = collector::clock_sample{900000, 904100};  // offset within 1 ms
    auto conversion = collector::get_clock_conversion(start, stop);
    EXPECT_EQ(conversion.offset, 4000);
    EXPECT_TRUE(conversion.stable);
    EXPECT_EQ(collector::to_boottime(conversion, 2000), 6000u);

    // Suspended for 10 s while tracing.
    auto suspended = collector::clock_sample{900000, 10000904000};
    conversion     = collector::get_clock_conversion(start, suspended);
    EXPECT_EQ(conversion.offset, 4000);
    EXPECT_FALSE(conversion.stable);

    auto real =
        collector::get_clock_conversion(collector::sample_clocks(), collector::sample_clocks());
    EXPECT_TRUE(real.stable);
}
