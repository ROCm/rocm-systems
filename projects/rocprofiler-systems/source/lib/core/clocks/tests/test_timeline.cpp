// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/clocks/timeline.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <ratio>
#include <time.h>

namespace
{
using rocprofsys::clocks::k_timeline_clock_id;
using rocprofsys::clocks::timeline_now;
using rocprofsys::clocks::timeline_ns;

std::uint64_t
read_boottime_ns()
{
    timespec specs{};
    EXPECT_EQ(clock_gettime(CLOCK_BOOTTIME, &specs), 0);
    return static_cast<std::uint64_t>(specs.tv_sec) * std::nano::den +
           static_cast<std::uint64_t>(specs.tv_nsec);
}
}  // namespace

TEST(timeline_clock_test, matches_rocprofiler_sdk_host_clock)
{
    EXPECT_EQ(k_timeline_clock_id, CLOCK_BOOTTIME);
}

TEST(timeline_clock_test, reads_the_timeline_clock)
{
    const auto before_duration = read_boottime_ns();
    const auto now_duration    = timeline_now();
    const auto after_duration  = read_boottime_ns();
    EXPECT_LE(before_duration, static_cast<std::uint64_t>(now_duration.count()));
    EXPECT_LE(static_cast<std::uint64_t>(now_duration.count()), after_duration);

    const auto before_ns = read_boottime_ns();
    const auto now_ns    = timeline_ns();
    const auto after_ns  = read_boottime_ns();
    EXPECT_LE(before_ns, now_ns);
    EXPECT_LE(now_ns, after_ns);
}
