// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/clocks/timeline.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <ratio>
#include <time.h>

namespace
{
namespace timeline = rocprofsys::clocks::timeline;

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
    EXPECT_EQ(timeline::k_clock_id, CLOCK_BOOTTIME);
}

TEST(timeline_clock_test, reads_the_timeline_clock)
{
    const auto before_duration = read_boottime_ns();
    const auto now_duration    = timeline::now();
    const auto after_duration  = read_boottime_ns();
    EXPECT_LE(before_duration, static_cast<std::uint64_t>(now_duration.count()));
    EXPECT_LE(static_cast<std::uint64_t>(now_duration.count()), after_duration);

    const auto before_ns = read_boottime_ns();
    const auto now_ns    = timeline::now_ns();
    const auto after_ns  = read_boottime_ns();
    EXPECT_LE(before_ns, now_ns);
    EXPECT_LE(now_ns, after_ns);
}
