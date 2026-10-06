// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/control/clocks/timeline.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <ratio>
#include <time.h>

namespace
{
using rocprofsys::control::clocks::timeline_available;
using rocprofsys::control::clocks::timeline_now;
using rocprofsys::control::clocks::timeline_ns;

std::uint64_t
read_boottime_ns()
{
    timespec specs{};
    EXPECT_EQ(clock_gettime(CLOCK_BOOTTIME, &specs), 0);
    return static_cast<std::uint64_t>(specs.tv_sec) * std::nano::den +
           static_cast<std::uint64_t>(specs.tv_nsec);
}
}  // namespace

TEST(timeline_clock_test, available_on_linux) { EXPECT_TRUE(timeline_available()); }

TEST(timeline_clock_test, uses_clock_boottime)
{
    ASSERT_TRUE(timeline_available());
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
