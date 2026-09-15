// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/time.hpp"

#include <chrono>
#include <cstdint>
#include <gtest/gtest.h>
#include <time.h>
#include <type_traits>

namespace
{
std::uint64_t
get_boottime_ns()
{
    timespec ts{};
    EXPECT_EQ(clock_gettime(CLOCK_BOOTTIME, &ts), 0);
    return static_cast<std::uint64_t>(ts.tv_sec) * std::nano::den +
           static_cast<std::uint64_t>(ts.tv_nsec);
}
}  // namespace

TEST(TimeTest, TimelineUsesClockBoottime)
{
    const auto before = get_boottime_ns();
    const auto now    = rocprofsys::common::time::timeline_ns();
    const auto after  = get_boottime_ns();

    EXPECT_LE(before, now);
    EXPECT_LE(now, after);
}

TEST(TimeTest, TimelineClockMeetsChronoClockRequirements)
{
    using clock_type = rocprofsys::common::time::timeline_clock;

    static_assert(clock_type::is_steady);
    static_assert(std::is_same_v<clock_type::duration, std::chrono::nanoseconds>);

    const auto before = clock_type::now();
    const auto after  = clock_type::now();
    EXPECT_LE(before, after);
}
