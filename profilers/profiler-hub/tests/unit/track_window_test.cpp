// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "track_window.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace
{

using profiler_hub::make_window_filter;

constexpr auto int64_max = static_cast<size_t>(std::numeric_limits<std::int64_t>::max());

TEST(track_window_test, both_bounds_zero_means_no_filter)
{
    const auto filter = make_window_filter(0, 0);

    EXPECT_FALSE(filter.time_window.start.has_value());
    EXPECT_FALSE(filter.time_window.end.has_value());
}

TEST(track_window_test, only_end_zero_means_open_end)
{
    const auto filter = make_window_filter(5, 0);

    EXPECT_EQ(filter.time_window.start, 5U);
    EXPECT_EQ(filter.time_window.end, int64_max);
}

TEST(track_window_test, only_start_zero_is_a_window_from_zero)
{
    const auto filter = make_window_filter(0, 9);

    EXPECT_EQ(filter.time_window.start, 0U);
    EXPECT_EQ(filter.time_window.end, 9U);
}

TEST(track_window_test, normal_bounds_pass_through)
{
    const auto filter = make_window_filter(10, 20);

    EXPECT_EQ(filter.time_window.start, 10U);
    EXPECT_EQ(filter.time_window.end, 20U);
}

TEST(track_window_test, bounds_above_int64_max_are_clamped)
{
    const auto filter = make_window_filter(UINT64_MAX, UINT64_MAX);

    EXPECT_EQ(filter.time_window.start, int64_max);
    EXPECT_EQ(filter.time_window.end, int64_max);
}

TEST(track_window_test, bounds_equal_to_int64_max_are_kept)
{
    const auto filter = make_window_filter(int64_max, int64_max);

    EXPECT_EQ(filter.time_window.start, int64_max);
    EXPECT_EQ(filter.time_window.end, int64_max);
}

}  // namespace
