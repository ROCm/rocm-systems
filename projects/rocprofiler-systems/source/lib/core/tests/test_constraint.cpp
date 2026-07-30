// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/constraint.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

using rocprofsys::constraint::parse_trace_period_clock_id;
using rocprofsys::constraint::parse_trace_period_entry;
using rocprofsys::constraint::parse_trace_periods;

TEST(constraint_test, parse_trace_period_entry_defaults_missing_fields)
{
    const auto _s = parse_trace_period_entry("5", 1.0, 2.0);
    EXPECT_DOUBLE_EQ(_s.delay, 5.0);
    EXPECT_DOUBLE_EQ(_s.duration, 2.0);
    EXPECT_EQ(_s.repeat, 1u);
}

TEST(constraint_test, parse_trace_period_entry_overrides_all_fields)
{
    const auto _s = parse_trace_period_entry("10:20:3", 1.0, 2.0);
    EXPECT_DOUBLE_EQ(_s.delay, 10.0);
    EXPECT_DOUBLE_EQ(_s.duration, 20.0);
    EXPECT_EQ(_s.repeat, 3u);
}

TEST(constraint_test, parse_trace_periods_splits_multiple_entries)
{
    const auto _v = parse_trace_periods("1:2 3:4;5:6", 0.0, 0.0);
    ASSERT_EQ(_v.size(), 3u);
    EXPECT_DOUBLE_EQ(_v[0].delay, 1.0);
    EXPECT_DOUBLE_EQ(_v[0].duration, 2.0);
    EXPECT_DOUBLE_EQ(_v[1].delay, 3.0);
    EXPECT_DOUBLE_EQ(_v[1].duration, 4.0);
    EXPECT_DOUBLE_EQ(_v[2].delay, 5.0);
    EXPECT_DOUBLE_EQ(_v[2].duration, 6.0);
}

TEST(constraint_test, parse_trace_periods_empty_string_yields_no_entries)
{
    EXPECT_TRUE(parse_trace_periods("", 1.0, 2.0).empty());
}

TEST(constraint_test, parse_trace_period_clock_id_maps_realtime)
{
    EXPECT_EQ(parse_trace_period_clock_id("realtime"), CLOCK_REALTIME);
}

TEST(constraint_test, parse_trace_period_clock_id_maps_cputime)
{
    EXPECT_EQ(parse_trace_period_clock_id("cputime"), CLOCK_PROCESS_CPUTIME_ID);
}

TEST(constraint_test, parse_trace_period_clock_id_throws_on_unrecognized_value)
{
    EXPECT_THROW((void) parse_trace_period_clock_id("bogus"), std::runtime_error);
}
