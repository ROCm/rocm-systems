// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/config/trace_period_config.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

using rocprofsys::config::trace_config;
using rocprofsys::config::trace_period_settings;

namespace
{
trace_period_settings g_settings{};

// Distinct Tag values give each test its own instantiation, so no static state
// inside trace_config can leak between tests.
template <int Tag>
struct mock_externals
{
    static trace_period_settings get_trace_period_settings() { return g_settings; }
};

enum externals_tag : int
{
    nothing_configured            = 1,
    multiple_entries              = 2,
    empty_periods                 = 3,
    delay_only                    = 4,
    clock_realtime                = 5,
    clock_cputime                 = 6,
    clock_default                 = 7,
    clock_invalid                 = 8,
    both_delay_and_periods        = 9,
    negative_delay                = 10,
    negative_duration             = 11,
    period_entry_with_extra_field = 12,
};

template <int Tag>
using config_for = trace_config<mock_externals<Tag>>;

class trace_period_config_test : public ::testing::Test
{
protected:
    void SetUp() override { g_settings = trace_period_settings{}; }
    void TearDown() override { g_settings = trace_period_settings{}; }
};
}  // namespace

TEST_F(trace_period_config_test, get_trace_specs_without_configuration_yields_no_periods)
{
    EXPECT_TRUE(config_for<nothing_configured>::get_trace_specs().empty());
}

TEST_F(trace_period_config_test, get_trace_specs_splits_multiple_period_entries)
{
    g_settings.periods = "1:2 3:4;5:6";

    const auto _v = config_for<multiple_entries>::get_trace_specs();

    ASSERT_EQ(_v.size(), 3u);
    EXPECT_DOUBLE_EQ(_v[0].delay, 1.0);
    EXPECT_DOUBLE_EQ(_v[0].duration, 2.0);
    EXPECT_DOUBLE_EQ(_v[1].delay, 3.0);
    EXPECT_DOUBLE_EQ(_v[1].duration, 4.0);
    EXPECT_DOUBLE_EQ(_v[2].delay, 5.0);
    EXPECT_DOUBLE_EQ(_v[2].duration, 6.0);
}

TEST_F(trace_period_config_test,
       get_trace_specs_empty_periods_yields_only_delay_duration_window)
{
    g_settings.delay    = 1.0;
    g_settings.duration = 2.0;

    const auto _v = config_for<empty_periods>::get_trace_specs();

    ASSERT_EQ(_v.size(), 1u);
    EXPECT_DOUBLE_EQ(_v[0].delay, 1.0);
    EXPECT_DOUBLE_EQ(_v[0].duration, 2.0);
    EXPECT_EQ(_v[0].repeat, 1u);
}

TEST_F(trace_period_config_test, get_trace_specs_emits_window_when_only_delay_is_set)
{
    g_settings.delay = 3.0;

    const auto _v = config_for<delay_only>::get_trace_specs();

    ASSERT_EQ(_v.size(), 1u);
    EXPECT_DOUBLE_EQ(_v[0].delay, 3.0);
    EXPECT_DOUBLE_EQ(_v[0].duration, 0.0);
}

TEST_F(trace_period_config_test,
       get_trace_specs_throws_when_delay_duration_and_periods_are_both_set)
{
    g_settings.delay    = 1.0;
    g_settings.duration = 2.0;
    g_settings.periods  = "5";

    EXPECT_THROW((void) config_for<both_delay_and_periods>::get_trace_specs(),
                 std::runtime_error);
}

TEST_F(trace_period_config_test, get_trace_specs_throws_on_negative_delay)
{
    g_settings.delay = -1.0;

    EXPECT_THROW((void) config_for<negative_delay>::get_trace_specs(),
                 std::runtime_error);
}

TEST_F(trace_period_config_test, get_trace_specs_throws_on_negative_duration)
{
    g_settings.duration = -1.0;

    EXPECT_THROW((void) config_for<negative_duration>::get_trace_specs(),
                 std::runtime_error);
}

TEST_F(trace_period_config_test, get_trace_specs_ignores_period_fields_past_the_third)
{
    g_settings.periods = "5:10:3:99";

    const auto _v = config_for<period_entry_with_extra_field>::get_trace_specs();

    ASSERT_EQ(_v.size(), 1u);
    EXPECT_DOUBLE_EQ(_v[0].delay, 5.0);
    EXPECT_DOUBLE_EQ(_v[0].duration, 10.0);
    EXPECT_EQ(_v[0].repeat, 3u);
}

TEST_F(trace_period_config_test, get_trace_period_clock_id_maps_realtime)
{
    g_settings.period_clock = "realtime";
    EXPECT_EQ(config_for<clock_realtime>::get_trace_period_clock_id(), CLOCK_REALTIME);
}

TEST_F(trace_period_config_test, get_trace_period_clock_id_maps_cputime)
{
    g_settings.period_clock = "cputime";
    EXPECT_EQ(config_for<clock_cputime>::get_trace_period_clock_id(),
              CLOCK_PROCESS_CPUTIME_ID);
}

TEST_F(trace_period_config_test, get_trace_period_clock_id_defaults_to_realtime)
{
    EXPECT_EQ(config_for<clock_default>::get_trace_period_clock_id(), CLOCK_REALTIME);
}

TEST_F(trace_period_config_test, get_trace_period_clock_id_throws_on_unrecognized_value)
{
    g_settings.period_clock = "bogus";
    EXPECT_THROW((void) config_for<clock_invalid>::get_trace_period_clock_id(),
                 std::runtime_error);
}
