// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/constraint.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

using rocprofsys::constraint::trace_config;

namespace
{
// Settings the mock policy hands back. Each test assigns the fields it cares
// about; the defaults mirror "nothing configured".
struct mock_settings
{
    double      delay        = 0.0;
    double      duration     = 0.0;
    std::string periods      = {};
    std::string period_clock = "realtime";
};

mock_settings g_settings{};

// Distinct Tag values give each test its own instantiation, so no static state
// inside trace_config can leak between tests.
template <int Tag>
struct mock_externals
{
    static double      get_trace_delay() { return g_settings.delay; }
    static double      get_trace_duration() { return g_settings.duration; }
    static std::string get_trace_periods() { return g_settings.periods; }
    static std::string get_trace_period_clock() { return g_settings.period_clock; }
};

enum externals_tag : int
{
    nothing_configured = 1,
    entry_defaults     = 2,
    entry_overrides    = 3,
    multiple_entries   = 4,
    empty_periods      = 5,
    delay_only         = 6,
    clock_realtime     = 7,
    clock_cputime      = 8,
    clock_default      = 9,
    clock_invalid      = 10,
};

template <int Tag>
using config_for = trace_config<mock_externals<Tag>>;

class constraint_test : public ::testing::Test
{
protected:
    void SetUp() override { g_settings = mock_settings{}; }
    void TearDown() override { g_settings = mock_settings{}; }
};
}  // namespace

TEST_F(constraint_test, get_trace_specs_without_configuration_yields_no_periods)
{
    EXPECT_TRUE(config_for<nothing_configured>::get_trace_specs().empty());
}

TEST_F(constraint_test, get_trace_specs_defaults_missing_period_fields)
{
    g_settings.delay    = 1.0;
    g_settings.duration = 2.0;
    g_settings.periods  = "5";

    const auto _v = config_for<entry_defaults>::get_trace_specs();

    // [0] is the delay/duration window, [1] the parsed TRACE_PERIODS entry.
    ASSERT_EQ(_v.size(), 2u);
    EXPECT_DOUBLE_EQ(_v[1].delay, 5.0);
    EXPECT_DOUBLE_EQ(_v[1].duration, 2.0);
    EXPECT_EQ(_v[1].repeat, 1u);
}

TEST_F(constraint_test, get_trace_specs_overrides_all_period_fields)
{
    g_settings.delay    = 1.0;
    g_settings.duration = 2.0;
    g_settings.periods  = "10:20:3";

    const auto _v = config_for<entry_overrides>::get_trace_specs();

    ASSERT_EQ(_v.size(), 2u);
    EXPECT_DOUBLE_EQ(_v[1].delay, 10.0);
    EXPECT_DOUBLE_EQ(_v[1].duration, 20.0);
    EXPECT_EQ(_v[1].repeat, 3u);
}

TEST_F(constraint_test, get_trace_specs_splits_multiple_period_entries)
{
    g_settings.periods = "1:2 3:4;5:6";

    const auto _v = config_for<multiple_entries>::get_trace_specs();

    // delay/duration are zero, so no leading window is emitted.
    ASSERT_EQ(_v.size(), 3u);
    EXPECT_DOUBLE_EQ(_v[0].delay, 1.0);
    EXPECT_DOUBLE_EQ(_v[0].duration, 2.0);
    EXPECT_DOUBLE_EQ(_v[1].delay, 3.0);
    EXPECT_DOUBLE_EQ(_v[1].duration, 4.0);
    EXPECT_DOUBLE_EQ(_v[2].delay, 5.0);
    EXPECT_DOUBLE_EQ(_v[2].duration, 6.0);
}

TEST_F(constraint_test, get_trace_specs_empty_periods_yields_only_delay_duration_window)
{
    g_settings.delay    = 1.0;
    g_settings.duration = 2.0;

    const auto _v = config_for<empty_periods>::get_trace_specs();

    ASSERT_EQ(_v.size(), 1u);
    EXPECT_DOUBLE_EQ(_v[0].delay, 1.0);
    EXPECT_DOUBLE_EQ(_v[0].duration, 2.0);
    EXPECT_EQ(_v[0].repeat, 1u);
}

TEST_F(constraint_test, get_trace_specs_emits_window_when_only_delay_is_set)
{
    g_settings.delay = 3.0;

    const auto _v = config_for<delay_only>::get_trace_specs();

    ASSERT_EQ(_v.size(), 1u);
    EXPECT_DOUBLE_EQ(_v[0].delay, 3.0);
    EXPECT_DOUBLE_EQ(_v[0].duration, 0.0);
}

TEST_F(constraint_test, get_trace_period_clock_id_maps_realtime)
{
    g_settings.period_clock = "realtime";
    EXPECT_EQ(config_for<clock_realtime>::get_trace_period_clock_id(), CLOCK_REALTIME);
}

TEST_F(constraint_test, get_trace_period_clock_id_maps_cputime)
{
    g_settings.period_clock = "cputime";
    EXPECT_EQ(config_for<clock_cputime>::get_trace_period_clock_id(),
              CLOCK_PROCESS_CPUTIME_ID);
}

TEST_F(constraint_test, get_trace_period_clock_id_defaults_to_realtime)
{
    EXPECT_EQ(config_for<clock_default>::get_trace_period_clock_id(), CLOCK_REALTIME);
}

TEST_F(constraint_test, get_trace_period_clock_id_throws_on_unrecognized_value)
{
    g_settings.period_clock = "bogus";
    EXPECT_THROW((void) config_for<clock_invalid>::get_trace_period_clock_id(),
                 std::runtime_error);
}
