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
constexpr double SAMPLE_DURATION_SECONDS   = 2.0;
constexpr double SAMPLE_DELAY_ONLY_SECONDS = 3.0;

// Function-local static so initialization happens on first use rather than
// before main - avoids static initialization order concerns for a
// namespace-scope trace_period_settings.
trace_period_settings&
mock_settings()
{
    static trace_period_settings g_settings{};
    return g_settings;
}

enum class externals_tag : int
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

// Distinct Tag values give each test its own instantiation, so no static state
// inside trace_config can leak between tests.
template <externals_tag Tag>
struct mock_externals
{
    static trace_period_settings get_trace_period_settings() { return mock_settings(); }
};

template <externals_tag Tag>
using config_for = trace_config<mock_externals<Tag>>;

class trace_period_config_test : public ::testing::Test
{
protected:
    void SetUp() override { mock_settings() = trace_period_settings{}; }
    void TearDown() override { mock_settings() = trace_period_settings{}; }
};
}  // namespace

TEST_F(trace_period_config_test, get_trace_specs_without_configuration_yields_no_periods)
{
    EXPECT_TRUE(config_for<externals_tag::nothing_configured>::get_trace_specs().empty());
}

TEST_F(trace_period_config_test, get_trace_specs_splits_multiple_period_entries)
{
    mock_settings().periods = "1:2 3:4;5:6";

    const auto parsed = config_for<externals_tag::multiple_entries>::get_trace_specs();

    ASSERT_EQ(parsed.size(), 3u);
    EXPECT_DOUBLE_EQ(parsed[0].delay, 1.0);
    EXPECT_DOUBLE_EQ(parsed[0].duration, 2.0);
    EXPECT_DOUBLE_EQ(parsed[1].delay, 3.0);
    EXPECT_DOUBLE_EQ(parsed[1].duration, 4.0);
    EXPECT_DOUBLE_EQ(parsed[2].delay, 5.0);
    EXPECT_DOUBLE_EQ(parsed[2].duration, 6.0);
}

TEST_F(trace_period_config_test,
       get_trace_specs_empty_periods_yields_only_delay_duration_window)
{
    mock_settings().delay    = 1.0;
    mock_settings().duration = SAMPLE_DURATION_SECONDS;

    const auto parsed = config_for<externals_tag::empty_periods>::get_trace_specs();

    ASSERT_EQ(parsed.size(), 1u);
    EXPECT_DOUBLE_EQ(parsed[0].delay, 1.0);
    EXPECT_DOUBLE_EQ(parsed[0].duration, SAMPLE_DURATION_SECONDS);
    EXPECT_EQ(parsed[0].repeat, 1u);
}

TEST_F(trace_period_config_test, get_trace_specs_emits_window_when_only_delay_is_set)
{
    mock_settings().delay = SAMPLE_DELAY_ONLY_SECONDS;

    const auto parsed = config_for<externals_tag::delay_only>::get_trace_specs();

    ASSERT_EQ(parsed.size(), 1u);
    EXPECT_DOUBLE_EQ(parsed[0].delay, SAMPLE_DELAY_ONLY_SECONDS);
    EXPECT_DOUBLE_EQ(parsed[0].duration, 0.0);
}

TEST_F(trace_period_config_test,
       get_trace_specs_throws_when_delay_duration_and_periods_are_both_set)
{
    mock_settings().delay    = 1.0;
    mock_settings().duration = SAMPLE_DURATION_SECONDS;
    mock_settings().periods  = "5";

    EXPECT_THROW(
        (void) config_for<externals_tag::both_delay_and_periods>::get_trace_specs(),
        std::runtime_error);
}

TEST_F(trace_period_config_test, get_trace_specs_throws_on_negative_delay)
{
    mock_settings().delay = -1.0;

    EXPECT_THROW((void) config_for<externals_tag::negative_delay>::get_trace_specs(),
                 std::runtime_error);
}

TEST_F(trace_period_config_test, get_trace_specs_throws_on_negative_duration)
{
    mock_settings().duration = -1.0;

    EXPECT_THROW((void) config_for<externals_tag::negative_duration>::get_trace_specs(),
                 std::runtime_error);
}

TEST_F(trace_period_config_test, get_trace_specs_ignores_period_fields_past_the_third)
{
    mock_settings().periods = "5:10:3:99";

    const auto parsed =
        config_for<externals_tag::period_entry_with_extra_field>::get_trace_specs();

    ASSERT_EQ(parsed.size(), 1u);
    EXPECT_DOUBLE_EQ(parsed[0].delay, 5.0);
    EXPECT_DOUBLE_EQ(parsed[0].duration, 10.0);
    EXPECT_EQ(parsed[0].repeat, 3u);
}

TEST_F(trace_period_config_test, get_trace_period_clock_id_maps_realtime)
{
    mock_settings().period_clock = "realtime";
    EXPECT_EQ(config_for<externals_tag::clock_realtime>::get_trace_period_clock_id(),
              CLOCK_REALTIME);  // NOLINT(misc-include-cleaner)
}

TEST_F(trace_period_config_test, get_trace_period_clock_id_maps_cputime)
{
    mock_settings().period_clock = "cputime";
    EXPECT_EQ(config_for<externals_tag::clock_cputime>::get_trace_period_clock_id(),
              CLOCK_PROCESS_CPUTIME_ID);  // NOLINT(misc-include-cleaner)
}

TEST_F(trace_period_config_test, get_trace_period_clock_id_defaults_to_realtime)
{
    EXPECT_EQ(config_for<externals_tag::clock_default>::get_trace_period_clock_id(),
              CLOCK_REALTIME);
}

TEST_F(trace_period_config_test, get_trace_period_clock_id_throws_on_unrecognized_value)
{
    mock_settings().period_clock = "bogus";
    EXPECT_THROW(
        (void) config_for<externals_tag::clock_invalid>::get_trace_period_clock_id(),
        std::runtime_error);
}
