// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/common/settings.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace rocprofsys::pmc::collectors::testing
{

TEST(settings_policy_test, parse_enabled_metrics_accepts_spaces_after_separators)
{
    constexpr std::uint32_t k_busy_temp_power_mem_usage = 0xFFU;

    const auto metrics =
        settings_policy::parse_enabled_metrics("busy, temp, power, mem_usage");

    EXPECT_EQ(metrics.value, k_busy_temp_power_mem_usage);
}

TEST(settings_policy_test, parse_enabled_metrics_accepts_spaces_around_semicolon)
{
    constexpr std::uint32_t k_busy_temp = 0xF8U;

    const auto metrics = settings_policy::parse_enabled_metrics("busy ; temp");

    EXPECT_EQ(metrics.value, k_busy_temp);
}

TEST(settings_policy_test, parse_enabled_metrics_accepts_tabs_and_mixed_whitespace)
{
    constexpr std::uint32_t k_busy_temp_power = 0xFBU;

    const auto metrics = settings_policy::parse_enabled_metrics("busy,\ttemp \t;  power");

    EXPECT_EQ(metrics.value, k_busy_temp_power);
}

TEST(settings_policy_test, parse_enabled_metrics_trims_leading_and_trailing_space)
{
    constexpr std::uint32_t k_busy_temp = 0xF8U;

    const auto metrics = settings_policy::parse_enabled_metrics("  busy, temp  ");

    EXPECT_EQ(metrics.value, k_busy_temp);
}

TEST(settings_policy_test, parse_enabled_metrics_is_case_insensitive)
{
    constexpr std::uint32_t k_busy_temp = 0xF8U;

    const auto metrics = settings_policy::parse_enabled_metrics("BUSY, Temp");

    EXPECT_EQ(metrics.value, k_busy_temp);
}

TEST(settings_policy_test, parse_enabled_metrics_invalid_input_enables_all)
{
    for(const char* input :
        { "busy,,temp", "busy temp", "busy,", ",busy", "busy, ,temp" })
    {
        SCOPED_TRACE(input);
        EXPECT_EQ(settings_policy::parse_enabled_metrics(input).value,
                  ENABLE_ALL_METRICS);
    }
}

TEST(settings_policy_test,
     parse_enabled_metrics_unknown_token_after_separator_enables_all)
{
    for(const char* input : { "busy, bogus", "busy ;\tbogus", "bogus, busy" })
    {
        SCOPED_TRACE(input);
        EXPECT_EQ(settings_policy::parse_enabled_metrics(input).value,
                  ENABLE_ALL_METRICS);
    }
}

TEST(settings_policy_test, parse_enabled_metrics_all_none_and_empty)
{
    EXPECT_EQ(settings_policy::parse_enabled_metrics("all").value, ENABLE_ALL_METRICS);
    EXPECT_EQ(settings_policy::parse_enabled_metrics("").value, ENABLE_ALL_METRICS);
    EXPECT_EQ(settings_policy::parse_enabled_metrics("   ").value, ENABLE_ALL_METRICS);
    EXPECT_EQ(settings_policy::parse_enabled_metrics("none").value, DISABLE_ALL_METRICS);
}

}  // namespace rocprofsys::pmc::collectors::testing
