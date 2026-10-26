// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/common/settings.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace rocprofsys::pmc::collectors::testing
{

TEST(settings_policy_test, parse_enabled_metrics_accepts_spaces_after_separators)
{
    constexpr std::uint32_t busy_temp_power_mem_usage = 0xFFU;

    const auto metrics =
        settings_policy::parse_enabled_metrics("busy, temp, power, mem_usage");

    EXPECT_EQ(metrics.value, busy_temp_power_mem_usage);
}

}  // namespace rocprofsys::pmc::collectors::testing
