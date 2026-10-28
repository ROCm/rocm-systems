// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/common/settings.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <set>
#include <string>

namespace rocprofsys::pmc::collectors::testing
{

// get_enabled_metrics() caches its first result per process; this must stay the only
// test in the binary that calls it.
TEST(settings_policy_test, get_enabled_metrics_accepts_spaces_after_separators)
{
    constexpr std::uint32_t k_busy_temp_power_mem_usage = 0xFFU;
    const std::string       name{ env_vars::AMD_SMI_METRICS };

    ASSERT_EQ(::setenv(name.c_str(), "busy, temp, power, mem_usage", 1), 0);
    tim::settings::instance()->insert<std::string, std::string>(
        name, "amd_smi_metrics", "", std::string{ "all" }, std::set<std::string>{});

    EXPECT_EQ(settings_policy::get_enabled_metrics().value, k_busy_temp_power_mem_usage);
}

}  // namespace rocprofsys::pmc::collectors::testing
