// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// Pins the trait::name values that the sampler (rocpd_processor.cpp) and
// the registration site (cache_policy.hpp) both look up. Renaming a category
// trait without updating both sides will fail this test.

#include "core/categories.hpp"

#include <gtest/gtest.h>

namespace rocprofsys::pmc::collectors::cpu::testing
{

using ::tim::trait::name;
namespace category = ::tim::category;

TEST(cpu_pmc_name_contract, process_page_trait_matches_sampler_contract)
{
    EXPECT_STREQ(name<category::process_page>::value, "process_physical_memory");
}

TEST(cpu_pmc_name_contract, process_virt_trait_matches_sampler_contract)
{
    EXPECT_STREQ(name<category::process_virt>::value, "process_virtual_memory");
}

TEST(cpu_pmc_name_contract, process_peak_trait_matches_sampler_contract)
{
    EXPECT_STREQ(name<category::process_peak>::value, "process_memory_hwm");
}

TEST(cpu_pmc_name_contract, process_context_switch_trait_matches_sampler_contract)
{
    EXPECT_STREQ(name<category::process_context_switch>::value, "process_context_switch");
}

TEST(cpu_pmc_name_contract, process_page_fault_trait_matches_sampler_contract)
{
    EXPECT_STREQ(name<category::process_page_fault>::value, "process_page_fault");
}

TEST(cpu_pmc_name_contract, process_user_mode_time_trait_matches_sampler_contract)
{
    EXPECT_STREQ(name<category::process_user_mode_time>::value, "process_user_cpu_time");
}

TEST(cpu_pmc_name_contract, process_kernel_mode_time_trait_matches_sampler_contract)
{
    EXPECT_STREQ(name<category::process_kernel_mode_time>::value,
                 "process_kernel_cpu_time");
}

TEST(cpu_pmc_name_contract, cpu_freq_trait_matches_sampler_contract)
{
    EXPECT_STREQ(name<category::cpu_freq>::value, "cpu_frequency");
}

TEST(cpu_pmc_name_contract, cpu_load_trait_matches_sampler_contract)
{
    EXPECT_STREQ(name<category::cpu_load>::value, "cpu_load");
}

TEST(cpu_pmc_name_contract, old_pre_fix_literals_do_not_match_traits)
{
    EXPECT_STRNE(name<category::process_page>::value, "process_page_rss");
    EXPECT_STRNE(name<category::process_virt>::value, "process_virt_mem");
    EXPECT_STRNE(name<category::process_peak>::value, "process_peak_rss");
    EXPECT_STRNE(name<category::process_context_switch>::value, "process_ctx_switches");
    EXPECT_STRNE(name<category::process_page_fault>::value, "process_page_faults");
}

}  // namespace rocprofsys::pmc::collectors::cpu::testing
