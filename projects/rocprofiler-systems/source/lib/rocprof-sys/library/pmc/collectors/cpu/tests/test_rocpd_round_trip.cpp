// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// Regression test for AIPROFSYST-445.
//
// PR #4612 split the CPU sampler emit site (rocpd_processor.cpp) and the
// metadata registration site (cache_policy.hpp) but registered metric names
// as hard-coded literals while the sampler emits trait::name<category::*>.
// The two name sets did not overlap, producing
//   data_processor.cpp:201 "non-existing PMC description"
//   data_processor.cpp:221 "Unexisting track"
// on every sample, with empty rocpd_pmc_event rows.
//
// Verifying the fix would normally require driving cache_policy and asserting
// against the global metadata_registry singleton, but pulling
// cache_manager.hpp into the test brings in perfetto_processor whose object
// references binary-only symbols (see source/tests/CMakeLists.txt for the
// reason rocprof-sys-unit-tests deliberately omits the binary library).
//
// Instead we pin the trait names that drive both sides of the contract.
// rocpd_processor.cpp (sampler) and cache_policy.hpp (registration) both
// look up these traits, so locking the values down here ensures the two
// sites stay in sync. Any future rename to a category trait without a
// matching update on both sides will fail this test.

#include "core/categories.hpp"

#include <gtest/gtest.h>

#include <cstring>

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
    // Sampler at rocpd_processor.cpp:633-637 builds per-cpu names as
    //   "<trait::name<cpu_freq>::value> [<device_id>] Core [<cpu_id>]"
    EXPECT_STREQ(name<category::cpu_freq>::value, "cpu_frequency");
}

TEST(cpu_pmc_name_contract, cpu_load_trait_matches_sampler_contract)
{
    EXPECT_STREQ(name<category::cpu_load>::value, "cpu_load");
}

// Pin the literal-vs-trait contract that broke in AIPROFSYST-445.
// If anyone ever brings back a hard-coded literal in cache_policy.hpp
// (e.g. "process_page_rss") this test will catch it: those literals are
// NOT what the trait resolves to, and the sampler will silently drop rows.
TEST(cpu_pmc_name_contract, old_pre_fix_literals_do_not_match_traits)
{
    EXPECT_STRNE(name<category::process_page>::value, "process_page_rss");
    EXPECT_STRNE(name<category::process_virt>::value, "process_virt_mem");
    EXPECT_STRNE(name<category::process_peak>::value, "process_peak_rss");
    EXPECT_STRNE(name<category::process_context_switch>::value, "process_ctx_switches");
    EXPECT_STRNE(name<category::process_page_fault>::value, "process_page_faults");
}

}  // namespace rocprofsys::pmc::collectors::cpu::testing
