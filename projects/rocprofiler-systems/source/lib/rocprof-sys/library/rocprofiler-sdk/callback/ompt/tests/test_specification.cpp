// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/ompt/specification.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>

namespace rocprofsys::domains::callback::ompt
{
namespace
{

// Regression guard: these values come from the OMPT specification and are relied
// upon by rules.hpp's static rule tables. A silent renumbering here would silently
// break flag decoding.
TEST(ompt_specification_test, parallel_flags_match_ompt_spec_values)
{
    EXPECT_EQ(
        static_cast<std::size_t>(ompt_parallel_flag_t::ompt_parallel_invoker_program),
        0x00000001U);
    EXPECT_EQ(
        static_cast<std::size_t>(ompt_parallel_flag_t::ompt_parallel_invoker_runtime),
        0x00000002U);
    EXPECT_EQ(static_cast<std::size_t>(ompt_parallel_flag_t::ompt_parallel_league),
              0x40000000U);
    EXPECT_EQ(static_cast<std::size_t>(ompt_parallel_flag_t::ompt_parallel_team),
              0x80000000U);
}

TEST(ompt_specification_test, cancel_flags_match_ompt_spec_values)
{
    EXPECT_EQ(static_cast<std::size_t>(ompt_cancel_flag_t::ompt_cancel_parallel), 0x01U);
    EXPECT_EQ(static_cast<std::size_t>(ompt_cancel_flag_t::ompt_cancel_sections), 0x02U);
    EXPECT_EQ(static_cast<std::size_t>(ompt_cancel_flag_t::ompt_cancel_loop), 0x04U);
    EXPECT_EQ(static_cast<std::size_t>(ompt_cancel_flag_t::ompt_cancel_taskgroup), 0x08U);
    EXPECT_EQ(static_cast<std::size_t>(ompt_cancel_flag_t::ompt_cancel_activated), 0x10U);
    EXPECT_EQ(static_cast<std::size_t>(ompt_cancel_flag_t::ompt_cancel_detected), 0x20U);
    EXPECT_EQ(static_cast<std::size_t>(ompt_cancel_flag_t::ompt_cancel_discarded_task),
              0x40U);
}

TEST(ompt_specification_test, thread_type_values_match_ompt_spec)
{
    EXPECT_EQ(static_cast<int>(ompt_thread_t::ompt_thread_initial), 1);
    EXPECT_EQ(static_cast<int>(ompt_thread_t::ompt_thread_worker), 2);
    EXPECT_EQ(static_cast<int>(ompt_thread_t::ompt_thread_other), 3);
    EXPECT_EQ(static_cast<int>(ompt_thread_t::ompt_thread_unknown), 4);
}

// has_flag/to_underlying are decoded against a bitmask, so every task flag must
// occupy a distinct bit -- a duplicate value would make two flags indistinguishable.
TEST(ompt_specification_test, task_flags_occupy_distinct_bits)
{
    constexpr std::array k_flags{
        ompt_task_flag_t::ompt_task_initial,   ompt_task_flag_t::ompt_task_implicit,
        ompt_task_flag_t::ompt_task_explicit,  ompt_task_flag_t::ompt_task_target,
        ompt_task_flag_t::ompt_task_taskwait,  ompt_task_flag_t::ompt_task_undeferred,
        ompt_task_flag_t::ompt_task_untied,    ompt_task_flag_t::ompt_task_final,
        ompt_task_flag_t::ompt_task_mergeable, ompt_task_flag_t::ompt_task_merged,
    };

    std::size_t seen_bits = 0;
    for(const auto flag : k_flags)
    {
        const auto bit = static_cast<std::size_t>(flag);
        EXPECT_EQ(seen_bits & bit, 0U) << "flag bit collides with an earlier flag";
        seen_bits |= bit;
    }
}

TEST(ompt_specification_test, cancel_flags_occupy_distinct_bits)
{
    constexpr std::array k_flags{
        ompt_cancel_flag_t::ompt_cancel_parallel,
        ompt_cancel_flag_t::ompt_cancel_sections,
        ompt_cancel_flag_t::ompt_cancel_loop,
        ompt_cancel_flag_t::ompt_cancel_taskgroup,
        ompt_cancel_flag_t::ompt_cancel_activated,
        ompt_cancel_flag_t::ompt_cancel_detected,
        ompt_cancel_flag_t::ompt_cancel_discarded_task,
    };

    std::size_t seen_bits = 0;
    for(const auto flag : k_flags)
    {
        const auto bit = static_cast<std::size_t>(flag);
        EXPECT_EQ(seen_bits & bit, 0U) << "flag bit collides with an earlier flag";
        seen_bits |= bit;
    }
}

}  // namespace
}  // namespace rocprofsys::domains::callback::ompt
