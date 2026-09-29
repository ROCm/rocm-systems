// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/ompt/rules.hpp"

#include <gtest/gtest.h>

#include <span>
#include <string_view>

namespace rocprofsys::domains::callback::ompt::rules
{
namespace
{

// decoder.hpp's flag_representation::groups() assumes each rule table is already
// partitioned into contiguous runs sharing {type, key, mode}: it walks the table
// once and closes a group as soon as it sees a rule that doesn't match. If a group
// were split into two non-adjacent runs, the second run would silently be decoded
// as its own (incomplete) group instead of being merged with the first.
template <typename FlagT>
void
expect_no_split_groups(std::span<const flag_rule<FlagT>> table)
{
    for(std::size_t i = 0; i < table.size(); ++i)
    {
        for(std::size_t j = i + 1; j < table.size(); ++j)
        {
            const bool same_group = table[i].type == table[j].type &&
                                    table[i].key == table[j].key &&
                                    table[i].mode == table[j].mode;
            if(!same_group)
            {
                continue;
            }

            for(std::size_t k = i + 1; k < j; ++k)
            {
                EXPECT_TRUE(table[k].type == table[i].type &&
                            table[k].key == table[i].key &&
                            table[k].mode == table[i].mode)
                    << "rule table interleaves group (" << table[i].type << ", "
                    << table[i].key << ") around index " << k;
            }
        }
    }
}

TEST(ompt_rules_test, k_parallel_rules_form_contiguous_groups)
{
    expect_no_split_groups<int>(k_parallel);
}

TEST(ompt_rules_test, k_task_create_rules_form_contiguous_groups)
{
    expect_no_split_groups<int>(k_task_create);
}

TEST(ompt_rules_test, k_implicit_task_rules_form_contiguous_groups)
{
    expect_no_split_groups<int>(k_implicit_task);
}

TEST(ompt_rules_test, k_cancel_rules_form_contiguous_groups)
{
    expect_no_split_groups<int>(k_cancel);
}

TEST(ompt_rules_test, k_parallel_has_expected_entry_count)
{
    EXPECT_EQ(k_parallel.size(), 4U);
}

TEST(ompt_rules_test, k_task_create_has_expected_entry_count)
{
    EXPECT_EQ(k_task_create.size(), 9U);
}

TEST(ompt_rules_test, k_implicit_task_has_expected_entry_count)
{
    EXPECT_EQ(k_implicit_task.size(), 2U);
}

TEST(ompt_rules_test, k_cancel_has_expected_entry_count)
{
    EXPECT_EQ(k_cancel.size(), 7U);
}

// The "properties" group in k_task_create is the only all_matches group with
// emit_none; a regression here would silently drop the "none" fallback value.
TEST(ompt_rules_test, k_task_create_properties_group_emits_none_on_no_match)
{
    for(const auto& rule : k_task_create)
    {
        if(rule.key != "properties")
        {
            continue;
        }

        EXPECT_EQ(rule.mode, flag_decode_mode::all_matches);
        EXPECT_EQ(rule.on_no_match, no_match_behavior::emit_none);
    }
}

}  // namespace
}  // namespace rocprofsys::domains::callback::ompt::rules
