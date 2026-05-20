// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Pure-data invariants on the descriptor tables exposed by core_flags.hpp.
// These tests need no parser_t — they walk the static tables and assert
// shape conditions that the validate_descriptor / interpreter logic
// silently relies on.

#include "core/argparse/core_flags.hpp"
#include "core/argparse/flag_descriptor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace
{
using rocprofsys::argparse::flag_descriptor;
using rocprofsys::argparse::flag_group;
using rocprofsys::argparse::join_with;
using rocprofsys::argparse::value_kind;
using rocprofsys::common::update_mode;

const std::vector<const flag_group*>&
all_core_groups()
{
    namespace argparse = rocprofsys::argparse;
    static const std::vector<const flag_group*> groups = {
        &argparse::debug_group(),     &argparse::general_group(),
        &argparse::launcher_group(),  &argparse::tracing_group(),
        &argparse::profile_group(),   &argparse::process_sampling_group(),
        &argparse::general_sampling_group(), &argparse::sampling_timer_group(),
        &argparse::hw_counter_group(), &argparse::misc_group(),
    };
    return groups;
}

[[nodiscard]] std::string_view
last_name_or_unnamed(const flag_descriptor& descriptor)
{
    return descriptor.names.empty() ? std::string_view{ "<unnamed>" }
                                    : descriptor.names.back();
}

}  // namespace

TEST(CoreFlagsTable, EveryDescriptorHasNonEmptyNames)
{
    for(const auto* group : all_core_groups())
    {
        for(const auto& descriptor : group->flags)
        {
            EXPECT_FALSE(descriptor.names.empty())
                << "descriptor in group '" << group->title << "' has no names";
        }
    }
}

TEST(CoreFlagsTable, LongNameIsLastAndDoubleDashed)
{
    // The interpreter derives the parser_key from names.back() via strip_dashes;
    // by convention the LONG name (--foo) must be last. Short names (-x) may
    // precede it.
    for(const auto* group : all_core_groups())
    {
        for(const auto& descriptor : group->flags)
        {
            if(descriptor.names.empty()) continue;
            const auto& last = descriptor.names.back();
            EXPECT_GE(last.size(), 3u) << "last name '" << last << "' too short";
            EXPECT_EQ(last.substr(0, 2), "--")
                << "last name '" << last << "' missing -- prefix";
        }
    }
}

TEST(CoreFlagsTable, ListKindOrAppendModeDeclaresJoinExceptForCustom)
{
    // Same invariant validate_descriptor enforces at register time. Walking
    // the static tables here catches the bug at test time, with file:line
    // pointing at the descriptor.
    for(const auto* group : all_core_groups())
    {
        for(const auto& descriptor : group->flags)
        {
            if(descriptor.custom != nullptr) continue;  // custom handles its own emission

            const bool needs_delim = descriptor.kind == value_kind::list ||
                                     descriptor.mode == update_mode::APPEND ||
                                     descriptor.mode == update_mode::PREPEND;
            if(needs_delim)
            {
                EXPECT_NE(descriptor.join, join_with::none)
                    << "descriptor '" << last_name_or_unnamed(descriptor)
                    << "' in group '" << group->title
                    << "' needs an explicit join_with";
            }
        }
    }
}

TEST(CoreFlagsTable, ParserKeysAreUniqueAcrossAllGroups)
{
    // parser_key derives from strip_dashes(names.back()). Duplicate keys
    // across the descriptor tables would silently shadow each other in the
    // parser.
    std::unordered_set<std::string> keys;
    for(const auto* group : all_core_groups())
    {
        for(const auto& descriptor : group->flags)
        {
            if(descriptor.names.empty()) continue;
            auto       name = std::string{ descriptor.names.back() };
            const auto first_non_dash = name.find_first_not_of('-');
            if(first_non_dash != std::string::npos) name = name.substr(first_non_dash);

            const bool inserted = keys.insert(name).second;
            EXPECT_TRUE(inserted) << "duplicate parser key '" << name << "' in group '"
                                  << group->title << "'";
        }
    }
}

TEST(CoreFlagsTable, DedupKeysDoNotCollideWithDerivedEnvKey)
{
    // emit_env writes to descriptor.env_vars; remember_processed writes the
    // descriptor's own env_key plus all dedup_keys into processed_environs.
    // A descriptor that names its own derived env_key in dedup_keys is
    // redundant; flag it so the table stays clean.
    for(const auto* group : all_core_groups())
    {
        for(const auto& descriptor : group->flags)
        {
            if(descriptor.names.empty()) continue;
            auto       parser_key = std::string{ descriptor.names.back() };
            const auto first_non_dash = parser_key.find_first_not_of('-');
            if(first_non_dash != std::string::npos)
                parser_key = parser_key.substr(first_non_dash);

            auto env_key = parser_key;
            std::replace(env_key.begin(), env_key.end(), '-', '_');

            for(const auto& alias : descriptor.dedup_keys)
            {
                EXPECT_NE(std::string{ alias }, env_key)
                    << "descriptor '" << last_name_or_unnamed(descriptor)
                    << "' lists its own derived env_key '" << env_key
                    << "' in dedup_keys (redundant)";
            }
        }
    }
}
