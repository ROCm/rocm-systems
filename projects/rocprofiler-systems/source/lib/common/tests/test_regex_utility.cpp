// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/regex_utility.hpp"

#include <gtest/gtest.h>

#include <regex>
#include <vector>

using rocprofsys::utility::regex_match_any;

TEST(regex_match_any, empty_regex_list_never_matches)
{
    EXPECT_FALSE(regex_match_any("anything", std::vector<std::regex>{}));
}

TEST(regex_match_any, matches_when_any_regex_matches)
{
    auto const regexes =
        std::vector<std::regex>{ std::regex{ "^foo" }, std::regex{ "bar$" } };
    EXPECT_TRUE(regex_match_any("foobaz", regexes));
    EXPECT_TRUE(regex_match_any("bazbar", regexes));
}

TEST(regex_match_any, no_match_when_none_match)
{
    auto const regexes =
        std::vector<std::regex>{ std::regex{ "^foo" }, std::regex{ "bar$" } };
    EXPECT_FALSE(regex_match_any("bazqux", regexes));
}

TEST(regex_match_any, uses_search_semantics_not_full_match)
{
    auto const regexes = std::vector<std::regex>{ std::regex{ "mid" } };
    EXPECT_TRUE(regex_match_any("start_mid_end", regexes));
}
