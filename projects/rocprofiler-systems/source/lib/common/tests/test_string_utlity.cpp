// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/string_utility.hpp"

#include <gtest/gtest.h>

using namespace rocprofsys::utility::string;

TEST(to_lower, basic_check) { EXPECT_EQ(to_lower("ABCD"), "abcd"); }

TEST(to_lower, empty_string_returns_empty_string) { EXPECT_EQ(to_lower(""), ""); }

TEST(to_lower, already_lowercase_is_unchanged)
{
    EXPECT_EQ(to_lower("already_lower"), "already_lower");
}

TEST(to_lower, mixed_case_lowercases_only_letters)
{
    EXPECT_EQ(to_lower("MiXeD123"), "mixed123");
}

TEST(to_lower, non_alpha_characters_are_unaffected)
{
    EXPECT_EQ(to_lower("123!@# \t_-"), "123!@# \t_-");
}

TEST(to_lower, single_character)
{
    EXPECT_EQ(to_lower("A"), "a");
    EXPECT_EQ(to_lower("a"), "a");
}

TEST(to_upper, basic_check) { EXPECT_EQ(to_upper("abcd"), "ABCD"); }

TEST(to_upper, empty_string_returns_empty_string) { EXPECT_EQ(to_upper(""), ""); }

TEST(to_upper, already_uppercase_is_unchanged)
{
    EXPECT_EQ(to_upper("ALREADY_UPPER"), "ALREADY_UPPER");
}

TEST(to_upper, mixed_case_uppercases_only_letters)
{
    EXPECT_EQ(to_upper("MiXeD123"), "MIXED123");
}

TEST(to_upper, non_alpha_characters_are_unaffected)
{
    EXPECT_EQ(to_upper("123!@# \t_-"), "123!@# \t_-");
}

TEST(to_upper, single_character)
{
    EXPECT_EQ(to_upper("a"), "A");
    EXPECT_EQ(to_upper("A"), "A");
}
