// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/environment.hpp"

#include <gtest/gtest.h>

using namespace rocprofsys::common;

TEST(to_bool_test, numeric_nonzero_true)
{
    EXPECT_TRUE(to_bool("1"));
    EXPECT_TRUE(to_bool("42"));
}

TEST(to_bool_test, numeric_zero_false) { EXPECT_FALSE(to_bool("0")); }

TEST(to_bool_test, double_zero_false) { EXPECT_FALSE(to_bool("00")); }

TEST(to_bool_test, overflowing_digits_true)
{
    EXPECT_TRUE(to_bool("9999999999999999999999999"));
}

TEST(to_bool_test, false_tokens)
{
    for(const auto* false_token : { "off", "false", "no", "n", "f" })
    {
        EXPECT_FALSE(to_bool(false_token)) << "value: " << false_token;
    }
}

TEST(to_bool_test, true_tokens)
{
    for(const auto* true_token : { "on", "true", "yes", "y", "t", "garbage" })
        EXPECT_TRUE(to_bool(true_token)) << "value: " << true_token;
}

TEST(to_bool_test, case_insensitive)
{
    EXPECT_TRUE(to_bool("TRUE"));
    EXPECT_FALSE(to_bool("Off"));
    EXPECT_FALSE(to_bool("No"));
}

TEST(to_bool_test, empty_returns_fallback)
{
    EXPECT_FALSE(to_bool("", false));
    EXPECT_TRUE(to_bool("", true));
}

TEST(to_bool_test, default_fallback_is_false) { EXPECT_FALSE(to_bool("")); }

TEST(to_bool_test, whitespace_is_trimmed)
{
    EXPECT_TRUE(to_bool(" 1"));
    EXPECT_TRUE(to_bool("1 "));
    EXPECT_TRUE(to_bool("\t1\n"));
    EXPECT_FALSE(to_bool(" off"));
    EXPECT_FALSE(to_bool("off "));
    EXPECT_FALSE(to_bool(" 0"));
    EXPECT_FALSE(to_bool("0 "));
}

TEST(to_bool_test, whitespace_only_returns_fallback)
{
    EXPECT_FALSE(to_bool("   ", false));
    EXPECT_TRUE(to_bool("   ", true));
    EXPECT_FALSE(to_bool("\t\n "));
}

TEST(to_bool_test, mixed_alphanumeric_is_truthy)
{
    EXPECT_TRUE(to_bool("1a"));
    EXPECT_TRUE(to_bool("a1"));
}
