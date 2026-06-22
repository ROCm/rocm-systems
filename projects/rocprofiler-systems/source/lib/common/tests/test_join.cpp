// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/join.hpp"

#include <gtest/gtest.h>
#include <string>

using namespace rocprofsys::common;

TEST(join_test, scalar_char_delim) { EXPECT_EQ(join('/', "a", "b", "c"), "a/b/c"); }

TEST(join_test, scalar_string_delim) { EXPECT_EQ(join(", ", "a", "b"), "a, b"); }

TEST(join_test, single_arg) { EXPECT_EQ(join('/', "x"), "x"); }

TEST(join_test, no_value_args) { EXPECT_EQ(join(','), ""); }

TEST(join_test, numeric_args) { EXPECT_EQ(join('-', 1, 2, 3), "1-2-3"); }

TEST(join_test, bool_args_boolalpha)
{
    // join sets std::boolalpha, so bools render as true/false
    EXPECT_EQ(join(' ', true, false), "true false");
}

TEST(join_test, quote_strings)
{
    // QuoteStrings quotes string-like args only; non-strings are left as-is
    EXPECT_EQ(join(QuoteStrings{}, ' ', "a", 1), "\"a\" 1");
}
