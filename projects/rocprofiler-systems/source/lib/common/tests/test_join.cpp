// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/join.hpp"

#include <gtest/gtest.h>
#include <string>
#include <string_view>

using namespace rocprofsys::common;

TEST(join_test, basic_space_separated) { EXPECT_EQ(join(" ", "a", "b", "c"), "a b c"); }

TEST(join_test, custom_delimiter)
{
    EXPECT_EQ(join(", ", "a", "b", "c"), "a, b, c");
    EXPECT_EQ(join("/", "a", "b"), "a/b");
}

TEST(join_test, single_arg_has_no_delimiter) { EXPECT_EQ(join(", ", "only"), "only"); }

TEST(join_test, no_args_is_empty) { EXPECT_EQ(join(", "), ""); }

TEST(join_test, mixed_types)
{
    EXPECT_EQ(join(" ", "count:", 42, "ratio:", 1.5), "count: 42 ratio: 1.5");
}

TEST(join_test, bool_renders_boolalpha)
{
    EXPECT_EQ(join(" ", true, false), "true false");
}

TEST(join_test, std_string_argument)
{
    EXPECT_EQ(join("-", std::string{ "x" }, std::string{ "y" }), "x-y");
}

TEST(join_test, unquoted_does_not_add_quotes) { EXPECT_EQ(join(", ", "a", "b"), "a, b"); }

TEST(join_test, empty_delimiter_concatenates)
{
    EXPECT_EQ(join("", "a", "b", "c"), "abc");
}

TEST(join_test, quoted_wraps_string_args)
{
    EXPECT_EQ(join_with_strings_quoted(", ", "a", 1, false), R"("a", 1, false)");
}

TEST(join_test, quoted_handles_null_char_ptr)
{
    const char* null_str = nullptr;
    EXPECT_EQ(join_with_strings_quoted(", ", null_str), R"("")");
}

TEST(join_test, quoted_quotes_std_string)
{
    EXPECT_EQ(join_with_strings_quoted(" ", std::string{ "hi" }), R"("hi")");
}

TEST(join_test, quoted_quotes_string_view)
{
    EXPECT_EQ(join_with_strings_quoted(" ", std::string_view{ "hi" }), R"("hi")");
}

TEST(join_test, quoted_bool_and_number_unquoted)
{
    EXPECT_EQ(join_with_strings_quoted(" ", true, 7), "true 7");
}
