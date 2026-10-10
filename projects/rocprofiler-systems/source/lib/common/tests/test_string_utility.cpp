// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/string_utility.hpp"

#include <gtest/gtest.h>

#include <cstdint>

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

TEST(ltrim, basic_check) { EXPECT_EQ(ltrim("  hello"), "hello"); }

TEST(ltrim, empty_string_returns_empty_string) { EXPECT_EQ(ltrim(""), ""); }

TEST(ltrim, all_whitespace_returns_empty_string) { EXPECT_EQ(ltrim("   "), ""); }

TEST(ltrim, no_leading_whitespace_is_unchanged)
{
    EXPECT_EQ(ltrim("nowhitespace"), "nowhitespace");
}

TEST(ltrim, trailing_whitespace_is_unaffected) { EXPECT_EQ(ltrim("hello  "), "hello  "); }

TEST(ltrim, mixed_whitespace_characters)
{
    EXPECT_EQ(ltrim(" \t\n\r\f\vhello"), "hello");
}

TEST(rtrim, basic_check) { EXPECT_EQ(rtrim("hello  "), "hello"); }

TEST(rtrim, empty_string_returns_empty_string) { EXPECT_EQ(rtrim(""), ""); }

TEST(rtrim, all_whitespace_returns_empty_string) { EXPECT_EQ(rtrim("   "), ""); }

TEST(rtrim, no_trailing_whitespace_is_unchanged)
{
    EXPECT_EQ(rtrim("nowhitespace"), "nowhitespace");
}

TEST(rtrim, leading_whitespace_is_unaffected) { EXPECT_EQ(rtrim("  hello"), "  hello"); }

TEST(rtrim, mixed_whitespace_characters)
{
    EXPECT_EQ(rtrim("hello \t\n\r\f\v"), "hello");
}

TEST(trim, basic_check) { EXPECT_EQ(trim("  hello  "), "hello"); }

TEST(trim, empty_string_returns_empty_string) { EXPECT_EQ(trim(""), ""); }

TEST(trim, all_whitespace_returns_empty_string) { EXPECT_EQ(trim("   "), ""); }

TEST(trim, no_whitespace_is_unchanged)
{
    EXPECT_EQ(trim("nowhitespace"), "nowhitespace");
}

TEST(trim, leading_only_whitespace) { EXPECT_EQ(trim("   hello"), "hello"); }

TEST(trim, trailing_only_whitespace) { EXPECT_EQ(trim("hello   "), "hello"); }

TEST(trim, mixed_whitespace_characters)
{
    EXPECT_EQ(trim(" \t\n\r\f\vhello \t\n\r\f\v"), "hello");
}

TEST(trim, internal_whitespace_is_preserved)
{
    EXPECT_EQ(trim("  hello   world  "), "hello   world");
}

TEST(strip_rocprofsys_prefix, strips_prefix_case_insensitively)
{
    EXPECT_EQ(strip_rocprofsys_prefix("ROCPROFSYS_SAMPLING_FREQ"), "sampling_freq");
    EXPECT_EQ(strip_rocprofsys_prefix("rocprofsys_trace"), "trace");
}

TEST(strip_rocprofsys_prefix, no_prefix_is_only_lowercased)
{
    EXPECT_EQ(strip_rocprofsys_prefix("SAMPLING_FREQ"), "sampling_freq");
}

TEST(strip_ansi, no_escape_sequences_is_unchanged)
{
    EXPECT_EQ(strip_ansi("hello world"), "hello world");
}

TEST(strip_ansi, empty_string_returns_empty_string) { EXPECT_EQ(strip_ansi(""), ""); }

TEST(strip_ansi, strips_single_color_code)
{
    EXPECT_EQ(strip_ansi("\033[01;34minfo\033[0m"), "info");
}

TEST(strip_ansi, strips_multiple_color_codes)
{
    EXPECT_EQ(strip_ansi("\033[01;34m[a]\033[0m \033[01;31m[b]\033[0m"), "[a] [b]");
}

TEST(strip_ansi, code_followed_by_text_strips_only_the_code)
{
    EXPECT_EQ(strip_ansi("\033[01;34mafter"), "after");
}

TEST(strip_ansi, unterminated_escape_sequence_consumes_rest_of_string)
{
    // no 'm' anywhere after the escape starts, so everything through the end
    // of the string is treated as (and dropped with) the escape sequence
    EXPECT_EQ(strip_ansi("before\033[01;34no_end_here"), "before");
}

TEST(strip_ansi, string_with_only_escape_sequence_returns_empty_string)
{
    EXPECT_EQ(strip_ansi("\033[0m"), "");
}

TEST(hex_padded, basic_check)
{
    EXPECT_EQ(hex_padded(std::uintptr_t{ 0x10 }), "0x0000000000000010");
}

TEST(hex_padded, zero_value)
{
    EXPECT_EQ(hex_padded(std::uintptr_t{ 0 }), "0x0000000000000000");
}

TEST(hex_padded, value_filling_default_width_needs_no_padding)
{
    EXPECT_EQ(hex_padded(std::uintptr_t{ 0x123456789abcdef0 }), "0x123456789abcdef0");
}

TEST(hex_padded, custom_width_is_honored)
{
    EXPECT_EQ(hex_padded(std::uintptr_t{ 0xff }, 4), "0x00ff");
}

TEST(hex_padded, custom_width_narrower_than_value_is_not_truncated)
{
    EXPECT_EQ(hex_padded(std::uintptr_t{ 0x1234 }, 2), "0x1234");
}
