// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/progress/bar.hpp"

#include <gtest/gtest.h>

#include <string>

namespace rocprofsys::progress::detail
{

TEST(format_bar_test, ascii_brackets_empty)
{
    EXPECT_EQ(format_bar(bar_style::ascii_brackets, 10, 0.0), "[..........]");
}

TEST(format_bar_test, ascii_brackets_half)
{
    EXPECT_EQ(format_bar(bar_style::ascii_brackets, 10, 0.5), "[#####.....]");
}

TEST(format_bar_test, ascii_brackets_full)
{
    EXPECT_EQ(format_bar(bar_style::ascii_brackets, 10, 1.0), "[##########]");
}

TEST(format_bar_test, ascii_brackets_clamps_above_one)
{
    // frac > 1.0 must clamp; otherwise size_t underflow corrupts the string.
    EXPECT_EQ(format_bar(bar_style::ascii_brackets, 10, 1.5), "[##########]");
}

TEST(format_bar_test, ascii_brackets_clamps_below_zero)
{
    EXPECT_EQ(format_bar(bar_style::ascii_brackets, 10, -0.5), "[..........]");
}

TEST(format_bar_test, ascii_arrow_empty)
{
    EXPECT_EQ(format_bar(bar_style::ascii_arrow, 10, 0.0), "[          ]");
}

TEST(format_bar_test, ascii_arrow_partial_has_arrowhead)
{
    EXPECT_EQ(format_bar(bar_style::ascii_arrow, 10, 0.5), "[====>     ]");
}

TEST(format_bar_test, ascii_arrow_full_no_arrowhead)
{
    EXPECT_EQ(format_bar(bar_style::ascii_arrow, 10, 1.0), "[==========]");
}

TEST(format_bar_test, unicode_blocks_full_uses_full_block)
{
    const std::string out = format_bar(bar_style::unicode_blocks, 4, 1.0);
    EXPECT_EQ(out, "[\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88]");
}

TEST(format_bar_test, unicode_blocks_empty_is_spaces)
{
    EXPECT_EQ(format_bar(bar_style::unicode_blocks, 4, 0.0), "[    ]");
}

TEST(format_bar_test, text_only_is_empty)
{
    EXPECT_EQ(format_bar(bar_style::text_only, 10, 0.5), "");
}

TEST(format_bar_test, unicode_dots_full_no_brackets)
{
    const std::string out = format_bar(bar_style::unicode_dots, 3, 1.0);
    EXPECT_EQ(out, "\xe2\x97\x8f\xe2\x97\x8f\xe2\x97\x8f");
}

}  // namespace rocprofsys::progress::detail
