// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/text_layout.hpp"

#include <string>

using rocprofsys::output::display_width;
using rocprofsys::output::summarize_command;

TEST(summarize_command, empty_returns_empty) { EXPECT_EQ(summarize_command(""), ""); }

TEST(summarize_command, bare_program_is_unchanged)
{
    EXPECT_EQ(summarize_command("python"), "python");
}

TEST(summarize_command, strips_path_to_basename)
{
    EXPECT_EQ(summarize_command("/usr/bin/python3"), "python3");
}

TEST(summarize_command, takes_first_token_only)
{
    EXPECT_EQ(summarize_command("python -c import_x --flag"), "python");
}

TEST(summarize_command, basename_of_first_token_with_args)
{
    EXPECT_EQ(summarize_command("/opt/rocm/bin/rocinfo --all"), "rocinfo");
}

TEST(summarize_command, strips_terminal_control_chars)
{
    EXPECT_EQ(summarize_command("\x1b[31mpython\x1b[0m"), "python");
}

TEST(display_width, ascii_equals_byte_count)
{
    EXPECT_EQ(display_width("hello"), 5u);
    EXPECT_EQ(display_width(""), 0u);
}

TEST(display_width, counts_each_multibyte_codepoint_once)
{
    // Two U+2500 box-drawing chars are 6 bytes but 2 display columns.
    EXPECT_EQ(display_width("──"), 2u);
}

TEST(display_width, mixed_ascii_and_multibyte) { EXPECT_EQ(display_width("a─b"), 3u); }
