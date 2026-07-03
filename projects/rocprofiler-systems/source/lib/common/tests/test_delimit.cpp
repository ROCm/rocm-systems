// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/delimit.hpp"

#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace rocprofsys::common;

using strvec = std::vector<std::string>;

TEST(delimit_test, basic_split)
{
    EXPECT_EQ(delimit("a,b,c", ","), (strvec{ "a", "b", "c" }));
}

TEST(delimit_test, default_delimiters)
{
    EXPECT_EQ(delimit(R"(a"b'c,d;e:f g)"), (strvec{ "a", "b", "c", "d", "e", "f", "g" }));
}

TEST(delimit_test, empty_tokens_dropped)
{
    EXPECT_EQ(delimit("a,,b", ","), (strvec{ "a", "b" }));
}

TEST(delimit_test, leading_trailing_delimiters)
{
    EXPECT_EQ(delimit(",a,b,", ","), (strvec{ "a", "b" }));
}

TEST(delimit_test, multi_char_delimiter_set)
{
    EXPECT_EQ(delimit("a,b;c d", ",; "), (strvec{ "a", "b", "c", "d" }));
}

TEST(delimit_test, no_delimiter_present)
{
    EXPECT_EQ(delimit("abc", ","), (strvec{ "abc" }));
}

TEST(delimit_test, empty_input) { EXPECT_EQ(delimit("", ","), strvec{}); }
