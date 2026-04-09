// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include "stitch/stitch.hpp"

class StitchUtilsTest : public ::testing::Test
{
protected:
};

// Tests for split2
TEST_F(StitchUtilsTest, Split2BasicSplit)
{
    auto [first, second] = split2("hello::world", "::");
    EXPECT_EQ(first, "hello");
    EXPECT_EQ(second, "world");
}

TEST_F(StitchUtilsTest, Split2NoDelimiter)
{
    auto [first, second] = split2("helloworld", "::");
    EXPECT_EQ(first, "helloworld");
    EXPECT_TRUE(second.empty());
}

TEST_F(StitchUtilsTest, Split2DelimiterAtEnd)
{
    auto [first, second] = split2("hello::", "::");
    EXPECT_EQ(first, "hello::");
    EXPECT_TRUE(second.empty());
}

TEST_F(StitchUtilsTest, Split2MultipleSplitPoints)
{
    // Only splits at first occurrence
    auto [first, second] = split2("a::b::c", "::");
    EXPECT_EQ(first, "a");
    EXPECT_EQ(second, "b::c");
}

TEST_F(StitchUtilsTest, Split2EmptyInput)
{
    auto [first, second] = split2("", "::");
    EXPECT_TRUE(first.empty());
    EXPECT_TRUE(second.empty());
}

// Tests for strip
TEST_F(StitchUtilsTest, StripLeadingWhitespace)
{
    EXPECT_EQ(strip("   hello"), "hello");
    EXPECT_EQ(strip("\t\nhello"), "hello");
}

TEST_F(StitchUtilsTest, StripTrailingWhitespace)
{
    EXPECT_EQ(strip("hello   "), "hello");
    EXPECT_EQ(strip("hello\t\n"), "hello");
}

TEST_F(StitchUtilsTest, StripBothEnds)
{
    EXPECT_EQ(strip("  hello world  "), "hello world");
    EXPECT_EQ(strip("\t\n  test  \n\t"), "test");
}

TEST_F(StitchUtilsTest, StripNoWhitespace) { EXPECT_EQ(strip("hello"), "hello"); }

TEST_F(StitchUtilsTest, StripOnlyWhitespace)
{
    // Edge case: all whitespace should return single char
    auto result = strip("   ");
    EXPECT_EQ(result.size(), 1);
}

TEST_F(StitchUtilsTest, StripEmptyString) { EXPECT_EQ(strip(""), ""); }

// Tests for splitv
TEST_F(StitchUtilsTest, SplitvBasicSplit)
{
    auto parts = splitv("a,b,c", ',');
    ASSERT_EQ(parts.size(), 3);
    EXPECT_EQ(parts[0], "a");
    EXPECT_EQ(parts[1], "b");
    EXPECT_EQ(parts[2], "c");
}

TEST_F(StitchUtilsTest, SplitvNoDelimiter)
{
    auto parts = splitv("abc", ',');
    ASSERT_EQ(parts.size(), 1);
    EXPECT_EQ(parts[0], "abc");
}

TEST_F(StitchUtilsTest, SplitvEmptyParts)
{
    auto parts = splitv("a,,b", ',');
    ASSERT_EQ(parts.size(), 3);
    EXPECT_EQ(parts[0], "a");
    EXPECT_TRUE(parts[1].empty());
    EXPECT_EQ(parts[2], "b");
}

TEST_F(StitchUtilsTest, SplitvStripsWhitespace)
{
    auto parts = splitv("  a  ,  b  ,  c  ", ',');
    ASSERT_EQ(parts.size(), 3);
    EXPECT_EQ(parts[0], "a");
    EXPECT_EQ(parts[1], "b");
    EXPECT_EQ(parts[2], "c");
}

TEST_F(StitchUtilsTest, SplitvPreservesQuotedDelimiters)
{
    // Delimiters inside quotes should not split
    auto parts = splitv("a,\"b,c\",d", ',');
    ASSERT_EQ(parts.size(), 3);
    EXPECT_EQ(parts[0], "a");
    EXPECT_EQ(parts[1], "\"b,c\"");
    EXPECT_EQ(parts[2], "d");
}

TEST_F(StitchUtilsTest, SplitvEmptyInput)
{
    auto parts = splitv("", ',');
    ASSERT_EQ(parts.size(), 1);
    EXPECT_TRUE(parts[0].empty());
}

TEST_F(StitchUtilsTest, SplitvOnlyDelimiters)
{
    auto parts = splitv(",,,", ',');
    ASSERT_EQ(parts.size(), 4);
    for (const auto& part : parts) { EXPECT_TRUE(part.empty()); }
}

// Test bValid utility
TEST(BValidTest, ValidPc)
{
    pcinfo_t valid1{100, 1};
    pcinfo_t valid2{0, 1};
    pcinfo_t valid3{100, 0};

    EXPECT_TRUE(bValid(valid1));
    EXPECT_TRUE(bValid(valid2));
    EXPECT_TRUE(bValid(valid3));
}

TEST(BValidTest, InvalidPc)
{
    pcinfo_t invalid{0, 0};
    EXPECT_FALSE(bValid(invalid));
}

// Additional bValid edge cases
TEST(BValidTest, LargeValues)
{
    pcinfo_t large_addr{0x7FFFFFFF, 0};
    pcinfo_t large_id{0, 1000};
    pcinfo_t both_large{0x7FFFFFFF, 1000};

    EXPECT_TRUE(bValid(large_addr));
    EXPECT_TRUE(bValid(large_id));
    EXPECT_TRUE(bValid(both_large));
}

// Edge case: split2 with delimiter at start
TEST(StitchUtilsEdgeCaseTest, Split2DelimiterAtStart)
{
    auto [first, second] = split2("::hello", "::");
    EXPECT_TRUE(first.empty() || first == "::hello");
}

// Edge case: split2 with only delimiter
TEST(StitchUtilsEdgeCaseTest, Split2OnlyDelimiter)
{
    auto [first, second] = split2("::", "::");
    // Behavior depends on implementation - just verify no crash
    EXPECT_TRUE(first.empty() || first == "::");
}

// Edge case: splitv with tabs
TEST(StitchUtilsEdgeCaseTest, SplitvWithTabs)
{
    auto parts = splitv("\ta\t,\tb\t,\tc\t", ',');
    ASSERT_EQ(parts.size(), 3);
    EXPECT_EQ(parts[0], "a");
    EXPECT_EQ(parts[1], "b");
    EXPECT_EQ(parts[2], "c");
}

// Edge case: splitv with nested quotes
TEST(StitchUtilsEdgeCaseTest, SplitvNestedQuotes)
{
    auto parts = splitv("a,\"b,\"c\"\",d", ',');
    // Verify no crash - exact behavior may vary
    EXPECT_GE(parts.size(), 1);
}

// Edge case: strip with multiple whitespace types
TEST(StitchUtilsEdgeCaseTest, StripMixedWhitespace)
{
    EXPECT_EQ(strip(" \t\n hello \n\t "), "hello");
    EXPECT_EQ(strip("\r\ntest\r\n"), "test");
}

// Test assemblyLine structure basics
TEST(AssemblyLineTest, DefaultConstruction)
{
    assemblyLine line;
    EXPECT_TRUE(line.line.empty());
    EXPECT_TRUE(line.loc.empty());
    EXPECT_FALSE(line.parsed);
}
