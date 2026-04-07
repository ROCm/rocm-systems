// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "library/pmc/device_providers/procfs/drivers/driver.hpp"

#include <gtest/gtest.h>

#include <string_view>

using namespace rocprofsys::pmc::drivers::procfs;

class parse_proc_stat_test : public ::testing::Test
{};

TEST_F(parse_proc_stat_test, parses_standard_four_cpu_content)
{
    constexpr auto content = "cpu  1000 50 300 9000 100 20 30 0 0 0\n"
                             "cpu0 200 10 150 9500 50 30 60 0 0 0\n"
                             "cpu1 300 20 50 9200 20 10 5 0 0 0\n"
                             "cpu2 250 10 50 9300 15 5 3 0 0 0\n"
                             "cpu3 250 10 50 9000 15 5 2 0 0 0\n"
                             "intr 12345 0 0 0\n"
                             "ctxt 67890\n"
                             "btime 1234567890\n";

    const auto result = parse_proc_stat(content);

    ASSERT_EQ(result.size(), 4u);
    EXPECT_EQ(result.at(0).user, 200u);
    EXPECT_EQ(result.at(0).idle, 9500u);
    EXPECT_EQ(result.at(1).user, 300u);
    EXPECT_EQ(result.at(2).system, 50u);
    EXPECT_EQ(result.at(3).softirq, 2u);
}

TEST_F(parse_proc_stat_test, skips_aggregate_cpu_line)
{
    constexpr auto content = "cpu  1000 50 300 9000 100 20 30 0 0 0\n"
                             "cpu0 200 10 150 9500 50 30 60 0 0 0\n";

    const auto result = parse_proc_stat(content);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.count(0), 1u);
}

TEST_F(parse_proc_stat_test, empty_content_returns_empty_map)
{
    const auto result = parse_proc_stat("");
    EXPECT_TRUE(result.empty());
}

TEST_F(parse_proc_stat_test, single_cpu_line)
{
    constexpr auto content = "cpu0 100 20 30 5000 10 5 3 0 0 0\n";

    const auto result = parse_proc_stat(content);

    ASSERT_EQ(result.size(), 1u);
    const auto& j = result.at(0);
    EXPECT_EQ(j.user, 100u);
    EXPECT_EQ(j.nice, 20u);
    EXPECT_EQ(j.system, 30u);
    EXPECT_EQ(j.idle, 5000u);
    EXPECT_EQ(j.iowait, 10u);
    EXPECT_EQ(j.irq, 5u);
    EXPECT_EQ(j.softirq, 3u);
}

TEST_F(parse_proc_stat_test, no_trailing_newline)
{
    constexpr auto content = "cpu0 100 20 30 5000 10 5 3 0 0 0";

    const auto result = parse_proc_stat(content);
    ASSERT_EQ(result.size(), 1u);
}

TEST_F(parse_proc_stat_test, malformed_line_skipped)
{
    constexpr auto content = "cpu0 200 10 150 9500 50 30 60 0 0 0\n"
                             "cpuX bad data here\n"
                             "cpu1 300 20 50 9200 20 10 5 0 0 0\n";

    const auto result = parse_proc_stat(content);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result.count(0), 1u);
    EXPECT_EQ(result.count(1), 1u);
}

TEST_F(parse_proc_stat_test, non_cpu_lines_ignored)
{
    constexpr auto content = "intr 12345 0 0 0\n"
                             "ctxt 67890\n"
                             "btime 1234567890\n"
                             "processes 999\n";

    const auto result = parse_proc_stat(content);
    EXPECT_TRUE(result.empty());
}

TEST_F(parse_proc_stat_test, total_and_active_computed_correctly)
{
    constexpr auto content = "cpu0 100 10 50 5000 20 5 3 0 0 0\n";

    const auto result = parse_proc_stat(content);
    ASSERT_EQ(result.size(), 1u);

    const auto& j = result.at(0);
    EXPECT_EQ(j.total(), 100u + 10u + 50u + 5000u + 20u + 5u + 3u);
    EXPECT_EQ(j.active(), 100u + 10u + 50u + 5u + 3u);
}

TEST_F(parse_proc_stat_test, high_cpu_id)
{
    constexpr auto content = "cpu255 100 20 30 5000 10 5 3 0 0 0\n";

    const auto result = parse_proc_stat(content);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.count(255), 1u);
}

class parse_statm_test : public ::testing::Test
{};

TEST_F(parse_statm_test, parses_normal_content)
{
    constexpr auto content = "12345 6789 1000 500 0 2000 0\n";

    const auto result = parse_statm(content);

    ASSERT_TRUE(result.has_value());
    const long page_size = sysconf(_SC_PAGESIZE);
    EXPECT_EQ(result->virt_mem, 12345L * page_size);
    EXPECT_EQ(result->page_rss, 6789L * page_size);
}

TEST_F(parse_statm_test, empty_content_returns_nullopt)
{
    const auto result = parse_statm("");
    EXPECT_FALSE(result.has_value());
}

TEST_F(parse_statm_test, single_number_returns_nullopt)
{
    const auto result = parse_statm("12345");
    EXPECT_FALSE(result.has_value());
}

TEST_F(parse_statm_test, leading_whitespace_handled)
{
    constexpr auto content = "  \t 100 200 0 0 0 0 0\n";

    const auto result = parse_statm(content);

    ASSERT_TRUE(result.has_value());
    const long page_size = sysconf(_SC_PAGESIZE);
    EXPECT_EQ(result->virt_mem, 100L * page_size);
    EXPECT_EQ(result->page_rss, 200L * page_size);
}

TEST_F(parse_statm_test, non_numeric_returns_nullopt)
{
    const auto result = parse_statm("abc def");
    EXPECT_FALSE(result.has_value());
}

class parsing_utilities_test : public ::testing::Test
{};

TEST_F(parsing_utilities_test, starts_with_match)
{
    EXPECT_TRUE(starts_with("cpu0 200", "cpu"));
    EXPECT_TRUE(starts_with("processor", "processor"));
    EXPECT_TRUE(starts_with("abc", ""));
}

TEST_F(parsing_utilities_test, starts_with_no_match)
{
    EXPECT_FALSE(starts_with("cpu0", "gpu"));
    EXPECT_FALSE(starts_with("cp", "cpu"));
    EXPECT_FALSE(starts_with("", "cpu"));
}

TEST_F(parsing_utilities_test, ltrim_removes_leading_whitespace)
{
    EXPECT_EQ(ltrim("  hello"), "hello");
    EXPECT_EQ(ltrim("\t  world"), "world");
    EXPECT_EQ(ltrim("nowhitespace"), "nowhitespace");
}

TEST_F(parsing_utilities_test, ltrim_empty_and_all_whitespace)
{
    EXPECT_EQ(ltrim(""), "");
    EXPECT_EQ(ltrim("   "), "");
}

TEST_F(parsing_utilities_test, split_lines_basic)
{
    const auto lines = split_lines("line1\nline2\nline3\n");

    ASSERT_EQ(lines.size(), 3u);
    EXPECT_EQ(lines[0], "line1");
    EXPECT_EQ(lines[1], "line2");
    EXPECT_EQ(lines[2], "line3");
}

TEST_F(parsing_utilities_test, split_lines_no_trailing_newline)
{
    const auto lines = split_lines("line1\nline2");

    ASSERT_EQ(lines.size(), 2u);
    EXPECT_EQ(lines[0], "line1");
    EXPECT_EQ(lines[1], "line2");
}

TEST_F(parsing_utilities_test, split_lines_empty)
{
    const auto lines = split_lines("");
    EXPECT_TRUE(lines.empty());
}

TEST_F(parsing_utilities_test, split_lines_single_line)
{
    const auto lines = split_lines("only one");

    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0], "only one");
}
