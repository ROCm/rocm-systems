// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/utility.hpp"

#include <cstdint>
#include <set>
#include <unordered_set>
#include <vector>

namespace
{
using rocprofsys::utility::parse_numeric_range;

// Convenience wrappers pinning the element type to the int64 instantiations that
// are explicitly instantiated in utility.cpp.
std::set<std::int64_t>
parse_set(const std::string& input)
{
    return parse_numeric_range<std::int64_t, std::set<std::int64_t>>(input, "ranks", 1L);
}

std::vector<std::int64_t>
parse_vec(const std::string& input)
{
    return parse_numeric_range<std::int64_t, std::vector<std::int64_t>>(input, "ranks",
                                                                        1L);
}
}  // namespace

TEST(parse_numeric_range, single_value)
{
    EXPECT_EQ(parse_set("2"), (std::set<std::int64_t>{ 2 }));
}

TEST(parse_numeric_range, comma_separated_values)
{
    EXPECT_EQ(parse_set("0,2,4"), (std::set<std::int64_t>{ 0, 2, 4 }));
}

TEST(parse_numeric_range, simple_range_inclusive)
{
    EXPECT_EQ(parse_set("0-4"), (std::set<std::int64_t>{ 0, 1, 2, 3, 4 }));
}

TEST(parse_numeric_range, range_with_inline_increment)
{
    EXPECT_EQ(parse_set("20-40:10"), (std::set<std::int64_t>{ 20, 30, 40 }));
}

TEST(parse_numeric_range, increment_stops_before_exceeding_end)
{
    // When the stride does not divide the range evenly, the last value is the
    // greatest one <= end: 20, 30, 40; the next step (50) exceeds 45 and stops,
    // so 45 itself is never emitted.
    EXPECT_EQ(parse_set("20-45:10"), (std::set<std::int64_t>{ 20, 30, 40 }));
}

TEST(parse_numeric_range, mixed_values_and_ranges)
{
    EXPECT_EQ(parse_set("0-3,8,10-12"),
              (std::set<std::int64_t>{ 0, 1, 2, 3, 8, 10, 11, 12 }));
}

TEST(parse_numeric_range, whitespace_and_semicolon_delimiters)
{
    EXPECT_EQ(parse_set("0; 2\t4"), (std::set<std::int64_t>{ 0, 2, 4 }));
}

TEST(parse_numeric_range, empty_string_yields_empty)
{
    EXPECT_TRUE(parse_set("").empty());
}

TEST(parse_numeric_range, vector_preserves_insertion_order)
{
    EXPECT_EQ(parse_vec("3,1,2"), (std::vector<std::int64_t>{ 3, 1, 2 }));
}

// --- invalid input is ignored (warned + skipped), not thrown ---

TEST(parse_numeric_range, non_numeric_token_ignored)
{
    EXPECT_TRUE(parse_set("garbage").empty());
}

TEST(parse_numeric_range, reversed_range_ignored)
{
    EXPECT_TRUE(parse_set("10-0").empty());
}

TEST(parse_numeric_range, leading_dash_negative_ignored)
{
    EXPECT_TRUE(parse_set("-1").empty());
}

TEST(parse_numeric_range, trailing_dash_ignored) { EXPECT_TRUE(parse_set("5-").empty()); }

TEST(parse_numeric_range, double_dash_ignored) { EXPECT_TRUE(parse_set("5--7").empty()); }

TEST(parse_numeric_range, too_many_range_parts_ignored)
{
    EXPECT_TRUE(parse_set("1-2-3").empty());
}

TEST(parse_numeric_range, invalid_tokens_skipped_valid_kept)
{
    // 'garbage' non-numeric, '10-0' reversed, '-1' negative all dropped;
    // '2' and '4-5' survive.
    EXPECT_EQ(parse_set("garbage,10-0,-1,2,4-5"), (std::set<std::int64_t>{ 2, 4, 5 }));
}

TEST(parse_numeric_range, single_element_range)
{
    EXPECT_EQ(parse_set("7-7"), (std::set<std::int64_t>{ 7 }));
}
