// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "track_read_options.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace
{

using profiler_hub::parse_size;
using profiler_hub::track_read_options;

constexpr size_t default_min_events = 1000000;
constexpr size_t default_parts      = 8;

track_read_options::env_lookup_t
lookup_of(std::map<std::string, std::string> values)
{
    return [values = std::move(values)](const char* name) -> const char* {
        const auto it = values.find(name);
        return it == values.end() ? nullptr : it->second.c_str();
    };
}

TEST(parse_size_test, valid_number_returns_value) { EXPECT_EQ(parse_size("42", 7), 42U); }

TEST(parse_size_test, null_returns_fallback) { EXPECT_EQ(parse_size(nullptr, 7), 7U); }

TEST(parse_size_test, empty_returns_fallback) { EXPECT_EQ(parse_size("", 7), 7U); }

TEST(parse_size_test, zero_returns_zero) { EXPECT_EQ(parse_size("0", 7), 0U); }

TEST(parse_size_test, non_numeric_returns_fallback)
{
    EXPECT_EQ(parse_size("abc", 7), 7U);
}

TEST(parse_size_test, trailing_characters_return_fallback)
{
    EXPECT_EQ(parse_size("12x", 7), 7U);
    EXPECT_EQ(parse_size("12 ", 7), 7U);
}

TEST(parse_size_test, leading_space_or_sign_returns_fallback)
{
    EXPECT_EQ(parse_size(" 12", 7), 7U);
    EXPECT_EQ(parse_size("+12", 7), 7U);
    EXPECT_EQ(parse_size("-1", 7), 7U);
}

TEST(parse_size_test, overflow_returns_fallback)
{
    EXPECT_EQ(parse_size("99999999999999999999999999", 7), 7U);
}

TEST(parse_size_test, largest_size_t_is_accepted)
{
    const auto largest = std::numeric_limits<size_t>::max();
    EXPECT_EQ(parse_size(std::to_string(largest).c_str(), 7), largest);
}

TEST(track_read_options_test, unset_variables_keep_defaults)
{
    const auto options = track_read_options::from_env(lookup_of({}));

    EXPECT_EQ(options.parallel_read_min_events, default_min_events);
    EXPECT_EQ(options.parallel_read_parts, default_parts);
}

TEST(track_read_options_test, valid_values_are_used)
{
    const auto options = track_read_options::from_env(
        lookup_of({ { "PH_READ_MIN_EVENTS", "500" }, { "PH_READ_PARTS", "4" } }));

    EXPECT_EQ(options.parallel_read_min_events, 500U);
    EXPECT_EQ(options.parallel_read_parts, 4U);
}

TEST(track_read_options_test, invalid_values_keep_defaults)
{
    const auto options = track_read_options::from_env(
        lookup_of({ { "PH_READ_MIN_EVENTS", "many" }, { "PH_READ_PARTS", "4 parts" } }));

    EXPECT_EQ(options.parallel_read_min_events, default_min_events);
    EXPECT_EQ(options.parallel_read_parts, default_parts);
}

TEST(track_read_options_test, zero_parts_become_one)
{
    const auto options =
        track_read_options::from_env(lookup_of({ { "PH_READ_PARTS", "0" } }));

    EXPECT_EQ(options.parallel_read_parts, 1U);
}

TEST(track_read_options_test, zero_min_events_is_accepted)
{
    const auto options =
        track_read_options::from_env(lookup_of({ { "PH_READ_MIN_EVENTS", "0" } }));

    EXPECT_EQ(options.parallel_read_min_events, 0U);
}

TEST(track_read_options_test, lookup_is_called_with_the_documented_names)
{
    std::vector<std::string> requested;

    std::ignore = track_read_options::from_env([&requested](const char* name) {
        requested.emplace_back(name);
        return static_cast<const char*>(nullptr);
    });

    EXPECT_THAT(requested,
                ::testing::UnorderedElementsAre("PH_READ_MIN_EVENTS", "PH_READ_PARTS"));
}

}  // namespace
