// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/natural_merge_sort.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace
{

using profiler_hub::common::natural_merge_sort;

std::vector<int>
sorted_copy(std::vector<int> values)
{
    natural_merge_sort(values.begin(), values.end(), std::less<>{});
    return values;
}

TEST(natural_merge_sort_test, empty_range_is_left_alone)
{
    EXPECT_TRUE(sorted_copy({}).empty());
}

TEST(natural_merge_sort_test, single_element_is_left_alone)
{
    EXPECT_THAT(sorted_copy({ 7 }), ::testing::ElementsAre(7));
}

TEST(natural_merge_sort_test, sorted_input_stays_sorted)
{
    EXPECT_THAT(sorted_copy({ 1, 2, 3, 4, 5 }), ::testing::ElementsAre(1, 2, 3, 4, 5));
}

TEST(natural_merge_sort_test, reversed_input_is_sorted)
{
    EXPECT_THAT(sorted_copy({ 5, 4, 3, 2, 1 }), ::testing::ElementsAre(1, 2, 3, 4, 5));
}

TEST(natural_merge_sort_test, several_sorted_runs_are_merged)
{
    EXPECT_THAT(sorted_copy({ 4, 5, 6, 1, 2, 3, 7, 8, 0 }),
                ::testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8));
}

TEST(natural_merge_sort_test, an_odd_number_of_runs_is_merged)
{
    EXPECT_THAT(sorted_copy({ 3, 4, 1, 2, 0 }), ::testing::ElementsAre(0, 1, 2, 3, 4));
}

TEST(natural_merge_sort_test, duplicates_are_kept)
{
    EXPECT_THAT(sorted_copy({ 2, 1, 2, 1, 2 }), ::testing::ElementsAre(1, 1, 2, 2, 2));
}

TEST(natural_merge_sort_test, equal_keys_keep_their_original_order)
{
    std::vector<std::pair<int, int>> items;
    for(int i = 0; i < 40; ++i)
    {
        items.emplace_back((i * 7) % 5, i);
    }

    natural_merge_sort(items.begin(), items.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });

    for(size_t i = 1; i < items.size(); ++i)
    {
        ASSERT_LE(items[i - 1].first, items[i].first);
        if(items[i - 1].first == items[i].first)
        {
            EXPECT_LT(items[i - 1].second, items[i].second) << "unstable at " << i;
        }
    }
}

TEST(natural_merge_sort_test, only_the_given_subrange_is_sorted)
{
    std::vector<int> values{ 9, 5, 4, 3, 0 };

    natural_merge_sort(values.begin() + 1, values.begin() + 4, std::less<>{});

    EXPECT_THAT(values, ::testing::ElementsAre(9, 3, 4, 5, 0));
}

TEST(natural_merge_sort_test, a_custom_comparison_orders_descending)
{
    std::vector<int> values{ 1, 3, 2 };

    natural_merge_sort(values.begin(), values.end(), std::greater<>{});

    EXPECT_THAT(values, ::testing::ElementsAre(3, 2, 1));
}

TEST(natural_merge_sort_test, random_inputs_match_stable_sort)
{
    std::mt19937 generator{ 12345 };

    for(size_t size = 0; size < 200; ++size)
    {
        std::vector<int> values(size);
        for(auto& value : values)
        {
            value = static_cast<int>(generator() % 50);
        }
        auto expected = values;
        std::stable_sort(expected.begin(), expected.end());

        EXPECT_EQ(sorted_copy(values), expected) << "size " << size;
    }
}

}  // namespace
