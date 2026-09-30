// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/containers/stable_vector.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace
{
using rocprofsys::container::stable_vector;

// Small chunk size so growth across multiple chunks is exercised without huge fixtures.
using small_chunked_vector = stable_vector<int, 4>;
}  // namespace

TEST(StableVector, default_constructed_is_empty)
{
    const small_chunked_vector vec;
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0u);
}

TEST(StableVector, count_constructor)
{
    constexpr int        k_fill_value = 7;
    small_chunked_vector vec(3, k_fill_value);
    EXPECT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[0], k_fill_value);
    EXPECT_EQ(vec[2], k_fill_value);
}

TEST(StableVector, initializer_list_constructor)
{
    small_chunked_vector vec{ 1, 2, 3 };
    EXPECT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[1], 2);
}

TEST(StableVector, push_back_grows_across_chunks)
{
    constexpr int k_count = 10;

    small_chunked_vector vec;
    for(int i = 0; i < k_count; ++i)
    {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 10u);
    for(int i = 0; i < k_count; ++i)
    {
        EXPECT_EQ(vec[static_cast<size_t>(i)], i);
    }
}

TEST(StableVector, emplace_back)
{
    small_chunked_vector vec;
    vec.emplace_back(1);
    vec.emplace_back(2);
    EXPECT_EQ(vec.size(), 2u);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
}

TEST(StableVector, at_bounds_checked)
{
    small_chunked_vector vec{ 1, 2, 3 };
    EXPECT_EQ(vec.at(2), 3);
    EXPECT_THROW(vec.at(3), std::out_of_range);
}

TEST(StableVector, front_and_back)
{
    small_chunked_vector vec{ 1, 2, 3 };
    EXPECT_EQ(vec.front(), 1);
    EXPECT_EQ(vec.back(), 3);
}

TEST(StableVector, reserve_increases_capacity)
{
    constexpr size_t k_min_capacity = 9;

    small_chunked_vector vec;
    vec.reserve(k_min_capacity);
    EXPECT_GE(vec.capacity(), k_min_capacity);
}

TEST(StableVector, pointer_stability_across_growth)
{
    constexpr int k_count = 20;

    small_chunked_vector vec;
    vec.push_back(1);
    const int* first_ptr = &vec[0];

    for(int i = 0; i < k_count; ++i)
    {
        vec.push_back(i);
    }

    EXPECT_EQ(first_ptr, &vec[0]);
}

TEST(StableVector, iteration_and_equality)
{
    constexpr int k_fill_value = 5;

    small_chunked_vector       vec_a{ 1, 2, 3, 4, k_fill_value };
    const small_chunked_vector vec_b{ 1, 2, 3, 4, k_fill_value };

    EXPECT_TRUE(vec_a == vec_b);

    const int sum = std::accumulate(vec_a.begin(), vec_a.end(), 0);
    EXPECT_EQ(sum, 15);
}

TEST(StableVector, copy_constructor_is_deep)
{
    small_chunked_vector vec_a{ 1, 2, 3 };
    small_chunked_vector vec_b       = vec_a;
    constexpr int        k_new_value = 42;
    vec_b[0]                         = k_new_value;

    EXPECT_EQ(vec_a[0], 1);
    EXPECT_EQ(vec_b[0], k_new_value);
}

TEST(StableVector, move_constructor_transfers_storage)
{
    small_chunked_vector vec_a{ 1, 2, 3 };
    small_chunked_vector vec_b = std::move(vec_a);

    EXPECT_EQ(vec_b.size(), 3u);
    EXPECT_EQ(vec_b[0], 1);
}

TEST(StableVector, swap)
{
    constexpr int k_third_value = 5;

    small_chunked_vector vec_a{ 1, 2 };
    small_chunked_vector vec_b{ 3, 4, k_third_value };

    swap(vec_a, vec_b);
    EXPECT_EQ(vec_a.size(), 3u);
    EXPECT_EQ(vec_b.size(), 2u);
}

TEST(StableVector, resize_helper_grows_and_default_constructs)
{
    constexpr size_t k_resize_target = 6;

    small_chunked_vector vec;
    auto                 new_size = resize(vec, k_resize_target);
    EXPECT_EQ(new_size, k_resize_target);
    EXPECT_EQ(vec.size(), k_resize_target);
}
