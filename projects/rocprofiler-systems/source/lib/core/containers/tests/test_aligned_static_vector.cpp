// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/containers/aligned_static_vector.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace
{
using rocprofsys::container::aligned_static_vector;

constexpr bool
check_push_and_size()
{
    aligned_static_vector<int, 4> vec;
    vec.push_back(1);
    vec.push_back(2);
    return vec.size() == 2 && vec[0] == 1 && vec[1] == 2;
}

constexpr bool
check_pop_and_clear()
{
    aligned_static_vector<int, 4> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.pop_back();
    const bool ok_after_pop = (vec.size() == 1);
    vec.clear();
    return ok_after_pop && vec.empty();
}

static_assert(aligned_static_vector<int, 4>{}.capacity() == 4,
              "capacity is a compile-time constant");
static_assert(check_push_and_size(), "push_back/size must work at compile time");
static_assert(check_pop_and_clear(), "pop_back/clear must work at compile time");
}  // namespace

TEST(AlignedStaticVector, default_constructed_is_empty)
{
    aligned_static_vector<int, 4> vec;
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0u);
    EXPECT_EQ(vec.capacity(), 4u);
}

TEST(AlignedStaticVector, fill_constructor)
{
    constexpr size_t                       k_capacity   = 5;
    constexpr int                          k_fill_value = 9;
    aligned_static_vector<int, k_capacity> vec(3, k_fill_value);
    EXPECT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[0], k_fill_value);
    EXPECT_EQ(vec[2], k_fill_value);
}

TEST(AlignedStaticVector, push_back_and_emplace_back)
{
    aligned_static_vector<int, 4> vec;
    vec.push_back(1);
    vec.emplace_back(2);
    EXPECT_EQ(vec.size(), 2u);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
}

TEST(AlignedStaticVector, emplace_back_throws_when_full)
{
    aligned_static_vector<int, 2> vec;
    vec.push_back(1);
    vec.push_back(2);
    EXPECT_THROW(vec.push_back(3), std::out_of_range);
    EXPECT_EQ(vec.size(), 2u);
}

TEST(AlignedStaticVector, pop_back_and_clear)
{
    aligned_static_vector<int, 4> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.pop_back();
    EXPECT_EQ(vec.size(), 1u);

    vec.clear();
    EXPECT_TRUE(vec.empty());
}

TEST(AlignedStaticVector, at_bounds_checked)
{
    // at() bounds-checks against the fixed capacity N (delegates to std::array::at),
    // not the logical size(); only an index >= N is out of range.
    aligned_static_vector<int, 4> vec;
    vec.push_back(1);
    EXPECT_EQ(vec.at(0), 1);
    EXPECT_THROW(vec.at(4), std::out_of_range);
}

TEST(AlignedStaticVector, front_and_back)
{
    aligned_static_vector<int, 4> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    EXPECT_EQ(vec.front(), 1);
    EXPECT_EQ(vec.back(), 3);
}

TEST(AlignedStaticVector, initializer_list_assignment)
{
    aligned_static_vector<int, 4> vec;
    vec = { 1, 2, 3 };
    EXPECT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[2], 3);
}

TEST(AlignedStaticVector, initializer_list_assignment_throws_on_overflow)
{
    aligned_static_vector<int, 2> vec;
    EXPECT_THROW((vec = { 1, 2, 3 }), std::out_of_range);
}

TEST(AlignedStaticVector, iteration)
{
    aligned_static_vector<int, 4> vec;
    vec = { 1, 2, 3 };

    int sum = 0;
    for(auto elem : vec)
    {
        sum += elem;
    }
    constexpr int k_expected_sum = 6;
    EXPECT_EQ(sum, k_expected_sum);
}

TEST(AlignedStaticVector, swap)
{
    constexpr int k_third_value = 5;

    aligned_static_vector<int, 4> vec_a;
    aligned_static_vector<int, 4> vec_b;
    vec_a = { 1, 2 };
    vec_b = { 3, 4, k_third_value };

    vec_a.swap(vec_b);
    EXPECT_EQ(vec_a.size(), 3u);
    EXPECT_EQ(vec_b.size(), 2u);
    EXPECT_EQ(vec_a[0], 3);
    EXPECT_EQ(vec_b[0], 1);
}

TEST(AlignedStaticVector, alignment_is_honored)
{
    constexpr size_t k_alignment = 64;

    aligned_static_vector<int, 4, k_alignment> vec;
    vec.push_back(1);
    vec.push_back(2);
    auto addr0 = reinterpret_cast<std::uintptr_t>(&vec[0]);
    auto addr1 = reinterpret_cast<std::uintptr_t>(&vec[1]);
    EXPECT_EQ(addr0 % k_alignment, 0u);
    EXPECT_EQ(addr1 % k_alignment, 0u);
}

TEST(AlignedStaticVector, atomic_size_variant_is_usable)
{
    aligned_static_vector<int, 4, alignof(int), true> vec;
    vec.push_back(1);
    vec.push_back(2);
    EXPECT_EQ(vec.size(), 2u);
    EXPECT_EQ(vec[0], 1);
}
