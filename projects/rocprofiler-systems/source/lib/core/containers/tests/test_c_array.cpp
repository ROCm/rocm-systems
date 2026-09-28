// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/containers/c_array.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <stdexcept>

namespace
{
using rocprofsys::container::c_array;
using rocprofsys::container::wrap_c_array;

constexpr size_t                      k_raw_size = 5;
constexpr std::array<int, k_raw_size> k_raw      = { 1, 2, 3, 4, 5 };

constexpr bool
check_index_access()
{
    c_array<const int> arr{ k_raw.data(), k_raw_size };
    return arr.size() == k_raw_size && arr[0] == 1 && arr[4] == k_raw.back();
}

constexpr bool
check_pop_front_back()
{
    c_array<const int> arr{ k_raw.data(), k_raw_size };
    arr.pop_front();
    arr.pop_back();
    return arr.size() == 3 && arr[0] == 2 && arr[2] == 4;
}

constexpr bool
check_slice()
{
    c_array<const int> arr{ k_raw.data(), k_raw_size };
    auto               sliced = arr.slice(1, 4);
    return sliced.size() == 3 && sliced[0] == 2 && sliced[2] == 4;
}

static_assert(check_index_access(), "c_array index access must work at compile time");
static_assert(check_pop_front_back(),
              "c_array pop_front/pop_back must work at compile time");
static_assert(check_slice(), "c_array slice must work at compile time");
}  // namespace

TEST(CArray, size_and_index)
{
    constexpr size_t        k_size   = 4;
    constexpr int           k_first  = 10;
    constexpr int           k_second = 20;
    constexpr int           k_third  = 30;
    constexpr int           k_fourth = 40;
    std::array<int, k_size> data{ k_first, k_second, k_third, k_fourth };
    c_array<int>            arr{ data.data(), data.size() };

    EXPECT_EQ(arr.size(), k_size);
    EXPECT_EQ(arr[0], k_first);
    EXPECT_EQ(arr[3], k_fourth);
}

TEST(CArray, at_bounds_checked)
{
    constexpr size_t        k_size = 3;
    std::array<int, k_size> data{ 1, 2, 3 };
    c_array<int>            arr{ data.data(), data.size() };

    EXPECT_EQ(arr.at(2), 3);
    EXPECT_THROW(arr.at(3), std::out_of_range);
}

TEST(CArray, mutation_through_operator_index)
{
    constexpr size_t        k_size = 2;
    std::array<int, k_size> data{ 1, 2 };
    c_array<int>            arr{ data.data(), data.size() };

    constexpr int k_new_value = 42;
    arr[0]                    = k_new_value;
    EXPECT_EQ(data[0], k_new_value);
}

TEST(CArray, pop_front_and_back)
{
    constexpr size_t        k_size       = 5;
    constexpr int           k_last_value = 5;
    std::array<int, k_size> data{ 1, 2, 3, 4, k_last_value };
    c_array<int>            arr{ data.data(), data.size() };

    arr.pop_front();
    arr.pop_back();

    EXPECT_EQ(arr.size(), 3u);
    EXPECT_EQ(arr[0], 2);
    EXPECT_EQ(arr[2], 4);
}

TEST(CArray, slice)
{
    constexpr size_t        k_size       = 5;
    constexpr int           k_last_value = 5;
    std::array<int, k_size> data{ 1, 2, 3, 4, k_last_value };
    c_array<int>            arr{ data.data(), data.size() };

    auto sliced = arr.slice(1, 4);
    EXPECT_EQ(sliced.size(), 3u);
    EXPECT_EQ(sliced[0], 2);
    EXPECT_EQ(sliced[2], 4);
}

TEST(CArray, range_based_for)
{
    constexpr size_t        k_size = 4;
    std::array<int, k_size> data{ 1, 2, 3, 4 };
    const c_array<int>      arr{ data.data(), data.size() };

    int sum = 0;
    for(auto value : arr)
    {
        sum += value;
    }

    EXPECT_EQ(sum, 10);
}

TEST(CArray, wrap_c_array_deduces_type)
{
    constexpr size_t           k_size   = 3;
    constexpr double           k_first  = 1.5;
    constexpr double           k_second = 2.5;
    constexpr double           k_third  = 3.5;
    std::array<double, k_size> data{ k_first, k_second, k_third };
    auto                       arr = wrap_c_array(data.data(), data.size());

    EXPECT_EQ(arr.size(), k_size);
    EXPECT_DOUBLE_EQ(arr[1], 2.5);
}

TEST(CArray, implicit_pointer_conversion)
{
    constexpr size_t        k_size   = 2;
    constexpr int           k_first  = 7;
    constexpr int           k_second = 8;
    std::array<int, k_size> data{ k_first, k_second };
    const c_array<int>      arr{ data.data(), data.size() };

    const int* raw_ptr = arr;
    EXPECT_EQ(raw_ptr, data.data());
}
