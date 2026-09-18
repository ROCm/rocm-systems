// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/containers/static_vector.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

namespace
{
using rocprofsys::container::static_vector;

constexpr bool
check_push_and_size()
{
    static_vector<int, 4> vec;
    vec.push_back(1);
    vec.push_back(2);
    return vec.size() == 2 && vec[0] == 1 && vec[1] == 2;
}

constexpr bool
check_fill_ctor()
{
    constexpr int         k_fill_value = 7;
    static_vector<int, 3> vec(3, k_fill_value);
    return vec.size() == 3 && vec[0] == k_fill_value && vec[2] == k_fill_value;
}

constexpr bool
check_pop_and_clear()
{
    static_vector<int, 4> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.pop_back();
    const bool ok_after_pop = (vec.size() == 1);
    vec.clear();
    return ok_after_pop && vec.empty();
}

static_assert(static_vector<int, 4>{}.capacity() == 4,
              "capacity is a compile-time constant");
static_assert(check_push_and_size(), "push_back/size must work at compile time");
static_assert(check_fill_ctor(), "fill constructor must work at compile time");
static_assert(check_pop_and_clear(), "pop_back/clear must work at compile time");

// Mirrors the "consteval factory returning a populated fixed-capacity container" shape
// used by SDK domain-collection code: a consteval function builds a static_vector whose
// contents depend on a compile-time-only condition (here, a template version gate).
struct domain_descriptor
{
    int id      = 0;
    int version = 0;
};

template <int VersionMajor>
struct domain_collector
{
    consteval static auto collect()
    {
        constexpr size_t                                k_max_domains = 8;
        static_vector<domain_descriptor, k_max_domains> result;

        result.push_back(domain_descriptor{ .id = 1, .version = 0 });
        result.push_back(domain_descriptor{ .id = 2, .version = 0 });

        if constexpr(VersionMajor >= 2)
        {
            result.push_back(domain_descriptor{ .id = 3, .version = 2 });
            result.push_back(domain_descriptor{ .id = 4, .version = 2 });
        }

        return result;
    }
};

static_assert(domain_collector<1>::collect().size() == 2,
              "consteval factory omits entries gated behind a higher version");
static_assert(domain_collector<2>::collect().size() == 4,
              "consteval factory includes entries gated at/above the version threshold");
static_assert(domain_collector<2>::collect()[2].id == 3,
              "consteval factory preserves push_back order");
static_assert(domain_collector<2>::collect()[3].version == 2,
              "consteval factory preserves per-entry field values");
}  // namespace

TEST(StaticVector, default_constructed_is_empty)
{
    static_vector<int, 4> vec;
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0u);
    EXPECT_EQ(vec.capacity(), 4u);
}

TEST(StaticVector, fill_constructor)
{
    constexpr size_t               k_capacity   = 5;
    constexpr int                  k_fill_value = 9;
    static_vector<int, k_capacity> vec(3, k_fill_value);
    EXPECT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[0], k_fill_value);
    EXPECT_EQ(vec[2], k_fill_value);
}

TEST(StaticVector, push_back_and_emplace_back)
{
    static_vector<int, 4> vec;
    vec.push_back(1);
    vec.emplace_back(2);
    EXPECT_EQ(vec.size(), 2u);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
}

TEST(StaticVector, emplace_back_throws_when_full)
{
    static_vector<int, 2> vec;
    vec.push_back(1);
    vec.push_back(2);
    EXPECT_THROW(vec.push_back(3), std::out_of_range);
    EXPECT_EQ(vec.size(), 2u);
}

TEST(StaticVector, pop_back_and_clear)
{
    static_vector<int, 4> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.pop_back();
    EXPECT_EQ(vec.size(), 1u);

    vec.clear();
    EXPECT_TRUE(vec.empty());
}

TEST(StaticVector, at_bounds_checked)
{
    // at() bounds-checks against the fixed capacity N (delegates to std::array::at),
    // not the logical size(); only an index >= N is out of range.
    static_vector<int, 4> vec;
    vec.push_back(1);
    EXPECT_EQ(vec.at(0), 1);
    EXPECT_THROW(vec.at(4), std::out_of_range);
}

TEST(StaticVector, front_and_back)
{
    static_vector<int, 4> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    EXPECT_EQ(vec.front(), 1);
    EXPECT_EQ(vec.back(), 3);
}

TEST(StaticVector, initializer_list_assignment)
{
    static_vector<int, 4> vec;
    vec = { 1, 2, 3 };
    EXPECT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[2], 3);
}

TEST(StaticVector, initializer_list_assignment_throws_on_overflow)
{
    static_vector<int, 2> vec;
    EXPECT_THROW((vec = { 1, 2, 3 }), std::out_of_range);
}

TEST(StaticVector, iteration)
{
    static_vector<int, 4> vec;
    vec = { 1, 2, 3 };

    int sum = 0;
    for(auto elem : vec)
    {
        sum += elem;
    }
    constexpr int k_expected_sum = 6;
    EXPECT_EQ(sum, k_expected_sum);
}

TEST(StaticVector, swap)
{
    constexpr int k_third_value = 5;

    static_vector<int, 4> vec_a;
    static_vector<int, 4> vec_b;
    vec_a = { 1, 2 };
    vec_b = { 3, 4, k_third_value };

    vec_a.swap(vec_b);
    EXPECT_EQ(vec_a.size(), 3u);
    EXPECT_EQ(vec_b.size(), 2u);
    EXPECT_EQ(vec_a[0], 3);
    EXPECT_EQ(vec_b[0], 1);
}

TEST(StaticVector, copy_and_move)
{
    static_vector<int, 4> vec_a;
    vec_a = { 1, 2, 3 };

    const static_vector<int, 4> copy = vec_a;
    EXPECT_EQ(copy.size(), 3u);

    const static_vector<int, 4> moved = vec_a;
    EXPECT_EQ(moved.size(), 3u);
}

TEST(StaticVector, consteval_factory_populates_static_vector)
{
    constexpr auto k_low_version_domains  = domain_collector<1>::collect();
    constexpr auto k_high_version_domains = domain_collector<2>::collect();

    EXPECT_EQ(k_low_version_domains.size(), 2u);
    EXPECT_EQ(k_high_version_domains.size(), 4u);
    EXPECT_EQ(k_high_version_domains[0].id, 1);
    EXPECT_EQ(k_high_version_domains[2].id, 3);
    EXPECT_EQ(k_high_version_domains[3].version, 2);
}

TEST(StaticVector, atomic_size_variant_is_usable)
{
    static_vector<int, 4, true> vec;
    vec.push_back(1);
    vec.push_back(2);
    EXPECT_EQ(vec.size(), 2u);
    EXPECT_EQ(vec[0], 1);
}
